#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Yawn Detector Module

This module detects yawning based on mouth aspect ratio using MediaPipe facial landmarks.
It tracks:
- Mouth Aspect Ratio (MAR)
- Yawn frequency and duration
"""

import numpy as np
import time
from collections import deque

class YawnDetector:
    # MediaPipe landmark indices for mouth (correct indices for mouth region)
    # Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
    # Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
    # Mouth corners: 61 (left), 291 (right)
    # For MAR calculation, we need:
    # - Vertical distances: upper to lower lip at different points
    # - Horizontal distance: left to right mouth corner
    
    def __init__(self):
        # Thresholds for yawn detection
        self.MAR_THRESHOLD = 0.6       # Mouth aspect ratio threshold for yawn (adjusted for correct calculation)
        self.YAWN_DURATION_THRESHOLD = 0.8  # Seconds threshold for minimum yawn duration to count
        self.YAWN_FREQUENCY_THRESHOLD = 3   # Yawns per 1 minute threshold for Level 2 alert
        self.CONTINUOUS_YAWN_THRESHOLD = 5.0  # 5 seconds continuous yawning for Level 2 alert
        self.MIN_YAWN_FRAMES = 5  # Minimum consecutive frames with mouth open to confirm yawning
        
        # State variables
        self.yawning = False
        self.yawn_start_time = None
        self.current_yawn_duration = 0
        self.yawn_timestamps = deque(maxlen=20)  # Store last 20 yawns
        self.frame_start_time = time.time()
        self.baseline_mar = None
        self.baseline_count = 0
        self.high_mar_frame_count = 0  # Track consecutive frames with high MAR
        # Fixed-threshold mode: baseline phase disabled
        self.BASELINE_FRAMES_NEEDED = 0
    
    def _calculate_mar(self, landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR) using correct MediaPipe landmarks
        MAR = (sum of vertical distances) / (2 * horizontal distance)
        Higher MAR = wider mouth opening (yawning)
        """
        try:
            # MediaPipe mouth landmarks (correct indices)
            # Upper lip points
            upper_lip_top = landmarks.landmark[13]      # Upper lip center top
            upper_lip_left = landmarks.landmark[82]     # Upper lip left
            upper_lip_right = landmarks.landmark[312]   # Upper lip right
            
            # Lower lip points  
            lower_lip_bottom = landmarks.landmark[14]   # Lower lip center bottom
            lower_lip_left = landmarks.landmark[87]     # Lower lip left
            lower_lip_right = landmarks.landmark[317]   # Lower lip right
            
            # Mouth corners
            mouth_left = landmarks.landmark[61]         # Left corner
            mouth_right = landmarks.landmark[291]       # Right corner
            
            # Calculate vertical distances (mouth opening height at different points)
            v1 = self._calculate_distance(
                [upper_lip_top.x, upper_lip_top.y],
                [lower_lip_bottom.x, lower_lip_bottom.y]
            )
            v2 = self._calculate_distance(
                [upper_lip_left.x, upper_lip_left.y],
                [lower_lip_left.x, lower_lip_left.y]
            )
            v3 = self._calculate_distance(
                [upper_lip_right.x, upper_lip_right.y],
                [lower_lip_right.x, lower_lip_right.y]
            )
            
            # Calculate horizontal distance (mouth width)
            h = self._calculate_distance(
                [mouth_left.x, mouth_left.y],
                [mouth_right.x, mouth_right.y]
            )
            
            # MAR formula: average of vertical distances divided by horizontal
            # This increases when mouth opens (yawning)
            mar = (v1 + v2 + v3) / (3.0 * h) if h > 0 else 0
            
            return mar
            
        except (IndexError, AttributeError) as e:
            # Fallback if landmarks not available
            return 0.0
    
    def _estimate_mar_from_face_context(self, landmarks):
        """Estimate MAR using face context when mouth may be occluded"""
        # Get additional face points for context
        context_points = []
        for idx in self.FACE_CONTEXT_INDICES:
            context_points.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
        
        # Look at jaw movement and facial muscle tension
        # Chin distance from neutral position can indicate yawning
        chin_position = context_points[0]
        cheek_left = context_points[1]
        cheek_right = context_points[2]
        
        # Calculate jaw angle and cheek displacement (changes during yawning)
        jaw_width = self._calculate_distance(cheek_left, cheek_right)
        chin_to_cheek_ratio = (self._calculate_distance(chin_position, cheek_left) + 
                             self._calculate_distance(chin_position, cheek_right)) / (2 * jaw_width)
        
        # This ratio increases during a yawn as the chin moves down
        # Convert to a MAR-like scale for compatibility with existing logic
        estimated_mar = chin_to_cheek_ratio - 0.8  # Adjust baseline
        return max(0, estimated_mar)
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _calculate_yawn_frequency(self):
        """Calculate yawn frequency (yawns in last 60 seconds only)"""
        current_time = time.time()
        
        # Remove yawns older than 60 seconds - keep only recent yawns
        cutoff_time = current_time - 60.0
        
        # Clean up old timestamps
        while self.yawn_timestamps and self.yawn_timestamps[0] < cutoff_time:
            self.yawn_timestamps.popleft()
        
        # Return count of yawns in last 60 seconds (not per-minute rate)
        # This is the actual number of yawns in the sliding 60-second window
        return len(self.yawn_timestamps)
    
    def process(self, face_landmarks):
        """
        Process face landmarks to detect yawning
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            dict: Status information about yawn state
        """
        # Calculate MAR with occlusion compensation
        mar = self._calculate_mar(face_landmarks)
        
        # Current time for duration calculations
        current_time = time.time()

        # Build a baseline MAR for the first N frames (assumed non-yawning)
        if self.baseline_count < self.BASELINE_FRAMES_NEEDED:
            if self.baseline_mar is None:
                self.baseline_mar = mar
            else:
                self.baseline_mar = (self.baseline_mar * self.baseline_count + mar) / (self.baseline_count + 1)
            self.baseline_count += 1

            # During baseline collection, do not report yawning
            return {
                "mar": mar,
                "is_yawning": False,
                "yawn_duration": 0,
                "yawn_completed": False,
                "yawn_frequency": 0.0,
                "drowsiness": "none"
            }

        dynamic_threshold = self.MAR_THRESHOLD

        # Only detect yawning if MAR is significantly above threshold AND sustained
        mouth_is_open = mar > dynamic_threshold
        
        # Track consecutive frames with high MAR to avoid false positives from brief movements
        if mouth_is_open:
            self.high_mar_frame_count += 1
        else:
            self.high_mar_frame_count = 0
        
        # Only consider it yawning if mouth has been open for minimum frames
        is_yawning = self.high_mar_frame_count >= self.MIN_YAWN_FRAMES
        yawn_completed = False
        
        # Update yawn state
        if is_yawning and not self.yawning:
            # Start of a new yawn
            self.yawning = True
            self.yawn_start_time = current_time
        elif is_yawning and self.yawning:
            # Continuing yawn
            self.current_yawn_duration = current_time - self.yawn_start_time
        elif not is_yawning and self.yawning:
            # End of yawn
            self.yawning = False
            yawn_duration = current_time - self.yawn_start_time
            
            # Only count as a yawn if it lasted long enough
            if yawn_duration >= self.YAWN_DURATION_THRESHOLD:
                self.yawn_timestamps.append(current_time)
                yawn_completed = True
            
            self.current_yawn_duration = 0
        
        # Calculate yawn frequency
        yawn_frequency = self._calculate_yawn_frequency()
        
        # Determine drowsiness level based on yawning with GRADUAL scoring
        # Level 2 (medium/high) alert conditions:
        # 1. Continuous yawning for more than 5 seconds
        # 2. More than 3 yawns in 1 minute (gradual reduction as frequency drops)
        
        if self.current_yawn_duration > self.CONTINUOUS_YAWN_THRESHOLD:
            yawn_drowsiness = "high"  # Level 2+: > 5 seconds continuous yawning
        elif yawn_frequency >= self.YAWN_FREQUENCY_THRESHOLD:
            yawn_drowsiness = "medium"  # Level 2: >= 3 yawns per minute
        elif yawn_frequency == 2:
            # Exactly 2 yawns in minute - transitioning down from medium
            yawn_drowsiness = "low-medium"  # Between low and medium
        elif yawn_frequency == 1:
            # Exactly 1 yawn in the last minute but not currently yawning
            yawn_drowsiness = "low"
        elif self.yawning and self.current_yawn_duration > self.YAWN_DURATION_THRESHOLD:
            # Currently yawning (first time, not yet counted in frequency)
            yawn_drowsiness = "low"  # Level 1: Single yawn in progress
        else:
            # No yawns in last minute and not currently yawning
            yawn_drowsiness = "none"
        
        return {
            "mar": mar,
            "is_yawning": is_yawning,
            "yawn_duration": self.current_yawn_duration if self.yawning else 0,
            "yawn_completed": yawn_completed,
            "yawn_frequency": yawn_frequency,
            "drowsiness": yawn_drowsiness
        }
        
    def _check_facial_context(self, face_landmarks):
        """Check additional facial context for yawn indicators"""
        # This looks at multiple facial features beyond just mouth opening
        # 1. Jaw and chin position
        chin = face_landmarks.landmark[152]  # Chin landmark
        nose = face_landmarks.landmark[4]    # Nose tip
        
        # During a yawn, the chin-to-nose distance increases
        chin_drop = self._calculate_distance(
            [chin.x, chin.y],
            [nose.x, nose.y]
        )
        
        # 2. Cheek muscle tension (changes during yawn)
        left_cheek = face_landmarks.landmark[200]
        right_cheek = face_landmarks.landmark[429]
        cheek_distance = self._calculate_distance(
            [left_cheek.x, left_cheek.y],
            [right_cheek.x, right_cheek.y]
        )
        
        # Combine evidence (normalized values)
        # Higher values indicate stronger evidence of yawning
        context_score = (chin_drop * 3 + cheek_distance) / 4
        
        return context_score