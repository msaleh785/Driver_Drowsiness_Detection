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
    # MediaPipe landmark indices for mouth
    # Upper and lower lip points, left and right mouth corners
    MOUTH_INDICES = [13, 14, 78, 80, 81, 82, 178, 87, 88, 91, 95, 185]
    
    # Additional face landmarks for context (chin, cheeks, nose)
    # Used for detecting yawns when mouth may be partially covered
    FACE_CONTEXT_INDICES = [152, 200, 175, 58, 172, 136]
    
    def __init__(self):
        # Thresholds for yawn detection
        self.MAR_THRESHOLD = 0.65       # Mouth aspect ratio threshold for yawn
        self.YAWN_DURATION_THRESHOLD = 1.2  # Seconds threshold for yawn duration
        self.YAWN_FREQUENCY_THRESHOLD = 2   # Yawns per 1 minute threshold
        
        # State variables
        self.yawning = False
        self.yawn_start_time = None
        self.current_yawn_duration = 0
        self.yawn_timestamps = deque(maxlen=20)  # Store last 20 yawns
        self.frame_start_time = time.time()
        self.baseline_mar = None
        self.baseline_count = 0
        # Fixed-threshold mode: baseline phase disabled
        self.BASELINE_FRAMES_NEEDED = 0
    
    def _calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) with compensation for potential occlusion"""
        # Extract mouth landmarks
        points = []
        valid_points = True
        
        for idx in self.MOUTH_INDICES:
            # Check if landmark is visible (not occluded)
            # MediaPipe landmarks have a visibility score we can check
            landmark = landmarks.landmark[idx]
            points.append([landmark.x, landmark.y])
            
            # If landmark is at edge of image or has unusual position, it might be occluded
            if (landmark.x < 0.05 or landmark.x > 0.95 or 
                landmark.y < 0.05 or landmark.y > 0.95):
                valid_points = False
        
        # Calculate vertical distances (average of several points for robustness)
        v1 = self._calculate_distance(points[1], points[7])  # Center top to bottom
        v2 = self._calculate_distance(points[2], points[6])  # Left of center
        v3 = self._calculate_distance(points[3], points[5])  # Right of center
        
        # Calculate horizontal distance
        h = self._calculate_distance(points[0], points[4])  # Left to right corners
        
        # Calculate standard MAR
        mar = (v1 + v2 + v3) / (3.0 * h) if h > 0 else 0
        
        # If mouth points seem invalid (possibly occluded), use secondary method
        if not valid_points or mar > 1.5:  # Unrealistic MAR, likely occlusion
            mar = self._estimate_mar_from_face_context(landmarks)
        
        return mar
    
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
        """Calculate yawn frequency (yawns per 1-minute sliding window)"""
        current_time = time.time()
        
        # Remove yawns older than 60 seconds
        cutoff_time = current_time - 60
        while self.yawn_timestamps and self.yawn_timestamps[0] < cutoff_time:
            self.yawn_timestamps.popleft()
        
        # Yawns per minute over the last 60 seconds
        window_minutes = 1.0
        yawn_frequency = len(self.yawn_timestamps) / window_minutes
        
        return yawn_frequency
    
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

        is_yawning = mar > dynamic_threshold
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
        
        # Determine drowsiness level based on yawning
        if self.current_yawn_duration > 2 * self.YAWN_DURATION_THRESHOLD:
            yawn_drowsiness = "high"
        elif yawn_frequency > self.YAWN_FREQUENCY_THRESHOLD:
            yawn_drowsiness = "medium"
        elif self.yawning and self.current_yawn_duration > self.YAWN_DURATION_THRESHOLD:
            yawn_drowsiness = "low"
        else:
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