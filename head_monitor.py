#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Head Movement Monitor

This module analyzes head position and movement to detect drowsiness signs
such as nodding and improper head position.
"""

import numpy as np
import time
from collections import deque

class HeadMonitor:
    # Key face landmarks for head pose estimation
    # Nose, forehead, chin, left eye, right eye, left mouth, right mouth
    HEAD_POSE_INDICES = [1, 10, 152, 33, 263, 61, 291]
    
    def __init__(self):
        # Reference values
        self.reference_pose = None
        self.reference_frame_count = 0
        self.REFERENCE_FRAMES_NEEDED = 45  # Increased frames to establish reference pose
        
        # Thresholds (all made less sensitive)
        self.NOD_THRESHOLD = 0.20         # Increased vertical angle change threshold
        self.TILT_THRESHOLD = 0.15        # Increased horizontal angle change threshold
        self.FORWARD_THRESHOLD = 0.30     # Increased forward lean threshold
        
        # State tracking
        self.head_nods = 0
        self.nod_start_time = None
        self.nodding = False
        self.nod_timestamps = deque(maxlen=20)
        self.head_pose_history = deque(maxlen=10)  # Last 10 head poses
        self.last_stable_time = time.time()
        self.frame_start_time = time.time()
    
    def _extract_face_landmarks(self, landmarks):
        """Extract key points for head pose estimation"""
        points = []
        for idx in self.HEAD_POSE_INDICES:
            point = [
                landmarks.landmark[idx].x,
                landmarks.landmark[idx].y,
                landmarks.landmark[idx].z
            ]
            points.append(point)
        return np.array(points)
    
    def _calculate_head_pose(self, face_points):
        """
        Calculate approximate head pose (pitch, yaw, relative size)
        This is a simplified approach without full pose estimation
        """
        # Face center (nose tip)
        nose = face_points[0]
        
        # Calculate pitch (up/down) using vertical positions
        forehead = face_points[1]
        chin = face_points[2]
        pitch = np.arctan2(chin[1] - forehead[1], chin[2] - forehead[2])
        
        # Calculate yaw (left/right) using horizontal positions
        left_eye = face_points[3]
        right_eye = face_points[4]
        yaw = np.arctan2(right_eye[0] - left_eye[0], right_eye[2] - left_eye[2])
        
        # Calculate face size as distance between key points (for forward/backward)
        left_mouth = face_points[5]
        right_mouth = face_points[6]
        size = np.linalg.norm(np.array(right_mouth) - np.array(left_mouth))
        
        return {
            "pitch": pitch,
            "yaw": yaw,
            "size": size,
            "nose_position": nose
        }
    
    def _detect_nodding(self, current_pose):
        """Detect nodding motion (vertical head movement)"""
        # Need history to detect nodding
        if len(self.head_pose_history) < 3:
            return False, 0
        
        # Calculate pitch changes over recent history
        pitch_changes = [
            abs(pose["pitch"] - self.head_pose_history[i-1]["pitch"])
            for i, pose in enumerate(self.head_pose_history) if i > 0
        ]
        
        # Check for consistent downward motion followed by upward motion
        is_nodding = False
        current_time = time.time()
        
        # Detect rapid pitch changes as nodding
        if max(pitch_changes) > self.NOD_THRESHOLD:
            if not self.nodding:
                self.nodding = True
                self.nod_start_time = current_time
            elif current_time - self.nod_start_time > 1.5:
                # A complete nod takes about 1-2 seconds
                self.nodding = False
                self.head_nods += 1
                self.nod_timestamps.append(current_time)
                is_nodding = True
        else:
            # Reset if no significant movement
            if self.nodding and current_time - self.nod_start_time > 3:
                self.nodding = False
        
        # Calculate nod frequency (nods per minute)
        nod_frequency = 0
        if self.nod_timestamps:
            # Remove nods older than 1 minute
            cutoff_time = current_time - 60
            while self.nod_timestamps and self.nod_timestamps[0] < cutoff_time:
                self.nod_timestamps.popleft()
            
            # Calculate frequency
            elapsed_minutes = (current_time - self.frame_start_time) / 60
            nod_frequency = len(self.nod_timestamps) / max(elapsed_minutes, 1/60)
        
        return is_nodding, nod_frequency
    
    def _detect_improper_position(self, current_pose):
        """Detect improper head position (tilted or leaning)"""
        if not self.reference_pose:
            return False, "normal"
        
        # Check for tilted head (yaw deviation)
        yaw_diff = abs(current_pose["yaw"] - self.reference_pose["yaw"])
        is_tilted = yaw_diff > self.TILT_THRESHOLD
        
        # Check for forward/backward lean (using relative size change)
        size_ratio = current_pose["size"] / self.reference_pose["size"]
        is_leaning_forward = size_ratio > (1 + self.FORWARD_THRESHOLD)
        is_leaning_backward = size_ratio < (1 - self.FORWARD_THRESHOLD/2)
        
        # Determine position state
        position = "normal"
        if is_tilted:
            position = "tilted"
        elif is_leaning_forward:
            position = "forward"
        elif is_leaning_backward:
            position = "backward"
        
        return (is_tilted or is_leaning_forward or is_leaning_backward), position
    
    def process(self, face_landmarks):
        """
        Process face landmarks to detect head movement and position
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            dict: Status information about head position and movement
        """
        # Extract relevant face points
        face_points = self._extract_face_landmarks(face_landmarks)
        
        # Calculate head pose
        current_pose = self._calculate_head_pose(face_points)
        
        # Update history
        self.head_pose_history.append(current_pose)
        
        # Update reference pose if needed
        if self.reference_frame_count < self.REFERENCE_FRAMES_NEEDED:
            if not self.reference_pose:
                self.reference_pose = current_pose
            else:
                # Update reference by averaging
                alpha = 1.0 / (self.reference_frame_count + 1)
                self.reference_pose["pitch"] = (1-alpha) * self.reference_pose["pitch"] + alpha * current_pose["pitch"]
                self.reference_pose["yaw"] = (1-alpha) * self.reference_pose["yaw"] + alpha * current_pose["yaw"]
                self.reference_pose["size"] = (1-alpha) * self.reference_pose["size"] + alpha * current_pose["size"]
            
            self.reference_frame_count += 1
        
        # Detect head movements
        is_nodding, nod_frequency = self._detect_nodding(current_pose)
        is_improper_position, position_state = self._detect_improper_position(current_pose)
        
        # Calculate time since stable head position
        current_time = time.time()
        if not is_nodding and not is_improper_position:
            self.last_stable_time = current_time
        time_since_stable = current_time - self.last_stable_time
        
        # Determine drowsiness level based on head movement (less sensitive)
        if is_nodding and nod_frequency > 7:  # Increased from 5 to 7 nods per minute
            head_drowsiness = "high"
        elif time_since_stable > 15 and is_improper_position:  # Increased from 10 to 15 seconds
            head_drowsiness = "medium"
        elif is_nodding or (time_since_stable > 8 and is_improper_position):  # Increased from 5 to 8 seconds
            head_drowsiness = "low"
        else:
            head_drowsiness = "none"
        
        return {
            "is_nodding": is_nodding,
            "nod_frequency": nod_frequency,
            "position": position_state,
            "is_improper_position": is_improper_position,
            "time_since_stable": time_since_stable,
            "drowsiness": head_drowsiness,
            # Add head pose angles for temporal model and calibration
            "yaw": float(current_pose["yaw"]),
            "pitch": float(current_pose["pitch"]),
            "roll": 0.0  # Roll not calculated in current implementation, can be added if needed
        }