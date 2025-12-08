#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eye Monitor Module

This module analyzes eye aspects to detect drowsiness using MediaPipe facial landmarks.
It calculates:
- Eye Aspect Ratio (EAR)
- PERCLOS (percentage of eye closure)
- Blink frequency and duration
"""

import numpy as np
import time
from collections import deque

class EyeMonitor:
    # MediaPipe landmark indices for left and right eyes
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Upper and lower points
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Upper and lower points
    
    def __init__(self):
        # Thresholds for eye detection
        self.EAR_THRESHOLD = 0.18        # Eye aspect ratio threshold for closed eye (lowered to be less sensitive)
        self.EAR_CONSEC_FRAMES = 3       # Number of consecutive frames for eye closure
        self.PERCLOS_WINDOW = 150        # Window size for PERCLOS calculation (frames)
        self.PERCLOS_THRESHOLD = 0.35    # PERCLOS threshold for drowsiness (increased to be less sensitive)
        self.BLINK_FREQ_THRESHOLD = 35   # Blinks per minute threshold for drowsiness (increased)
        self.SLEEP_THRESHOLD = 90        # Consecutive frames with closed eyes indicates sleep (~3s at 30 FPS)
        
        # State variables
        self.eye_closed_counter = 0
        self.eye_closed_history = deque([False] * self.PERCLOS_WINDOW, maxlen=self.PERCLOS_WINDOW)
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_timestamps = deque(maxlen=60)  # Store last 60 blinks
        self.frame_start_time = time.time()
        self.blink_duration_history = deque(maxlen=20)  # Track recent blink durations
        self.sleep_detected = False
        
    def _calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio (EAR) for the given eye"""
        # Extract eye landmarks
        points = []
        for idx in eye_indices:
            points.append([landmarks.landmark[idx].x, landmarks.landmark[idx].y])
        
        # Calculate the vertical distances
        v1 = self._calculate_distance(points[1], points[5])
        v2 = self._calculate_distance(points[2], points[4])
        
        # Calculate the horizontal distance
        h = self._calculate_distance(points[0], points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h) if h > 0 else 0
        
        return ear
    
    def _calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def _update_perclos(self, is_closed):
        """Update PERCLOS (Percentage of Eyelid Closure)"""
        self.eye_closed_history.append(is_closed)
        perclos = sum(self.eye_closed_history) / len(self.eye_closed_history)
        return perclos
    
    def _update_blink_rate(self):
        """Update and calculate blink frequency (blinks per minute)"""
        current_time = time.time()
        
        # Calculate elapsed time in minutes
        elapsed_minutes = (current_time - self.frame_start_time) / 60
        
        # Clean up old blinks outside the 1-minute window
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60:
            self.blink_timestamps.popleft()
        
        # Calculate current blink rate per minute
        blink_rate = len(self.blink_timestamps) / max(elapsed_minutes, 1/60)
        
        return blink_rate
    
    def process(self, face_landmarks):
        """
        Process eye landmarks to detect drowsiness
        
        Args:
            face_landmarks: MediaPipe face landmarks
            
        Returns:
            dict: Status information about eye state
        """
        # Calculate EAR for both eyes
        left_ear = self._calculate_ear(face_landmarks, self.LEFT_EYE_INDICES)
        right_ear = self._calculate_ear(face_landmarks, self.RIGHT_EYE_INDICES)
        
        # Average EAR between both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if eyes are closed
        is_closed = avg_ear < self.EAR_THRESHOLD
        
        # Detect blinks and potential sleep
        blink_detected = False
        blink_duration = 0
        
        if is_closed:
            self.eye_closed_counter += 1
            # Check for potential sleep (eyes closed for extended period)
            if self.eye_closed_counter >= self.SLEEP_THRESHOLD:
                self.sleep_detected = True
        else:
            if self.eye_closed_counter > 0:
                # If eyes were closed and now open, it's a blink or eye reopening after longer closure
                blink_duration = self.eye_closed_counter
                
                if 3 <= self.eye_closed_counter <= 7:  # Typical blink is 3-7 frames
                    blink_detected = True
                    self.blink_count += 1
                    self.blink_timestamps.append(time.time())
                    self.blink_duration_history.append(blink_duration)
                elif self.eye_closed_counter > 7:
                    # This was a longer closure, not just a blink
                    self.blink_duration_history.append(blink_duration)
            
            self.eye_closed_counter = 0
            self.sleep_detected = False
        
        # Update PERCLOS
        perclos = self._update_perclos(is_closed)
        
        # Update blink rate
        blink_rate = self._update_blink_rate()
        
        # Calculate average blink duration (longer blinks indicate drowsiness)
        avg_blink_duration = 0
        if self.blink_duration_history:
            avg_blink_duration = sum(self.blink_duration_history) / len(self.blink_duration_history)
        
        # Analyze eye state with improved logic and less sensitivity
        if self.sleep_detected or self.eye_closed_counter >= self.SLEEP_THRESHOLD:
            eye_drowsiness = "high"  # Definite sleep or severe drowsiness
        elif perclos > self.PERCLOS_THRESHOLD:
            eye_drowsiness = "high"
        elif avg_blink_duration > 7 or blink_rate > self.BLINK_FREQ_THRESHOLD:
            eye_drowsiness = "medium"
        elif avg_blink_duration > 6:
            eye_drowsiness = "low"
        else:
            eye_drowsiness = "none"
        
        return {
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "is_closed": is_closed,
            "eye_closed_counter": self.eye_closed_counter,
            "perclos": perclos,
            "blink_rate": blink_rate,
            "blink_detected": blink_detected,
            "blink_duration": blink_duration,
            "avg_blink_duration": avg_blink_duration,
            "sleep_detected": self.sleep_detected,
            "drowsiness": eye_drowsiness
        }