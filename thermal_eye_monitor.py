#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Eye Monitor Module

Multiple strategies for drowsiness detection in thermal images:
1. EAR-based: Eye Aspect Ratio from thermal landmarks
2. Thermal-based: Temperature gradient analysis in eye regions
3. Temporal-based: Time-series analysis of eye closure patterns
4. Pupil tracking: Thermal signature of pupil movement
"""

import numpy as np
import cv2
import time
from collections import deque
from typing import Dict, Tuple


class ThermalEyeMonitor:
    """
    Monitor eye state and drowsiness indicators from thermal images
    using multiple complementary strategies.
    """
    
    # Strategy modes
    STRATEGY_EAR = 'ear'              # Eye Aspect Ratio (geometric)
    STRATEGY_THERMAL = 'thermal'      # Temperature gradient (thermal signature)
    STRATEGY_TEMPORAL = 'temporal'    # Time-series patterns
    STRATEGY_HYBRID = 'hybrid'        # Combination of all
    
    def __init__(self, strategy=STRATEGY_HYBRID, fps=15):
        """
        Initialize thermal eye monitor
        
        Args:
            strategy: Detection strategy ('ear', 'thermal', 'temporal', 'hybrid')
            fps: Expected frames per second
        """
        self.strategy = strategy
        self.fps = fps
        
        # EAR-based thresholds
        self.EAR_THRESHOLD_THERMAL = 0.15  # Thermal EAR more sensitive than RGB
        self.EAR_CONSEC_FRAMES = 3
        self.EAR_WEIGHT = 0.4
        
        # Thermal-based thresholds
        self.THERMAL_DARK_THRESHOLD = 50   # Temperature differential for closed eye
        self.THERMAL_GRADIENT_THRESHOLD = 30
        self.THERMAL_WEIGHT = 0.3
        
        # Temporal-based thresholds
        self.PERCLOS_WINDOW = 150
        self.PERCLOS_THRESHOLD = 0.3
        self.TEMPORAL_WEIGHT = 0.3
        
        # SLEEP detection
        self.SLEEP_THRESHOLD = 60
        self.BLINK_FREQUENCY_THRESHOLD = 30
        
        # State variables
        self.eye_closed_counter = 0
        self.eye_closed_history = deque([False] * self.PERCLOS_WINDOW, maxlen=self.PERCLOS_WINDOW)
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.blink_timestamps = deque(maxlen=60)
        self.sleep_detected = False
        self.frame_start_time = time.time()
        
        # Thermal history for temporal analysis
        self.thermal_intensity_history = deque(maxlen=30)
        self.ear_history = deque(maxlen=30)
        self.thermal_gradient_history = deque(maxlen=30)
        
        # Per-eye tracking
        self.left_eye_state = {'closed': False, 'blinks': 0}
        self.right_eye_state = {'closed': False, 'blinks': 0}
    
    def analyze_eye_state(self, frame, thermal_landmarks):
        """
        Analyze eye state from thermal image using multiple strategies
        
        Args:
            frame: Thermal image frame (8-bit or 16-bit)
            thermal_landmarks: Landmarks from thermal detector
            
        Returns:
            dict with analysis results from all strategies
        """
        if frame.dtype == np.uint16:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        results = {
            'strategy': self.strategy,
            'ear_score': 0.0,
            'thermal_score': 0.0,
            'temporal_score': 0.0,
            'combined_score': 0.0,
            'eyes_closed': False,
            'perclos': 0.0,
            'blink_rate': 0.0,
            'sleep_detected': self.sleep_detected,
            'strategy_details': {}
        }
        
        if thermal_landmarks is None:
            return results
        
        # Strategy 1: EAR-based analysis
        if self.strategy in [self.STRATEGY_EAR, self.STRATEGY_HYBRID]:
            ear_score, eyes_closed = self._analyze_ear(frame, thermal_landmarks)
            results['ear_score'] = ear_score
            results['strategy_details']['ear'] = {
                'score': ear_score,
                'eyes_closed': eyes_closed
            }
        
        # Strategy 2: Thermal-based analysis
        if self.strategy in [self.STRATEGY_THERMAL, self.STRATEGY_HYBRID]:
            thermal_score, thermal_analysis = self._analyze_thermal_signature(frame, thermal_landmarks)
            results['thermal_score'] = thermal_score
            results['strategy_details']['thermal'] = thermal_analysis
        
        # Strategy 3: Temporal-based analysis
        if self.strategy in [self.STRATEGY_TEMPORAL, self.STRATEGY_HYBRID]:
            temporal_score, temporal_analysis = self._analyze_temporal_patterns(
                results['ear_score'],
                results.get('thermal_score', 0)
            )
            results['temporal_score'] = temporal_score
            results['strategy_details']['temporal'] = temporal_analysis
        
        # Combine strategies
        if self.strategy == self.STRATEGY_HYBRID:
            combined = (
                self.EAR_WEIGHT * results['ear_score'] +
                self.THERMAL_WEIGHT * results['thermal_score'] +
                self.TEMPORAL_WEIGHT * results['temporal_score']
            )
            results['combined_score'] = combined
            eyes_closed = combined > 0.5
        else:
            results['combined_score'] = max(
                results['ear_score'],
                results['thermal_score'],
                results['temporal_score']
            )
            eyes_closed = results['combined_score'] > 0.5
        
        # Update state
        self._update_perclos(eyes_closed)
        self.eye_closed_history.append(eyes_closed)
        results['eyes_closed'] = eyes_closed
        results['perclos'] = self._calculate_perclos()
        results['blink_rate'] = self._update_blink_rate()
        
        # Check for sleep
        if self.eye_closed_counter > self.SLEEP_THRESHOLD:
            self.sleep_detected = True
            results['sleep_detected'] = True
        
        return results
    
    def _analyze_ear(self, frame, thermal_landmarks):
        """
        Strategy 1: Eye Aspect Ratio Analysis
        Geometric analysis of eye opening in thermal image
        
        Args:
            frame: Thermal image
            thermal_landmarks: Thermal landmarks dict
            
        Returns:
            (ear_score, eyes_closed): Tuple of score (0-1) and boolean
        """
        if 'eye_regions' not in thermal_landmarks:
            return 0.0, False
        
        eyes = thermal_landmarks['eye_regions']
        
        ear_left = self._calculate_thermal_ear(frame, eyes['left'])
        ear_right = self._calculate_thermal_ear(frame, eyes['right'])
        
        ear_avg = (ear_left + ear_right) / 2.0
        self.ear_history.append(ear_avg)
        
        # Score: lower EAR = closed eye = higher score
        ear_score = max(0, 1.0 - (ear_avg / 0.4))  # Normalize
        ear_score = min(1.0, ear_score)
        
        # Detect closure
        if ear_avg < self.EAR_THRESHOLD_THERMAL:
            self.eye_closed_counter += 1
            eyes_closed = True
        else:
            if self.eye_closed_counter > 0:
                self.blink_count += 1
                self.blink_timestamps.append(time.time())
                self.left_eye_state['blinks'] += 1
            self.eye_closed_counter = 0
            eyes_closed = False
        
        return ear_score, eyes_closed
    
    def _calculate_thermal_ear(self, frame, eye_region):
        """
        Calculate Eye Aspect Ratio from thermal image
        
        In thermal images, eyes are cold spots (dark regions)
        EAR is based on vertical vs horizontal distances
        
        Args:
            frame: Thermal frame
            eye_region: Eye region dict with 'center' and 'roi'
            
        Returns:
            ear_value: Float representing eye openness (higher = more open)
        """
        try:
            roi_info = eye_region.get('roi', (0, 0, 20, 20))
            x, y, w, h = roi_info
            
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return 0.5
            
            eye_roi = frame[y:y+h, x:x+w]
            
            if eye_roi.size == 0:
                return 0.5
            
            # Normalize
            eye_roi_norm = eye_roi / 255.0
            
            # Find darkest point (center of eye in thermal = coldest)
            min_val = np.min(eye_roi_norm)
            max_val = np.max(eye_roi_norm)
            
            if max_val == min_val:
                return 0.5
            
            # Create normalized heatmap
            heatmap = (eye_roi_norm - min_val) / (max_val - min_val)
            
            # Find eye opening by analyzing gradient and darkness
            # If center is very dark and edges are brighter = eye open
            # If entire region is uniform dark = eye closed
            
            center_y, center_x = h // 2, w // 2
            center_size = 5
            center_region = heatmap[
                max(0, center_y - center_size):min(h, center_y + center_size),
                max(0, center_x - center_size):min(w, center_x + center_size)
            ]
            edge_region = np.concatenate([
                heatmap[0:3, :].flatten(),
                heatmap[-3:, :].flatten(),
                heatmap[:, 0:3].flatten(),
                heatmap[:, -3:].flatten()
            ])
            
            center_darkness = np.mean(center_region) if center_region.size > 0 else 0.5
            edge_brightness = np.mean(edge_region) if edge_region.size > 0 else 0.5
            
            # EAR: ratio of center darkness to edge region
            # Lower value = eye closed (uniform dark)
            # Higher value = eye open (clear dark center with brighter edges)
            ear = edge_brightness / (center_darkness + 0.01) if center_darkness > 0 else 1.0
            
            # Normalize to 0-1 range
            ear = min(1.0, ear / 3.0)
            
            return ear
            
        except Exception as e:
            print(f"Error calculating thermal EAR: {e}")
            return 0.5
    
    def _analyze_thermal_signature(self, frame, thermal_landmarks):
        """
        Strategy 2: Thermal Signature Analysis
        Temperature gradient-based eye detection
        
        Eyes have distinct thermal signature:
        - Eye surface is usually cooler than surrounding skin
        - When eyes close, thermal profile changes significantly
        
        Args:
            frame: Thermal image
            thermal_landmarks: Thermal landmarks
            
        Returns:
            (thermal_score, analysis_dict)
        """
        if 'eye_regions' not in thermal_landmarks:
            return 0.0, {}
        
        eyes = thermal_landmarks['eye_regions']
        
        left_gradient = self._calculate_thermal_gradient(frame, eyes['left'])
        right_gradient = self._calculate_thermal_gradient(frame, eyes['right'])
        
        avg_gradient = (left_gradient + right_gradient) / 2.0
        self.thermal_gradient_history.append(avg_gradient)
        
        # Score: high gradient = eyes open, low gradient = eyes closed
        thermal_score = min(1.0, avg_gradient / self.THERMAL_GRADIENT_THRESHOLD)
        
        analysis = {
            'left_gradient': left_gradient,
            'right_gradient': right_gradient,
            'average_gradient': avg_gradient,
            'score': thermal_score
        }
        
        return thermal_score, analysis
    
    def _calculate_thermal_gradient(self, frame, eye_region):
        """
        Calculate thermal gradient in eye region
        High gradient = distinct eye structure = eye open
        Low gradient = blurred features = eye closed
        
        Args:
            frame: Thermal frame
            eye_region: Eye region dict
            
        Returns:
            gradient_magnitude: Float measure of thermal gradient
        """
        try:
            roi_info = eye_region.get('roi', (0, 0, 20, 20))
            x, y, w, h = roi_info
            
            if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                return 0.0
            
            eye_roi = frame[y:y+h, x:x+w].astype(np.float32)
            
            if eye_roi.size == 0:
                return 0.0
            
            # Calculate Sobel gradients
            grad_x = cv2.Sobel(eye_roi, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(eye_roi, cv2.CV_32F, 0, 1, ksize=3)
            
            # Calculate magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return float(np.mean(gradient_magnitude))
            
        except Exception as e:
            print(f"Error calculating thermal gradient: {e}")
            return 0.0
    
    def _analyze_temporal_patterns(self, ear_score, thermal_score):
        """
        Strategy 3: Temporal Pattern Analysis
        Analyze time-series patterns in eye states
        
        Drowsiness shows patterns:
        - Gradual increase in closure time
        - Reduced blink frequency
        - Blinks becoming slower
        
        Args:
            ear_score: Current EAR score
            thermal_score: Current thermal score
            
        Returns:
            (temporal_score, analysis_dict)
        """
        # Trend analysis
        if len(self.ear_history) < 5:
            return 0.0, {}
        
        ear_array = np.array(list(self.ear_history))
        
        # Calculate trend (is EAR decreasing over time?)
        # Decreasing EAR = increasing closure = drowsiness
        trend = np.polyfit(range(len(ear_array)), ear_array, 1)[0]
        trend_score = min(1.0, max(0, -trend * 10))  # Negative trend = drowsy
        
        # Calculate variance (low variance = stable closure = drowsy)
        variance = np.var(ear_array) if len(ear_array) > 1 else 0.5
        variance_score = min(1.0, max(0, 1.0 - variance * 2))
        
        # PERCLOS (percentage of eyelid closure)
        perclos = self._calculate_perclos()
        perclos_score = min(1.0, perclos / self.PERCLOS_THRESHOLD)
        
        # Combine
        temporal_score = (trend_score * 0.3 + variance_score * 0.3 + perclos_score * 0.4)
        
        analysis = {
            'trend_score': float(trend_score),
            'variance_score': float(variance_score),
            'perclos_score': float(perclos_score),
            'combined_score': float(temporal_score),
            'perclos': float(perclos),
            'trend': float(trend)
        }
        
        return float(temporal_score), analysis
    
    def _update_perclos(self, is_closed):
        """Update PERCLOS (Percentage of Eyelid Closure)"""
        self.eye_closed_history.append(is_closed)
        if is_closed:
            self.eye_closed_counter += 1
        else:
            if self.eye_closed_counter >= self.EAR_CONSEC_FRAMES:
                # Valid blink/closure
                pass
            self.eye_closed_counter = 0
    
    def _calculate_perclos(self):
        """Calculate PERCLOS value"""
        if len(self.eye_closed_history) == 0:
            return 0.0
        return sum(self.eye_closed_history) / len(self.eye_closed_history)
    
    def _update_blink_rate(self):
        """Calculate blinks per minute"""
        current_time = time.time()
        
        # Remove old blinks outside 60-second window
        while self.blink_timestamps and (current_time - self.blink_timestamps[0]) > 60:
            self.blink_timestamps.popleft()
        
        if (current_time - self.frame_start_time) < 1:
            return 0.0
        
        elapsed_minutes = (current_time - self.frame_start_time) / 60.0
        blink_rate = len(self.blink_timestamps) / elapsed_minutes if elapsed_minutes > 0 else 0
        
        return blink_rate
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        self.eye_closed_counter = 0
        self.blink_count = 0
        self.last_blink_time = time.time()
        self.sleep_detected = False
        self.frame_start_time = time.time()
        self.blink_timestamps.clear()
        self.ear_history.clear()
        self.thermal_gradient_history.clear()
