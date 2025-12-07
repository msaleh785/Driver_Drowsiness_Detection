#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Yawn Detector Module

Multiple strategies for detecting yawning in thermal images:
1. MAR-based: Mouth Aspect Ratio from thermal landmarks
2. Thermal-based: Temperature signature analysis of mouth opening
3. Temporal-based: Characteristic yawn motion patterns
4. Contour-based: Mouth shape analysis from thermal silhouette
"""

import numpy as np
import cv2
import time
from collections import deque
from typing import Dict, Tuple


class ThermalYawnDetector:
    """
    Detect yawning in thermal images using multiple complementary strategies.
    Thermal images provide unique advantages:
    - Clear detection of mouth opening (thermal signature changes)
    - Jaw motion visibility
    - Interior mouth cavity thermal signature
    """
    
    # Strategy modes
    STRATEGY_MAR = 'mar'                    # Mouth Aspect Ratio (geometric)
    STRATEGY_THERMAL = 'thermal'            # Thermal signature (temperature)
    STRATEGY_TEMPORAL = 'temporal'          # Time-series analysis
    STRATEGY_CONTOUR = 'contour'           # Mouth shape analysis
    STRATEGY_HYBRID = 'hybrid'              # Combination of all
    
    def __init__(self, strategy=STRATEGY_HYBRID, fps=15):
        """
        Initialize thermal yawn detector
        
        Args:
            strategy: Detection strategy
            fps: Expected frames per second
        """
        self.strategy = strategy
        self.fps = fps
        
        # MAR-based thresholds
        self.MAR_THRESHOLD_THERMAL = 0.5  # More sensitive than RGB
        self.MAR_WEIGHT = 0.35
        
        # Thermal-based thresholds
        self.MOUTH_INTERIOR_TEMP_DROP = 15  # Thermal signature in Celsius
        self.MOUTH_OPENING_SIZE = 10        # Minimum pixels
        self.THERMAL_WEIGHT = 0.35
        
        # Temporal-based thresholds
        self.YAWN_DURATION_MIN = 0.5       # Seconds
        self.YAWN_DURATION_MAX = 5.0       # Seconds
        self.YAWN_FREQUENCY_THRESHOLD = 3   # Per 5 minutes
        self.TEMPORAL_WEIGHT = 0.3
        
        # Contour-based thresholds
        self.MIN_MOUTH_AREA = 50
        self.CONTOUR_WEIGHT = 0.3
        
        # State variables
        self.yawning = False
        self.yawn_start_time = None
        self.current_yawn_duration = 0
        self.yawn_timestamps = deque(maxlen=20)
        self.frame_start_time = time.time()
        self.yawn_count = 0
        
        # History tracking
        self.mar_history = deque(maxlen=30)
        self.mouth_opening_history = deque(maxlen=30)
        self.thermal_signature_history = deque(maxlen=30)
        self.mouth_area_history = deque(maxlen=30)
    
    def analyze_yawn_state(self, frame, thermal_landmarks, eye_closure_score=None):
        """
        Analyze yawn state from thermal image using multiple strategies
        
        Args:
            frame: Thermal image frame (8-bit or 16-bit)
            thermal_landmarks: Landmarks from thermal detector
            eye_closure_score: (Optional) Eye closure score (0-1) to filter false yawns when eyes are closed
            
        Returns:
            dict with analysis results from all strategies
        """
        if frame.dtype == np.uint16:
            frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        results = {
            'strategy': self.strategy,
            'mar_score': 0.0,
            'thermal_score': 0.0,
            'temporal_score': 0.0,
            'contour_score': 0.0,
            'combined_score': 0.0,
            'yawning': False,
            'yawn_count': self.yawn_count,
            'yawn_rate': 0.0,
            'current_yawn_duration': 0.0,
            'strategy_details': {}
        }
        
        if thermal_landmarks is None or 'mouth_region' not in thermal_landmarks:
            return results
        
        # Strategy 1: MAR-based analysis
        if self.strategy in [self.STRATEGY_MAR, self.STRATEGY_HYBRID]:
            mar_score, yawning_mar = self._analyze_mar(frame, thermal_landmarks)
            results['mar_score'] = mar_score
            results['strategy_details']['mar'] = {
                'score': mar_score,
                'yawning': yawning_mar
            }
        
        # Strategy 2: Thermal signature analysis
        if self.strategy in [self.STRATEGY_THERMAL, self.STRATEGY_HYBRID]:
            thermal_score, thermal_analysis = self._analyze_thermal_signature(frame, thermal_landmarks)
            results['thermal_score'] = thermal_score
            results['strategy_details']['thermal'] = thermal_analysis
        
        # Strategy 3: Temporal pattern analysis
        if self.strategy in [self.STRATEGY_TEMPORAL, self.STRATEGY_HYBRID]:
            temporal_score, temporal_analysis = self._analyze_temporal_patterns()
            results['temporal_score'] = temporal_score
            results['strategy_details']['temporal'] = temporal_analysis
        
        # Strategy 4: Contour-based analysis
        if self.strategy in [self.STRATEGY_CONTOUR, self.STRATEGY_HYBRID]:
            contour_score, contour_analysis = self._analyze_mouth_contour(frame, thermal_landmarks)
            results['contour_score'] = contour_score
            results['strategy_details']['contour'] = contour_analysis
        
        # Combine strategies
        if self.strategy == self.STRATEGY_HYBRID:
            combined = (
                self.MAR_WEIGHT * results['mar_score'] +
                self.THERMAL_WEIGHT * results['thermal_score'] +
                self.TEMPORAL_WEIGHT * results['temporal_score'] +
                self.CONTOUR_WEIGHT * results['contour_score']
            )
            results['combined_score'] = combined
            yawning = combined > 0.5
        else:
            results['combined_score'] = max(
                results['mar_score'],
                results['thermal_score'],
                results['temporal_score'],
                results['contour_score']
            )
            yawning = results['combined_score'] > 0.5
        
        # Filter false yawns when eyes are clearly closed (score > 0.7)
        # When eyes are very closed, mouth region thermal signature looks like open mouth
        # This is a known limitation - we suppress yawn detection when eyes are closed
        if eye_closure_score is not None and eye_closure_score > 0.7:
            # Eyes are clearly closed, reduce yawn score significantly
            # Keep some residual detection in case both eyes AND mouth are closed (rare)
            results['combined_score'] *= 0.2  # Reduce by 80% when eyes very closed
            yawning = False
        elif eye_closure_score is not None and eye_closure_score > 0.5:
            # Eyes are mostly closed, reduce yawn score
            results['combined_score'] *= 0.5  # Reduce by 50%
            yawning = False
        
        # Update yawn state
        self._update_yawn_state(yawning)
        results['yawning'] = self.yawning
        results['yawn_count'] = self.yawn_count
        results['yawn_rate'] = self._calculate_yawn_rate()
        results['current_yawn_duration'] = self.current_yawn_duration
        
        return results
    
    def _analyze_mar(self, frame, thermal_landmarks):
        """
        Strategy 1: Mouth Aspect Ratio Analysis
        Geometric analysis of mouth opening in thermal image
        
        Args:
            frame: Thermal image
            thermal_landmarks: Thermal landmarks dict
            
        Returns:
            (mar_score, yawning): Tuple of score (0-1) and boolean
        """
        mouth_region = thermal_landmarks.get('mouth_region')
        if not mouth_region:
            return 0.0, False
        
        mouth_roi = mouth_region.get('roi', (0, 0, 20, 20))
        x, y, w, h = mouth_roi
        
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0.0, False
        
        mouth_frame = frame[y:y+h, x:x+w]
        
        # Calculate MAR using thermal image characteristics
        # In thermal: mouth opening appears as darker region (cooler interior)
        
        mar = self._calculate_thermal_mar(mouth_frame)
        self.mar_history.append(mar)
        
        # Score: higher MAR = mouth open = higher score
        mar_score = min(1.0, mar / self.MAR_THRESHOLD_THERMAL)
        
        # Detect yawn
        yawning = mar > self.MAR_THRESHOLD_THERMAL
        
        return mar_score, yawning
    
    def _calculate_thermal_mar(self, mouth_region):
        """
        Calculate Mouth Aspect Ratio from thermal image
        
        Thermal advantage: mouth interior is distinctly cooler when open
        
        Args:
            mouth_region: Cropped mouth region from thermal frame
            
        Returns:
            mar_value: Float representing mouth openness
        """
        try:
            if mouth_region.size == 0:
                return 0.0
            
            h, w = mouth_region.shape
            
            # Normalize
            mouth_norm = mouth_region.astype(np.float32) / 255.0
            
            # Find darkest regions (cooler = open mouth)
            # Divide into quadrants to measure opening
            mid_h = h // 2
            mid_w = w // 2
            
            top_half = mouth_norm[:mid_h, :]
            bottom_half = mouth_norm[mid_h:, :]
            
            # Look for vertical opening (top vs bottom)
            top_darkness = np.mean(top_half)
            bottom_darkness = np.mean(bottom_half)
            
            # Look for horizontal width (left vs right)
            left_half = mouth_norm[:, :mid_w]
            right_half = mouth_norm[:, mid_w:]
            
            left_width = np.std(left_half)
            right_width = np.std(right_half)
            
            # MAR combines vertical opening and width
            # Vertical component: difference between top and bottom darkness
            vertical_opening = abs(top_darkness - bottom_darkness)
            
            # Horizontal component: width consistency
            horizontal_width = (left_width + right_width) / 2.0
            
            # Combined MAR
            mar = (vertical_opening * 2.0) + (horizontal_width * 0.5)
            
            return float(mar)
            
        except Exception as e:
            print(f"Error calculating thermal MAR: {e}")
            return 0.0
    
    def _analyze_thermal_signature(self, frame, thermal_landmarks):
        """
        Strategy 2: Thermal Signature Analysis
        Temperature-based mouth opening detection
        
        When mouth opens, interior shows different thermal signature
        (typically cooler breath/interior of mouth)
        
        Args:
            frame: Thermal image
            thermal_landmarks: Thermal landmarks
            
        Returns:
            (thermal_score, analysis_dict)
        """
        mouth_region = thermal_landmarks.get('mouth_region')
        if not mouth_region:
            return 0.0, {}
        
        mouth_roi = mouth_region.get('roi', (0, 0, 20, 20))
        x, y, w, h = mouth_roi
        
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0.0, {}
        
        mouth_frame = frame[y:y+h, x:x+w].astype(np.float32)
        
        # Thermal signature detection
        mouth_avg = np.mean(mouth_frame)
        mouth_min = np.min(mouth_frame)
        mouth_max = np.max(mouth_frame)
        
        # Temperature gradient detection
        grad_x = cv2.Sobel(mouth_frame, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mouth_frame, cv2.CV_32F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_mag)
        
        self.thermal_signature_history.append({
            'mean': mouth_avg,
            'min': mouth_min,
            'max': mouth_max,
            'gradient': avg_gradient
        })
        
        # Score based on:
        # 1. Temperature differential (open mouth = cooler interior)
        temp_diff = mouth_max - mouth_min
        temp_score = min(1.0, temp_diff / self.MOUTH_INTERIOR_TEMP_DROP)
        
        # 2. Gradient strength (sharp edges = open mouth)
        gradient_score = min(1.0, avg_gradient / 30.0)
        
        # Combined thermal score
        thermal_score = (temp_score * 0.6 + gradient_score * 0.4)
        
        analysis = {
            'mean_temp': float(mouth_avg),
            'min_temp': float(mouth_min),
            'max_temp': float(mouth_max),
            'temp_differential': float(temp_diff),
            'gradient_magnitude': float(avg_gradient),
            'temp_score': float(temp_score),
            'gradient_score': float(gradient_score),
            'combined_score': float(thermal_score)
        }
        
        return float(thermal_score), analysis
    
    def _analyze_temporal_patterns(self):
        """
        Strategy 3: Temporal Pattern Analysis
        Detect characteristic yawning motion patterns
        
        Yawning shows:
        - Gradual opening (acceleration phase)
        - Hold phase
        - Gradual closing (deceleration phase)
        - Typical duration 0.5-5 seconds
        
        Returns:
            (temporal_score, analysis_dict)
        """
        if len(self.mar_history) < 5:
            return 0.0, {}
        
        mar_array = np.array(list(self.mar_history))
        
        # Analyze movement pattern
        # Yawn shows bell-curve-like pattern
        
        # Calculate first derivative (velocity)
        if len(mar_array) > 1:
            velocity = np.diff(mar_array)
            acceleration = np.diff(velocity) if len(velocity) > 1 else velocity
            
            # Score pattern characteristics
            # Good yawn: starts with positive velocity, then negative
            positive_phase = np.sum(velocity > 0)
            negative_phase = np.sum(velocity < 0)
            
            # Pattern score: balance of opening and closing
            pattern_score = min(positive_phase, negative_phase) / len(velocity) if len(velocity) > 0 else 0
        else:
            pattern_score = 0.0
        
        # Duration analysis
        if self.yawning and self.yawn_start_time:
            current_duration = time.time() - self.yawn_start_time
            # Score based on duration fit
            if self.YAWN_DURATION_MIN <= current_duration <= self.YAWN_DURATION_MAX:
                duration_score = 1.0
            elif current_duration < self.YAWN_DURATION_MIN:
                duration_score = current_duration / self.YAWN_DURATION_MIN
            else:
                duration_score = 1.0 - min(1.0, (current_duration - self.YAWN_DURATION_MAX) / 2.0)
        else:
            duration_score = 0.5
        
        temporal_score = (pattern_score * 0.6 + duration_score * 0.4)
        
        analysis = {
            'pattern_score': float(pattern_score),
            'duration_score': float(duration_score),
            'current_duration': self.current_yawn_duration,
            'combined_score': float(temporal_score)
        }
        
        return float(temporal_score), analysis
    
    def _analyze_mouth_contour(self, frame, thermal_landmarks):
        """
        Strategy 4: Mouth Contour Analysis
        Shape-based mouth opening detection
        
        Args:
            frame: Thermal image
            thermal_landmarks: Thermal landmarks
            
        Returns:
            (contour_score, analysis_dict)
        """
        mouth_region = thermal_landmarks.get('mouth_region')
        if not mouth_region:
            return 0.0, {}
        
        mouth_roi = mouth_region.get('roi', (0, 0, 20, 20))
        x, y, w, h = mouth_roi
        
        if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
            return 0.0, {}
        
        mouth_frame = frame[y:y+h, x:x+w]
        
        try:
            # Find contours of mouth region
            _, binary = cv2.threshold(mouth_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0, {}
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            self.mouth_area_history.append(contour_area)
            
            # Fit ellipse to mouth contour
            ellipse_score = 0.0
            if len(largest_contour) >= 5:
                ellipse = cv2.fitEllipse(largest_contour)
                (center, (major_axis, minor_axis), angle) = ellipse
                
                # Aspect ratio of ellipse
                if minor_axis > 0:
                    aspect_ratio = major_axis / minor_axis
                    # More open mouth = lower aspect ratio (more circular)
                    # Closed mouth = high aspect ratio (thin line)
                    ellipse_score = 1.0 / aspect_ratio if aspect_ratio > 0 else 0.0
                    ellipse_score = min(1.0, ellipse_score)
            
            # Area score: larger contour area = more open
            area_score = min(1.0, contour_area / self.MIN_MOUTH_AREA if self.MIN_MOUTH_AREA > 0 else 0.5)
            
            # Combined contour score
            contour_score = (ellipse_score * 0.6 + area_score * 0.4)
            
            analysis = {
                'contour_area': float(contour_area),
                'area_score': float(area_score),
                'ellipse_score': float(ellipse_score),
                'combined_score': float(contour_score)
            }
            
            return float(contour_score), analysis
            
        except Exception as e:
            print(f"Error analyzing mouth contour: {e}")
            return 0.0, {}
    
    def _update_yawn_state(self, detected_yawn):
        """Update yawn state machine"""
        current_time = time.time()
        
        if detected_yawn and not self.yawning:
            # Start of yawn
            self.yawning = True
            self.yawn_start_time = current_time
            self.current_yawn_duration = 0
        
        elif detected_yawn and self.yawning:
            # Yawn ongoing
            self.current_yawn_duration = current_time - self.yawn_start_time
        
        elif not detected_yawn and self.yawning:
            # End of yawn
            self.current_yawn_duration = current_time - self.yawn_start_time
            
            # Validate yawn duration
            if self.YAWN_DURATION_MIN <= self.current_yawn_duration <= self.YAWN_DURATION_MAX:
                self.yawn_count += 1
                self.yawn_timestamps.append(current_time)
            
            self.yawning = False
            self.yawn_start_time = None
            self.current_yawn_duration = 0
    
    def _calculate_yawn_rate(self):
        """Calculate yawns per 5 minutes"""
        if not self.yawn_timestamps:
            return 0.0
        
        current_time = time.time()
        
        # Remove old yawns outside 5-minute window
        while self.yawn_timestamps and (current_time - self.yawn_timestamps[0]) > 300:
            self.yawn_timestamps.popleft()
        
        # Calculate rate
        yawns_in_window = len(self.yawn_timestamps)
        window_minutes = 5.0
        
        return (yawns_in_window / window_minutes) * 60 if window_minutes > 0 else 0.0
    
    def reset_statistics(self):
        """Reset monitoring statistics"""
        self.yawning = False
        self.yawn_start_time = None
        self.current_yawn_duration = 0
        self.yawn_count = 0
        self.frame_start_time = time.time()
        self.yawn_timestamps.clear()
        self.mar_history.clear()
        self.thermal_signature_history.clear()
