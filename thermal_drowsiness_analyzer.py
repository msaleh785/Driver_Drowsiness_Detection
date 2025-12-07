#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Drowsiness Analyzer Module

Multi-modal fusion engine combining:
- Eye closure indicators (40% weight)
- Yawning indicators (25% weight)
- Thermal signatures (15% weight)
- Temporal trends (15% weight)
- Head position (5% weight)

Outputs 5-level drowsiness classification with confidence scores.
"""

import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple


class ThermalDrowsinessAnalyzer:
    """Multi-modal drowsiness analysis with thermal optimization"""
    
    # Drowsiness levels
    LEVEL_ALERT = 'ALERT'
    LEVEL_PRE_ALERT = 'PRE_ALERT'
    LEVEL_SOFT = 'SOFT'
    LEVEL_MEDIUM = 'MEDIUM'
    LEVEL_CRITICAL = 'CRITICAL'
    
    def __init__(self, config=None, eye_monitor=None, yawn_detector=None):
        """
        Initialize drowsiness analyzer
        
        Args:
            config: Configuration dictionary (optional)
            eye_monitor: ThermalEyeMonitor instance (optional)
            yawn_detector: ThermalYawnDetector instance (optional)
        """
        
        # Store references (optional, for future use)
        self.config = config or {}
        self.eye_monitor = eye_monitor
        self.yawn_detector = yawn_detector
        
        # Thresholds for level transitions (can be overridden by config)
        default_thresholds = {
            'pre_alert': 0.40,
            'soft': 0.55,
            'medium': 0.70,
            'critical': 0.85
        }
        
        # Load thresholds from config if available
        if self.config and 'thresholds' in self.config:
            self.THRESHOLDS = self.config['thresholds']
        else:
            self.THRESHOLDS = default_thresholds
        
        # Component weights (must sum to 1.0)
        default_weights = {
            'eye_closure': 0.40,
            'yawn': 0.25,
            'thermal_signature': 0.15,
            'temporal_trend': 0.15,
            'head_position': 0.05
        }
        
        # Load weights from config if available
        if self.config and 'weights' in self.config:
            self.WEIGHTS = self.config['weights']
        else:
            self.WEIGHTS = default_weights
        
        # State tracking
        self.current_level = self.LEVEL_ALERT
        self.drowsiness_score_history = deque(maxlen=30)
        self.level_history = deque(maxlen=30)
        self.level_consistency_counter = 0
        self.level_consistency_frames = 5
        
        # Session statistics
        self.total_frames = 0
        self.alert_frames = 0
        self.pre_alert_frames = 0
        self.soft_frames = 0
        self.medium_frames = 0
        self.critical_frames = 0
        self.session_start_time = time.time()
    
    def analyze_drowsiness(self, eye_closure_score, yawn_score, ear_score=None, 
                          perclos=None, blink_rate=None, yawn_count=None, 
                          yawn_frequency=None, head_angle=None):
        """
        Multi-modal drowsiness analysis
        
        Args:
            eye_closure_score: Eye closure score (0-1)
            yawn_score: Yawning score (0-1)
            ear_score: Eye Aspect Ratio (optional)
            perclos: Percentage eye closure (optional)
            blink_rate: Blinks per minute (optional)
            yawn_count: Number of yawns (optional)
            yawn_frequency: Yawns per 5 minutes (optional)
            head_angle: Head tilt angle in degrees (optional)
            
        Returns:
            dict with drowsiness analysis results
        """
        
        self.total_frames += 1
        
        # Calculate component scores
        component_scores = self._calculate_component_scores(
            eye_closure_score, yawn_score, ear_score, perclos, 
            blink_rate, yawn_count, yawn_frequency, head_angle
        )
        
        # Fuse scores
        drowsiness_score = self._fuse_scores(component_scores, eye_closure_score, yawn_score)
        
        # Determine level
        drowsiness_level = self._determine_level(drowsiness_score)
        
        # Update state machine
        self._update_level_state(drowsiness_level)
        
        # Identify contributing factors
        contributing_factors = self._identify_contributing_factors(
            eye_closure_score, yawn_score, perclos, blink_rate, yawn_count
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(drowsiness_level, contributing_factors)
        
        # Calculate confidence
        confidence = self._calculate_confidence(component_scores)
        
        # Update session statistics
        self._update_session_stats(drowsiness_level)
        
        # Store history
        self.drowsiness_score_history.append(drowsiness_score)
        self.level_history.append(drowsiness_level)
        
        # Map level to numeric value for compatibility
        level_map = {
            self.LEVEL_ALERT: 0,
            self.LEVEL_PRE_ALERT: 1,
            self.LEVEL_SOFT: 2,
            self.LEVEL_MEDIUM: 3,
            self.LEVEL_CRITICAL: 4
        }
        
        return {
            'drowsiness_score': drowsiness_score,
            'drowsiness_level': level_map.get(drowsiness_level, 0),
            'level_name': drowsiness_level,
            'alert_status': self._get_alert_status(drowsiness_level),
            'confidence': confidence,
            'component_scores': component_scores,
            'contributing_factors': contributing_factors,
            'recommendations': recommendations
        }
    
    def _calculate_component_scores(self, eye_closure_score, yawn_score, ear_score=None,
                                   perclos=None, blink_rate=None, yawn_count=None,
                                   yawn_frequency=None, head_angle=None):
        """Calculate individual component scores"""
        
        components = {
            'eye_closure': eye_closure_score,
            'yawn': yawn_score,
            'thermal_signature': eye_closure_score * 0.6 + yawn_score * 0.4,  # Derived
            'temporal_trend': self._calculate_temporal_trend(),
            'head_position': self._calculate_head_position_score(head_angle) if head_angle else 0.0
        }
        
        return components
    
    def _fuse_scores(self, component_scores, eye_closure_score, yawn_score):
        """
        Fuse component scores with multi-modal weights
        Plus boost when both eye closure AND yawning detected together
        """
        
        # Base weighted sum
        base_score = sum(
            component_scores[key] * self.WEIGHTS[key]
            for key in component_scores
        )
        
        # Boost for combined drowsiness indicators
        # If both eyes are closed AND yawning, this is strong drowsiness
        if eye_closure_score > 0.6 and yawn_score > 0.5:
            # Add boost (multiplier effect)
            combined_factor = 1.15  # 15% boost
            base_score *= combined_factor
            base_score = min(1.0, base_score)  # Cap at 1.0
        
        return base_score
    
    def _determine_level(self, drowsiness_score):
        """Determine drowsiness level from score"""
        
        if drowsiness_score >= self.THRESHOLDS['critical']:
            return self.LEVEL_CRITICAL
        elif drowsiness_score >= self.THRESHOLDS['medium']:
            return self.LEVEL_MEDIUM
        elif drowsiness_score >= self.THRESHOLDS['soft']:
            return self.LEVEL_SOFT
        elif drowsiness_score >= self.THRESHOLDS['pre_alert']:
            return self.LEVEL_PRE_ALERT
        else:
            return self.LEVEL_ALERT
    
    def _update_level_state(self, new_level):
        """Update level with hysteresis to prevent flickering"""
        
        if new_level == self.current_level:
            self.level_consistency_counter = 0
        else:
            self.level_consistency_counter += 1
            if self.level_consistency_counter >= self.level_consistency_frames:
                self.current_level = new_level
                self.level_consistency_counter = 0
    
    def _calculate_temporal_trend(self):
        """Calculate temporal trend in drowsiness"""
        
        if len(self.drowsiness_score_history) < 2:
            return 0.0
        
        # Calculate trend over last 10 frames
        recent_scores = list(self.drowsiness_score_history)[-10:]
        if len(recent_scores) < 2:
            return 0.0
        
        # Calculate acceleration (how fast drowsiness is increasing)
        trend = recent_scores[-1] - recent_scores[0]
        trend_score = min(1.0, max(0.0, trend / 0.5))  # Normalize to 0-1
        
        return trend_score
    
    def _calculate_head_position_score(self, head_angle):
        """Calculate score from head position/angle"""
        
        # Head tilt > 30 degrees indicates drowsiness
        if head_angle is None:
            return 0.0
        
        abs_angle = abs(head_angle)
        if abs_angle > 45:
            return 1.0
        elif abs_angle > 30:
            return (abs_angle - 30) / 15.0
        else:
            return 0.0
    
    def _identify_contributing_factors(self, eye_closure_score, yawn_score, 
                                      perclos=None, blink_rate=None, yawn_count=None):
        """Identify which factors are contributing to drowsiness"""
        
        factors = []
        
        if eye_closure_score > 0.6:
            factors.append("Eye closure detected")
        
        if perclos and perclos > 0.3:
            factors.append(f"High PERCLOS ({perclos:.1%})")
        
        if blink_rate and blink_rate < 8:
            factors.append(f"Low blink rate ({blink_rate:.1f}/min)")
        
        if yawn_score > 0.6:
            factors.append("Active yawning")
        
        if yawn_count and yawn_count > 2:
            factors.append(f"Frequent yawning ({yawn_count} yawns)")
        
        if eye_closure_score > 0.6 and yawn_score > 0.5:
            factors.append("Combined eye closure + yawning")
        
        if not factors:
            factors.append("No significant drowsiness indicators")
        
        return factors
    
    def _generate_recommendations(self, drowsiness_level, contributing_factors):
        """Generate recommendations based on drowsiness level"""
        
        recommendations = []
        
        if drowsiness_level == self.LEVEL_ALERT:
            recommendations.append("Driver is alert. Continue monitoring.")
        
        elif drowsiness_level == self.LEVEL_PRE_ALERT:
            recommendations.append("Early drowsiness indicators detected.")
            recommendations.append("Consider taking a short break.")
            recommendations.append("Blink rate is low - try increasing eye movement.")
        
        elif drowsiness_level == self.LEVEL_SOFT:
            recommendations.append("Moderate drowsiness detected.")
            recommendations.append("Take a 15-minute break immediately.")
            recommendations.append("Get some fresh air or exercise.")
        
        elif drowsiness_level == self.LEVEL_MEDIUM:
            recommendations.append("SEVERE DROWSINESS - STOP DRIVING!")
            recommendations.append("Pull over to a safe location immediately.")
            recommendations.append("Rest for at least 20 minutes.")
        
        elif drowsiness_level == self.LEVEL_CRITICAL:
            recommendations.append("CRITICAL DROWSINESS - EMERGENCY!")
            recommendations.append("PULL OVER IMMEDIATELY!")
            recommendations.append("Do not continue driving.")
        
        return recommendations
    
    def _calculate_confidence(self, component_scores):
        """Calculate confidence based on agreement between components"""
        
        scores = list(component_scores.values())
        if not scores:
            return 0.5
        
        # Confidence is higher when components agree
        mean_score = np.mean(scores)
        variance = np.var(scores)
        
        # Lower variance = higher confidence
        confidence = 1.0 - (variance / 0.25)  # Normalize by typical variance
        confidence = np.clip(confidence, 0.0, 1.0)
        
        return confidence
    
    def _get_alert_status(self, drowsiness_level):
        """Get alert status string"""
        
        if drowsiness_level == self.LEVEL_CRITICAL:
            return "CRITICAL_ALERT"
        elif drowsiness_level == self.LEVEL_MEDIUM:
            return "SEVERE_ALERT"
        elif drowsiness_level == self.LEVEL_SOFT:
            return "WARNING_ALERT"
        elif drowsiness_level == self.LEVEL_PRE_ALERT:
            return "ALERT"
        else:
            return "NORMAL"
    
    def _update_session_stats(self, drowsiness_level):
        """Update session statistics"""
        
        if drowsiness_level == self.LEVEL_ALERT:
            self.alert_frames += 1
        elif drowsiness_level == self.LEVEL_PRE_ALERT:
            self.pre_alert_frames += 1
        elif drowsiness_level == self.LEVEL_SOFT:
            self.soft_frames += 1
        elif drowsiness_level == self.LEVEL_MEDIUM:
            self.medium_frames += 1
        elif drowsiness_level == self.LEVEL_CRITICAL:
            self.critical_frames += 1
    
    def get_session_summary(self):
        """Get summary of entire session"""
        
        if self.total_frames == 0:
            return {
                'session_duration': 0.0,
                'total_frames_processed': 0,
                'alert_percentage': 0.0,
                'drowsy_percentage': 0.0,
                'critical_percentage': 0.0,
                'average_drowsiness_score': 0.0,
                'overall_assessment': 'No data collected'
            }
        
        # Calculate session duration
        if hasattr(self, 'session_start_time') and self.session_start_time:
            session_duration = time.time() - self.session_start_time
        else:
            session_duration = self.total_frames / 15.0  # Assume 15 FPS
        
        # Calculate percentages
        alert_pct = (self.alert_frames / self.total_frames) * 100
        pre_alert_pct = (self.pre_alert_frames / self.total_frames) * 100
        soft_pct = (self.soft_frames / self.total_frames) * 100
        medium_pct = (self.medium_frames / self.total_frames) * 100
        critical_pct = (self.critical_frames / self.total_frames) * 100
        
        drowsy_pct = pre_alert_pct + soft_pct + medium_pct + critical_pct
        
        # Average drowsiness score
        if len(self.drowsiness_score_history) > 0:
            avg_score = sum(self.drowsiness_score_history) / len(self.drowsiness_score_history)
        else:
            avg_score = 0.0
        
        # Overall assessment
        if critical_pct > 10:
            assessment = "CRITICAL - Driver showed severe drowsiness signs"
        elif medium_pct > 20:
            assessment = "HIGH RISK - Frequent moderate drowsiness detected"
        elif soft_pct > 30:
            assessment = "MODERATE RISK - Multiple drowsiness episodes"
        elif pre_alert_pct > 40:
            assessment = "LOW RISK - Some early drowsiness indicators"
        else:
            assessment = "NORMAL - Driver remained alert"
        
        return {
            'session_duration': session_duration,
            'total_frames_processed': self.total_frames,
            'alert_percentage': alert_pct,
            'pre_alert_percentage': pre_alert_pct,
            'soft_percentage': soft_pct,
            'medium_percentage': medium_pct,
            'critical_percentage': critical_pct,
            'drowsy_percentage': drowsy_pct,
            'average_drowsiness_score': avg_score,
            'overall_assessment': assessment
        }