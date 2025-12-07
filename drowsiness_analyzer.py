#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Drowsiness Analyzer Module

Combines temporal model output with rule-based multi-indicator logic.
Supports both temporal model (TCN/GRU) and traditional weighted fusion.
"""

import time
from collections import deque
from typing import Optional, Dict, Tuple

class DrowsinessAnalyzer:
    # Drowsiness levels
    LEVEL_ALERT = 0      # Driver is alert
    LEVEL_PRE_ALERT = 1  # Pre-alert (early warning)
    LEVEL_SOFT = 2       # Soft alert (slight drowsiness)
    LEVEL_MEDIUM = 3    # Medium alert (moderate drowsiness)
    LEVEL_CRITICAL = 4  # Critical alert (severe drowsiness)
    
    def __init__(self, config: Dict, temporal_model=None, calibration_manager=None):
        """
        Initialize drowsiness analyzer
        
        Args:
            config: Configuration dictionary
            temporal_model: Optional temporal model instance
            calibration_manager: Optional calibration manager instance
        """
        self.config = config
        self.temporal_model = temporal_model
        self.calibration_manager = calibration_manager
        
        # Get configuration
        drowsiness_config = config.get('drowsiness', {})
        thresholds = drowsiness_config.get('thresholds', {})
        rules_config = drowsiness_config.get('rules', {})
        
        # Thresholds for multi-indicator logic
        self.thresholds = {
            'pre_alert': thresholds.get('pre_alert', {'temporal_score': 0.5, 'duration_seconds': 0.5, 'min_rules_active': 0}),
            'soft': thresholds.get('soft', {'temporal_score': 0.6, 'duration_seconds': 0.5, 'min_rules_active': 1}),
            'medium': thresholds.get('medium', {'temporal_score': 0.7, 'duration_seconds': 1.0, 'min_rules_active': 2}),
            'critical': thresholds.get('critical', {'temporal_score': 0.8, 'duration_seconds': 1.5, 'min_rules_active': 2})
        }
        
        # Rule thresholds
        self.rule_thresholds = {
            'ear': rules_config.get('ear_threshold', 0.7),
            'mar': rules_config.get('mar_threshold', 1.4),
            'yaw': rules_config.get('yaw_threshold', 0.3),
            'pitch': rules_config.get('pitch_threshold', 0.3),
            'perclos': rules_config.get('perclos_threshold', 0.4)
        }
        
        # Fallback weights (if temporal model disabled)
        self.eye_weight = drowsiness_config.get('eye_weight', 0.5)
        self.yawn_weight = drowsiness_config.get('yawn_weight', 0.25)
        self.head_weight = drowsiness_config.get('head_weight', 0.25)
        total = self.eye_weight + self.yawn_weight + self.head_weight
        if total > 0:
            self.eye_weight /= total
            self.yawn_weight /= total
            self.head_weight /= total
        
        # State tracking
        self.current_level = self.LEVEL_ALERT
        self.level_start_time = time.time()
        self.level_consistency_frames = drowsiness_config.get('level_consistency_frames', 5)
        self.allow_immediate_down = drowsiness_config.get('allow_immediate_down', True)
        
        # Temporal score history
        self.temporal_score_history = deque(maxlen=30)  # ~2 seconds at 15 FPS
        self.rule_state_history = deque(maxlen=30)
        
        # Rule state tracking
        self.active_rules = set()
        self.rule_durations = {}  # Track how long each rule has been active
    
    def _evaluate_rules(self, normalized_features: Dict, eye_status: Dict, 
                       yawn_status: Dict, head_status: Dict) -> set:
        """
        Evaluate rule-based guards
        
        Args:
            normalized_features: Normalized features from calibration
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            
        Returns:
            Set of active rule names
        """
        active_rules = set()
        current_time = time.time()
        
        # Rule 1: EAR normalized < threshold for >0.5s
        ear_l_norm = normalized_features.get('EAR_L_norm', 1.0)
        ear_r_norm = normalized_features.get('EAR_R_norm', 1.0)
        ear_avg_norm = (ear_l_norm + ear_r_norm) / 2.0
        
        if ear_avg_norm < self.rule_thresholds['ear']:
            if 'ear_low' not in self.rule_durations:
                self.rule_durations['ear_low'] = current_time
            elif current_time - self.rule_durations['ear_low'] > 0.5:
                active_rules.add('ear_low')
        else:
            self.rule_durations.pop('ear_low', None)
        
        # Rule 2: MAR normalized > threshold (frequent yawns)
        mar_norm = normalized_features.get('MAR_norm', 1.0)
        if mar_norm > self.rule_thresholds['mar']:
            active_rules.add('mar_high')
        
        # Rule 3: Yaw relative > threshold (head tilt)
        yaw_rel = abs(normalized_features.get('yaw_rel', 0.0))
        if yaw_rel > self.rule_thresholds['yaw']:
            active_rules.add('yaw_deviation')
        
        # Rule 4: Pitch relative > threshold (head nod/droop)
        pitch_rel = abs(normalized_features.get('pitch_rel', 0.0))
        if pitch_rel > self.rule_thresholds['pitch']:
            active_rules.add('pitch_deviation')
        
        # Rule 5: PERCLOS > threshold
        perclos = eye_status.get('perclos', 0.0)
        if perclos > self.rule_thresholds['perclos']:
            active_rules.add('perclos_high')
        
        # Rule 6: Sleep detected
        if eye_status.get('sleep_detected', False):
            active_rules.add('sleep_detected')
        
        # Rule 7: Head nodding
        if head_status.get('is_nodding', False):
            active_rules.add('head_nodding')
        
        return active_rules
    
    def _determine_level_from_temporal_and_rules(self, temporal_score: float, 
                                                  active_rules: set) -> int:
        """
        Determine drowsiness level using temporal score + rules
        
        Args:
            temporal_score: Temporal model output (0.0-1.0)
            active_rules: Set of active rule names
            
        Returns:
            Drowsiness level (0-4)
        """
        num_active_rules = len(active_rules)
        current_time = time.time()
        
        # Check each level from highest to lowest
        # Critical
        crit_thresh = self.thresholds['critical']
        if (temporal_score >= crit_thresh['temporal_score'] and 
            num_active_rules >= crit_thresh['min_rules_active']):
            # Check duration
            if self.current_level == self.LEVEL_CRITICAL:
                if current_time - self.level_start_time >= crit_thresh['duration_seconds']:
                    return self.LEVEL_CRITICAL
            else:
                # Transitioning to critical - reset timer
                self.level_start_time = current_time
                return self.LEVEL_CRITICAL
        
        # Medium
        med_thresh = self.thresholds['medium']
        if (temporal_score >= med_thresh['temporal_score'] and 
            num_active_rules >= med_thresh['min_rules_active']):
            if self.current_level >= self.LEVEL_MEDIUM:
                if current_time - self.level_start_time >= med_thresh['duration_seconds']:
                    return self.LEVEL_MEDIUM
            else:
                self.level_start_time = current_time
                return self.LEVEL_MEDIUM
        
        # Soft
        soft_thresh = self.thresholds['soft']
        if (temporal_score >= soft_thresh['temporal_score'] and 
            num_active_rules >= soft_thresh['min_rules_active']):
            if self.current_level >= self.LEVEL_SOFT:
                if current_time - self.level_start_time >= soft_thresh['duration_seconds']:
                    return self.LEVEL_SOFT
            else:
                self.level_start_time = current_time
                return self.LEVEL_SOFT
        
        # Pre-alert
        pre_thresh = self.thresholds['pre_alert']
        if temporal_score >= pre_thresh['temporal_score']:
            if self.current_level >= self.LEVEL_PRE_ALERT:
                if current_time - self.level_start_time >= pre_thresh['duration_seconds']:
                    return self.LEVEL_PRE_ALERT
            else:
                self.level_start_time = current_time
                return self.LEVEL_PRE_ALERT
        
        # Alert (normal)
        return self.LEVEL_ALERT
    
    def _convert_level_to_score(self, level: str) -> float:
        """Convert text-based drowsiness level to numeric score"""
        if level == "high":
            return 1.0
        elif level == "medium":
            return 0.7
        elif level == "low":
            return 0.3
        else:  # "none"
            return 0.0
    
    def _calculate_weighted_score_fallback(self, eye_status: Dict, yawn_status: Dict, 
                                           head_status: Dict) -> float:
        """
        Fallback weighted score calculation (when temporal model disabled)
        """
        scores = {}
        raw_eye_drowsiness = eye_status.get("drowsiness", "none")
        is_closed = eye_status.get("is_closed", False)
        perclos = eye_status.get("perclos", 0.0)
        closed_frames = eye_status.get("eye_closed_counter", 0)

        # Eye contribution: be conservative when eyes are open so normal blinks
        # do not by themselves create level-2/3 alerts.
        if is_closed:
            eye_score = self._convert_level_to_score(raw_eye_drowsiness)
        else:
            if perclos < 0.25:
                eye_score = 0.0
            elif perclos < 0.5:
                eye_score = 0.2
            else:
                eye_score = 0.4

        # Yawn contribution. Map medium/high yaw drowsiness to a full
        # contribution (1.0) so that yawing alone can raise the level to at
        # least a soft alert via the weighted combination.
        raw_yawn_level = yawn_status.get("drowsiness", "none")
        if raw_yawn_level in ("high", "medium"):
            yawn_score = 1.0
        elif raw_yawn_level == "low":
            yawn_score = 0.7
        else:
            yawn_score = 0.0

        head_score = self._convert_level_to_score(head_status.get("drowsiness", "none"))
        
        # Critical condition: Extended eye closure
        if eye_status.get("sleep_detected", False):
            total = 1.0
            scores = {"eye": 1.0, "yawn": yawn_score, "head": head_score, "total": total}
            self.last_fallback_scores = scores
            return total
        
        # Synergy detection (yawn + eyes)
        if yawn_status.get("is_yawning", False) and eye_score > 0.5:
            eye_score = min(1.0, eye_score + 0.2)
        
        # If eyes are still closed and PERCLOS is high, force at least medium level,
        # but only when the closure is sustained (not a normal blink). Require a
        # minimum closed-eye duration so frequent short blinks do not trigger this.
        # Also do not jump directly to critical; reserve score~1.0 for sleep_detected.
        if is_closed and perclos > 0.4 and closed_frames >= 30:
            boosted = max(0.7, min(0.8, eye_score))
            total = boosted
            scores = {"eye": boosted, "yawn": yawn_score, "head": head_score, "total": total}
            self.last_fallback_scores = scores
            return total
        
        # Weighted combination (yaw contributes only via its weight)
        weighted_score = (
            self.eye_weight * eye_score +
            self.yawn_weight * yawn_score +
            self.head_weight * head_score
        )

        total = min(1.0, max(0.0, weighted_score))
        scores = {"eye": eye_score, "yawn": yawn_score, "head": head_score, "total": total}
        self.last_fallback_scores = scores
        return total
    
    def _get_level_description(self, level: int) -> str:
        """Get text description for drowsiness level"""
        descriptions = {
            self.LEVEL_ALERT: "ALERT",
            self.LEVEL_PRE_ALERT: "PRE-ALERT",
            self.LEVEL_SOFT: "SOFT ALERT",
            self.LEVEL_MEDIUM: "MODERATE DROWSINESS",
            self.LEVEL_CRITICAL: "CRITICAL DROWSINESS"
        }
        return descriptions.get(level, "UNKNOWN")
    
    def _generate_status_details(self, eye_status: Dict, yawn_status: Dict, 
                                head_status: Dict, active_rules: set, 
                                temporal_score: Optional[float]) -> list:
        """Generate detailed status message"""
        details = []
        
        # Temporal score
        if temporal_score is not None:
            details.append(f"Temporal score: {temporal_score:.2f}")
        
        # Active rules
        if active_rules:
            details.append(f"Active rules: {', '.join(active_rules)}")
        
        # Eye status
        if eye_status.get("drowsiness") == "high":
            details.append("Eyes closing frequently")
        elif eye_status.get("is_closed", False):
            details.append("Eyes currently closed")
        elif eye_status.get("perclos", 0) > 0.15:
            details.append(f"PERCLOS: {eye_status['perclos']:.2f}")
        
        # Yawn status
        if yawn_status.get("is_yawning", False):
            details.append(f"Yawning ({yawn_status.get('yawn_duration', 0):.1f}s)")
        elif yawn_status.get("yawn_frequency", 0) > 2:
            # Frequency is computed over a 1-minute sliding window
            details.append(f"Frequent yawning ({yawn_status['yawn_frequency']:.1f}/1min)")
        
        # Head movement
        if head_status.get("is_nodding", False):
            details.append("Head nodding detected")
        elif head_status.get("is_improper_position", False):
            details.append(f"Improper head position: {head_status.get('position', 'unknown')}")
        
        if not details:
            details.append("Driver appears alert")
        
        return details
    
    def analyze(self, eye_status: Dict, yawn_status: Dict, head_status: Dict,
                temporal_score: Optional[float] = None) -> Tuple[int, Dict]:
        """
        Analyze drowsiness using temporal model + rules or fallback method
        
        Args:
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            temporal_score: Optional temporal model score (if None, will use model if available)
            
        Returns:
            tuple: (drowsiness_level, detailed_status)
        """
        # Get normalized features if calibration enabled
        normalized_features = {}
        if self.calibration_manager:
            normalized_features = self.calibration_manager.normalize_features(
                eye_status, yawn_status, head_status
            )
        else:
            # Use raw values as normalized (no calibration)
            normalized_features = {
                "EAR_L_norm": eye_status.get('left_ear', 0.0) / 0.28,  # Rough default
                "EAR_R_norm": eye_status.get('right_ear', 0.0) / 0.27,
                "MAR_norm": yawn_status.get('mar', 0.0) / 0.35,
                "yaw_rel": head_status.get('yaw', 0.0),
                "pitch_rel": head_status.get('pitch', 0.0),
                "roll_rel": head_status.get('roll', 0.0)
            }
        
        # Evaluate rules
        active_rules = self._evaluate_rules(normalized_features, eye_status, yawn_status, head_status)
        
        # Get temporal score
        if temporal_score is None and self.temporal_model:
            temporal_score = self.temporal_model.update(eye_status, yawn_status, head_status)
        
        # Determine drowsiness level
        if temporal_score is not None:
            # Use temporal model + rules
            new_level = self._determine_level_from_temporal_and_rules(temporal_score, active_rules)
        else:
            # Fallback to weighted fusion
            fallback_score = self._calculate_weighted_score_fallback(eye_status, yawn_status, head_status)
            # Map to levels (simplified mapping)
            if fallback_score >= 0.85:
                new_level = self.LEVEL_CRITICAL
            elif fallback_score >= 0.70:
                new_level = self.LEVEL_MEDIUM
            elif fallback_score >= 0.50:
                new_level = self.LEVEL_SOFT
            elif fallback_score >= 0.30:
                new_level = self.LEVEL_PRE_ALERT
            else:
                new_level = self.LEVEL_ALERT
        
        # Update level with hysteresis
        if new_level != self.current_level:
            if self.allow_immediate_down and new_level < self.current_level:
                # Immediate decrease allowed
                self.current_level = new_level
                self.level_start_time = time.time()
            else:
                # Check consistency for increases
                # For now, allow immediate transition (can add consistency check if needed)
                self.current_level = new_level
                if new_level > self.current_level:
                    self.level_start_time = time.time()
        
        # Store history
        self.temporal_score_history.append(temporal_score if temporal_score is not None else 0.0)
        self.rule_state_history.append(len(active_rules))
        
        # Generate status
        level_description = self._get_level_description(self.current_level)
        status_details = self._generate_status_details(
            eye_status, yawn_status, head_status, active_rules, temporal_score
        )
        scores = getattr(self, "last_fallback_scores", {})
        if normalized_features:
            ear_l_norm = normalized_features.get("EAR_L_norm", 0.0)
            ear_r_norm = normalized_features.get("EAR_R_norm", 0.0)
            ear_avg_norm = (ear_l_norm + ear_r_norm) / 2.0
            s = dict(scores) if scores else {}
            s["ear_norm"] = ear_avg_norm
            s["mar_norm"] = normalized_features.get("MAR_norm", 0.0)
            s["perclos"] = eye_status.get("perclos", 0.0)
            scores = s
        
        detailed_status = {
            "level": self.current_level,
            "description": level_description,
            "temporal_score": temporal_score,
            "active_rules": list(active_rules),
            "num_active_rules": len(active_rules),
            "details": status_details,
            "eye_status": eye_status,
            "yawn_status": yawn_status,
            "head_status": head_status,
            "normalized_features": normalized_features,
            "scores": scores
        }
        
        return self.current_level, detailed_status
    
    def reset(self):
        """Reset analyzer state"""
        self.current_level = self.LEVEL_ALERT
        self.level_start_time = time.time()
        self.temporal_score_history.clear()
        self.rule_state_history.clear()
        self.active_rules.clear()
        self.rule_durations.clear()
