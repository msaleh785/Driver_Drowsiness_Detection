#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Calibration Manager Module

Handles per-driver calibration to establish baseline features (EAR, MAR, head pose).
Calibration runs for 10-30 seconds while driver is alert.
"""

import json
import os
import time
import numpy as np
from typing import Dict, Optional, List
from collections import deque

class CalibrationManager:
    """
    Manages driver calibration and baseline storage
    """
    
    def __init__(self, calibration_duration: int = 20, profiles_directory: str = "driver_profiles"):
        """
        Initialize calibration manager
        
        Args:
            calibration_duration: Duration of calibration in seconds (10-30)
            profiles_directory: Directory to store/load driver profiles
        """
        self.calibration_duration = max(10, min(30, calibration_duration))  # Clamp to 10-30s
        self.profiles_directory = profiles_directory
        os.makedirs(profiles_directory, exist_ok=True)
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_progress = 0.0
        
        # Data collection during calibration
        self.ear_l_samples = deque()
        self.ear_r_samples = deque()
        self.mar_samples = deque()
        self.yaw_samples = deque()
        self.pitch_samples = deque()
        self.roll_samples = deque()
        
        # Current baseline (loaded or default)
        self.baseline = None
        self.current_driver_id = None
    
    def start_calibration(self, driver_id: str = "default"):
        """
        Start calibration process
        
        Args:
            driver_id: Unique identifier for driver
        """
        self.is_calibrating = True
        self.calibration_start_time = time.time()
        self.calibration_progress = 0.0
        self.current_driver_id = driver_id
        
        # Clear previous samples
        self.ear_l_samples.clear()
        self.ear_r_samples.clear()
        self.mar_samples.clear()
        self.yaw_samples.clear()
        self.pitch_samples.clear()
        self.roll_samples.clear()
        
        print(f"\n=== Starting calibration for driver: {driver_id} ===")
        print(f"Please remain alert and look forward for {self.calibration_duration} seconds...")
    
    def update_calibration(self, eye_status: Dict, yawn_status: Dict, head_status: Dict) -> bool:
        """
        Update calibration with current frame data
        
        Args:
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            
        Returns:
            True if calibration is complete, False if still calibrating
        """
        if not self.is_calibrating:
            return False
        
        elapsed = time.time() - self.calibration_start_time
        self.calibration_progress = min(1.0, elapsed / self.calibration_duration)
        
        # Collect samples
        self.ear_l_samples.append(eye_status.get('left_ear', 0.0))
        self.ear_r_samples.append(eye_status.get('right_ear', 0.0))
        self.mar_samples.append(yawn_status.get('mar', 0.0))
        
        # Extract head pose if available
        if 'yaw' in head_status:
            self.yaw_samples.append(head_status['yaw'])
        if 'pitch' in head_status:
            self.pitch_samples.append(head_status['pitch'])
        if 'roll' in head_status:
            self.roll_samples.append(head_status['roll'])
        
        # Check if calibration is complete
        if elapsed >= self.calibration_duration:
            self._complete_calibration()
            return True
        
        return False
    
    def _complete_calibration(self):
        """Complete calibration and compute baselines"""
        if len(self.ear_l_samples) == 0:
            print("Warning: No calibration samples collected")
            self.is_calibrating = False
            return
        
        # Compute means (baselines)
        baseline = {
            "driver_id": self.current_driver_id,
            "EAR0_L": float(np.mean(self.ear_l_samples)),
            "EAR0_R": float(np.mean(self.ear_r_samples)),
            "MAR0": float(np.mean(self.mar_samples)),
            "yaw0": float(np.mean(self.yaw_samples)) if self.yaw_samples else 0.0,
            "pitch0": float(np.mean(self.pitch_samples)) if self.pitch_samples else -5.0,
            "roll0": float(np.mean(self.roll_samples)) if self.roll_samples else 0.0,
            # Optional: standard deviations for adaptive thresholds
            "EAR_L_std": float(np.std(self.ear_l_samples)),
            "EAR_R_std": float(np.std(self.ear_r_samples)),
            "MAR_std": float(np.std(self.mar_samples)),
            "calibration_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.baseline = baseline
        self.is_calibrating = False
        
        # Save profile
        self.save_profile(self.current_driver_id, baseline)
        
        print(f"\n=== Calibration complete ===")
        print(f"Baselines:")
        print(f"  EAR_L: {baseline['EAR0_L']:.3f} ± {baseline['EAR_L_std']:.3f}")
        print(f"  EAR_R: {baseline['EAR0_R']:.3f} ± {baseline['EAR_R_std']:.3f}")
        print(f"  MAR: {baseline['MAR0']:.3f} ± {baseline['MAR_std']:.3f}")
        print(f"  Head pose: yaw={baseline['yaw0']:.2f}, pitch={baseline['pitch0']:.2f}, roll={baseline['roll0']:.2f}")
        print(f"Profile saved to: {self.get_profile_path(self.current_driver_id)}")
    
    def cancel_calibration(self):
        """Cancel ongoing calibration"""
        self.is_calibrating = False
        self.calibration_start_time = None
        self.calibration_progress = 0.0
        print("Calibration cancelled")
    
    def save_profile(self, driver_id: str, baseline: Dict):
        """
        Save driver profile to disk
        
        Args:
            driver_id: Driver identifier
            baseline: Baseline dictionary
        """
        profile_path = self.get_profile_path(driver_id)
        try:
            with open(profile_path, 'w') as f:
                json.dump(baseline, f, indent=2)
            print(f"Profile saved: {profile_path}")
        except Exception as e:
            print(f"Error saving profile: {e}")
    
    def load_profile(self, driver_id: str) -> Optional[Dict]:
        """
        Load driver profile from disk
        
        Args:
            driver_id: Driver identifier
            
        Returns:
            Baseline dictionary or None if not found
        """
        profile_path = self.get_profile_path(driver_id)
        if not os.path.exists(profile_path):
            return None
        
        try:
            with open(profile_path, 'r') as f:
                baseline = json.load(f)
            self.baseline = baseline
            self.current_driver_id = driver_id
            print(f"Loaded profile for driver: {driver_id}")
            return baseline
        except Exception as e:
            print(f"Error loading profile: {e}")
            return None
    
    def get_profile_path(self, driver_id: str) -> str:
        """Get file path for driver profile"""
        return os.path.join(self.profiles_directory, f"{driver_id}_profile.json")
    
    def set_default_baseline(self, default_baseline: Dict):
        """
        Set default baseline (used if no profile loaded)
        
        Args:
            default_baseline: Default baseline dictionary
        """
        self.baseline = default_baseline
        print("Using default baseline values")
    
    def normalize_features(self, eye_status: Dict, yawn_status: Dict, 
                          head_status: Dict) -> Dict:
        """
        Normalize features using current baseline
        
        Args:
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            
        Returns:
            Dictionary with normalized features
        """
        if not self.baseline:
            # Return original values if no baseline
            return {
                "EAR_L_norm": eye_status.get('left_ear', 0.0),
                "EAR_R_norm": eye_status.get('right_ear', 0.0),
                "MAR_norm": yawn_status.get('mar', 0.0),
                "yaw_rel": head_status.get('yaw', 0.0),
                "pitch_rel": head_status.get('pitch', 0.0),
                "roll_rel": head_status.get('roll', 0.0)
            }
        
        # Normalize
        normalized = {
            "EAR_L_norm": eye_status.get('left_ear', 0.0) / max(self.baseline['EAR0_L'], 0.01),
            "EAR_R_norm": eye_status.get('right_ear', 0.0) / max(self.baseline['EAR0_R'], 0.01),
            "MAR_norm": yawn_status.get('mar', 0.0) / max(self.baseline['MAR0'], 0.01),
            "yaw_rel": head_status.get('yaw', 0.0) - self.baseline['yaw0'],
            "pitch_rel": head_status.get('pitch', 0.0) - self.baseline['pitch0'],
            "roll_rel": head_status.get('roll', 0.0) - self.baseline['roll0']
        }
        
        return normalized
    
    def get_calibration_progress(self) -> float:
        """Get calibration progress (0.0-1.0)"""
        return self.calibration_progress if self.is_calibrating else 0.0
    
    def is_calibration_active(self) -> bool:
        """Check if calibration is currently active"""
        return self.is_calibrating

