#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Loader Module

Loads and validates configuration from YAML file.
"""

import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path="config.yaml"):
        """
        Load configuration from YAML file
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self._ensure_directories()
    
    def _load_config(self):
        """Load YAML configuration file"""
        if not os.path.exists(self.config_path):
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Return default configuration"""
        return {
            'face_detection': {
                'method': 'mediapipe',
                'mediapipe': {
                    'max_num_faces': 1,
                    'refine_landmarks': True,
                    'min_detection_confidence': 0.5,
                    'min_tracking_confidence': 0.5
                }
            },
            'temporal_model': {
                'enabled': False,
                'model_type': 'tcn'
            },
            'calibration': {
                'enabled': False,
                'calibration_duration': 20
            },
            'drowsiness': {
                'eye_weight': 0.5,
                'yawn_weight': 0.25,
                'head_weight': 0.25
            }
        }
    
    def _validate_config(self):
        """Validate configuration values"""
        # Validate face detection method
        if self.config['face_detection']['method'] not in ['mediapipe', 'blazeface']:
            print("Warning: Invalid face_detection.method, defaulting to 'mediapipe'")
            self.config['face_detection']['method'] = 'mediapipe'
        
        # Validate temporal model type
        if 'temporal_model' in self.config:
            if self.config['temporal_model'].get('model_type') not in ['tcn', 'gru']:
                print("Warning: Invalid temporal_model.model_type, defaulting to 'tcn'")
                self.config['temporal_model']['model_type'] = 'tcn'
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        if 'calibration' in self.config:
            profiles_dir = self.config['calibration'].get('profiles_directory', 'driver_profiles')
            os.makedirs(profiles_dir, exist_ok=True)
        
        if 'reporting' in self.config:
            reports_dir = self.config['reporting'].get('reports_directory', 'reports')
            os.makedirs(reports_dir, exist_ok=True)
    
    def get(self, key_path, default=None):
        """
        Get configuration value using dot-notation path
        
        Args:
            key_path: Dot-separated path (e.g., 'face_detection.method')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def is_enabled(self, feature):
        """
        Check if a feature is enabled
        
        Args:
            feature: Feature name (e.g., 'temporal_model', 'calibration')
            
        Returns:
            True if enabled, False otherwise
        """
        return self.get(f'{feature}.enabled', False)

