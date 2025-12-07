#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Temporal Model Module

Lightweight temporal reasoning using TCN or GRU for drowsiness detection.
Uses TFLite quantized models for efficient inference on Raspberry Pi 4.
"""

import numpy as np
import time
from collections import deque
from typing import Optional, Dict, List

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TensorFlow Lite not available. Temporal model will not work.")

class TemporalModel:
    """
    Temporal Convolutional Network or GRU for drowsiness prediction
    """
    
    def __init__(self, model_path: str, model_type: str = "tcn",
                 window_size_seconds: float = 2.0, fps: int = 15,
                 input_features: int = 10):
        """
        Initialize temporal model
        
        Args:
            model_path: Path to TFLite model file
            model_type: "tcn" or "gru"
            window_size_seconds: Time window for features (1-2 seconds)
            fps: Expected FPS for window calculation
            input_features: Number of input features per frame
        """
        self.model_type = model_type
        self.window_size_seconds = window_size_seconds
        self.fps = fps
        self.input_features = input_features
        
        # Calculate buffer size
        self.window_size_frames = int(window_size_seconds * fps)
        
        # Feature buffer (rolling window)
        self.feature_buffer = deque(maxlen=self.window_size_frames)
        
        # Initialize TFLite interpreter
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available. Install tflite-runtime or tensorflow.")
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Loaded temporal {model_type.upper()} model from {model_path}")
            print(f"Window size: {self.window_size_frames} frames ({window_size_seconds}s at {fps} FPS)")
        except Exception as e:
            raise IOError(f"Failed to load temporal model: {e}")
        
        # Initialize buffer with zeros
        zero_features = [0.0] * self.input_features
        for _ in range(self.window_size_frames):
            self.feature_buffer.append(zero_features)
    
    def _extract_features(self, eye_status: Dict, yawn_status: Dict, 
                         head_status: Dict) -> List[float]:
        """
        Extract feature vector from status dictionaries
        
        Args:
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            
        Returns:
            Feature vector [EAR_L, EAR_R, MAR, yaw, pitch, roll, blink_flag, yawn_flag, perclos, head_nod_flag]
        """
        # Extract features
        ear_l = eye_status.get('left_ear', 0.0)
        ear_r = eye_status.get('right_ear', 0.0)
        mar = yawn_status.get('mar', 0.0)
        
        # Head pose (simplified - extract from head_status if available)
        yaw = head_status.get('yaw', 0.0) if 'yaw' in head_status else 0.0
        pitch = head_status.get('pitch', 0.0) if 'pitch' in head_status else 0.0
        roll = head_status.get('roll', 0.0) if 'roll' in head_status else 0.0
        
        # Flags
        blink_flag = 1.0 if eye_status.get('blink_detected', False) else 0.0
        yawn_flag = 1.0 if yawn_status.get('is_yawning', False) else 0.0
        head_nod_flag = 1.0 if head_status.get('is_nodding', False) else 0.0
        
        # PERCLOS
        perclos = eye_status.get('perclos', 0.0)
        
        return [ear_l, ear_r, mar, yaw, pitch, roll, blink_flag, yawn_flag, perclos, head_nod_flag]
    
    def update(self, eye_status: Dict, yawn_status: Dict, head_status: Dict) -> Optional[float]:
        """
        Update feature buffer and get drowsiness probability
        
        Args:
            eye_status: Eye monitor status
            yawn_status: Yawn detector status
            head_status: Head monitor status
            
        Returns:
            Drowsiness probability (0.0-1.0) or None if buffer not ready
        """
        # Extract features
        features = self._extract_features(eye_status, yawn_status, head_status)
        
        # Add to buffer (automatically drops oldest)
        self.feature_buffer.append(features)
        
        # Check if buffer is ready
        if len(self.feature_buffer) < self.window_size_frames:
            return None
        
        # Convert buffer to numpy array
        feature_array = np.array(list(self.feature_buffer), dtype=np.float32)
        
        # Reshape for model input (batch_size, time_steps, features)
        # TFLite models typically expect shape: [1, time_steps, features]
        input_shape = (1, self.window_size_frames, self.input_features)
        feature_array = feature_array.reshape(input_shape)
        
        # Run inference
        try:
            self.interpreter.set_tensor(self.input_details[0]['index'], feature_array)
            self.interpreter.invoke()
            
            # Get output (drowsiness probability)
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            p_drowsy = float(output_data[0][0])  # Assuming single output value
            
            return p_drowsy
        except Exception as e:
            print(f"Error in temporal model inference: {e}")
            return None
    
    def reset(self):
        """Reset feature buffer"""
        zero_features = [0.0] * self.input_features
        self.feature_buffer.clear()
        for _ in range(self.window_size_frames):
            self.feature_buffer.append(zero_features)

