#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BlazeFace Face Detector Module

Lightweight face detection using BlazeFace TFLite model with frame skipping.
After detection, uses MediaPipe for landmark extraction.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TensorFlow Lite not available. BlazeFace will not work.")

class BlazeFaceDetector:
    """
    BlazeFace face detector with frame skipping optimization
    """
    
    def __init__(self, model_path: str, detection_interval: int = 3, 
                 score_threshold: float = 0.5, use_mediapipe_landmarks: bool = True):
        """
        Initialize BlazeFace detector
        
        Args:
            model_path: Path to BlazeFace TFLite model
            detection_interval: Detect every Nth frame (3 = every 3rd frame)
            score_threshold: Minimum confidence score for detection
            use_mediapipe_landmarks: Whether to use MediaPipe for landmarks after detection
        """
        self.detection_interval = detection_interval
        self.score_threshold = score_threshold
        self.use_mediapipe_landmarks = use_mediapipe_landmarks
        self.frame_count = 0
        self.last_bbox = None  # Last detected bounding box for tracking
        
        # Initialize TFLite interpreter
        if not TFLITE_AVAILABLE:
            raise ImportError("TensorFlow Lite not available. Install tflite-runtime or tensorflow.")
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Loaded BlazeFace model from {model_path}")
        except Exception as e:
            raise IOError(f"Failed to load BlazeFace model: {e}")
        
        # Initialize MediaPipe for landmarks (if enabled)
        if self.use_mediapipe_landmarks:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("MediaPipe landmarks enabled for BlazeFace pipeline")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for BlazeFace input
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Preprocessed frame (RGB, resized, normalized)
        """
        # BlazeFace expects 128x128 RGB input
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (128, 128))
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=0)
        return frame_expanded
    
    def _postprocess_detections(self, outputs: List, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Postprocess BlazeFace outputs to get bounding box
        
        Args:
            outputs: Model output tensors
            frame_shape: (height, width) of original frame
            
        Returns:
            Bounding box (x, y, width, height) or None
        """
        # BlazeFace output format: [x_center, y_center, width, height, score]
        # This is a simplified version - actual BlazeFace has more complex output
        # Adjust based on your specific BlazeFace model
        
        if len(outputs) < 2:
            return None
        
        # Get detection boxes and scores
        boxes = outputs[0]  # Shape: [1, N, 4] or similar
        scores = outputs[1]  # Shape: [1, N] or similar
        
        if boxes.size == 0 or scores.size == 0:
            return None
        
        # Find best detection
        best_idx = np.argmax(scores)
        best_score = scores.flatten()[best_idx]
        
        if best_score < self.score_threshold:
            return None
        
        # Extract bounding box (normalized coordinates)
        box = boxes[0][best_idx] if len(boxes.shape) > 2 else boxes[best_idx]
        
        # Convert normalized coordinates to pixel coordinates
        h, w = frame_shape[:2]
        x_center = int(box[0] * w)
        y_center = int(box[1] * h)
        width = int(box[2] * w)
        height = int(box[3] * h)
        
        # Convert center+size to x, y, width, height
        x = max(0, x_center - width // 2)
        y = max(0, y_center - height // 2)
        
        return (x, y, width, height)
    
    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect face in frame (with frame skipping)
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            Bounding box (x, y, width, height) or None
        """
        self.frame_count += 1
        
        # Only run detection every Nth frame
        if self.frame_count % self.detection_interval != 0:
            # Return last known bounding box (simple tracking)
            return self.last_bbox
        
        # Preprocess
        input_data = self._preprocess_frame(frame)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in self.output_details:
            output_data = self.interpreter.get_tensor(output_detail['index'])
            outputs.append(output_data)
        
        # Postprocess
        bbox = self._postprocess_detections(outputs, frame.shape)
        
        if bbox:
            self.last_bbox = bbox
        
        return bbox
    
    def extract_landmarks(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional:
        """
        Extract facial landmarks using MediaPipe
        
        Args:
            frame: Input frame (BGR)
            bbox: Optional bounding box to crop face region
            
        Returns:
            MediaPipe face landmarks or None
        """
        if not self.use_mediapipe_landmarks:
            return None
        
        # Crop to face region if bbox provided (optional optimization)
        if bbox:
            x, y, w, h = bbox
            # Add padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            face_roi = frame[y:y+h, x:x+w]
        else:
            face_roi = frame
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        
        return None
    
    def process_frame(self, frame: np.ndarray) -> Optional:
        """
        Complete pipeline: detect face and extract landmarks
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            MediaPipe face landmarks or None
        """
        # Detect face
        bbox = self.detect_face(frame)
        
        if bbox is None:
            return None
        
        # Extract landmarks
        landmarks = self.extract_landmarks(frame, bbox)
        
        return landmarks
    
    def reset(self):
        """Reset frame counter and tracking state"""
        self.frame_count = 0
        self.last_bbox = None

