#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Face Detector Factory

Factory pattern to support multiple face detection models:
- MediaPipe (default, most accurate for RGB)
- BlazeFace (lightweight TFLite model)
- dlib (classical ML, good for thermal)

Enables easy switching and fallback between models.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict


class MediaPipeFaceDetector:
    """MediaPipe Face Mesh detector"""
    
    def __init__(self, config: Dict = None):
        """Initialize MediaPipe detector"""
        self.mp_face_mesh = mp.solutions.face_mesh
        cfg = config or {}
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=cfg.get('max_num_faces', 1),
            refine_landmarks=cfg.get('refine_landmarks', True),
            min_detection_confidence=cfg.get('min_detection_confidence', 0.5),
            min_tracking_confidence=cfg.get('min_tracking_confidence', 0.5)
        )
        self.name = "MediaPipe"
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[any], float]:
        """
        Detect face in frame
        
        Returns:
            (detected: bool, landmarks: mediapipe landmarks or None, confidence: float)
        """
        if frame.ndim == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            return True, results.multi_face_landmarks[0], 0.7
        return False, None, 0.0


class BlazeFaceFaceDetector:
    """BlazeFace TFLite detector"""
    
    def __init__(self, config: Dict = None):
        """Initialize BlazeFace detector"""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                raise ImportError("TensorFlow Lite not available")
        
        cfg = config or {}
        model_path = cfg.get('model_path', 'models/blazeface.tflite')
        
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.score_threshold = cfg.get('score_threshold', 0.5)
            self.use_mediapipe_landmarks = cfg.get('use_mediapipe_landmarks', True)
            
            if self.use_mediapipe_landmarks:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5
                )
            
            self.name = "BlazeFace"
        except Exception as e:
            raise IOError(f"Failed to load BlazeFace model: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[any], float]:
        """Detect face using BlazeFace"""
        # Simplified implementation - returns basic detection
        # Full BlazeFace would need proper TFLite processing
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            if self.use_mediapipe_landmarks:
                results = self.face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    return True, results.multi_face_landmarks[0], 0.6
        except:
            pass
        
        return False, None, 0.0


class DlibFaceDetector:
    """dlib face detector"""
    
    def __init__(self, config: Dict = None):
        """Initialize dlib detector"""
        try:
            import dlib
        except ImportError:
            raise ImportError("dlib not installed. Install with: pip install dlib")
        
        cfg = config or {}
        model_path = cfg.get('face_detector_model', 'models/dlib_face_detector.svm')
        
        try:
            self.detector = dlib.simple_object_detector(model_path)
            self.dlib = dlib
            self.name = "dlib"
        except Exception as e:
            raise IOError(f"Failed to load dlib face detector: {e}")
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[Dict], float]:
        """
        Detect face using dlib
        
        Returns:
            (detected: bool, face_roi: dict or None, confidence: float)
        """
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        try:
            dets = self.detector(frame_gray, 1)  # upsample_num_times=1
            
            if len(dets) > 0:
                # Pick face closest to center (not the largest)
                # This avoids false positives on body regions
                h, w = frame_gray.shape
                center_x, center_y = w // 2, h // 2
                
                best_det = None
                min_distance = float('inf')
                
                for det in dets:
                    # Calculate distance from center
                    det_center_x = det.left() + det.width() // 2
                    det_center_y = det.top() + det.height() // 2
                    distance = abs(det_center_x - center_x) + abs(det_center_y - center_y)
                    
                    # Prefer face-sized detections (roughly square aspect ratio)
                    aspect_ratio = det.width() / det.height() if det.height() > 0 else 0
                    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
                        continue  # Skip non-face-like detections
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_det = det
                
                if best_det:
                    x, y, w, h = best_det.left(), best_det.top(), best_det.width(), best_det.height()
                    
                    face_roi = {
                        'roi': (int(x), int(y), int(w), int(h)),
                        'rect': best_det,  # Keep dlib rectangle for landmark prediction
                        'confidence': 0.8  # dlib doesn't return confidence
                    }
                    return True, face_roi, 0.8
        except Exception as e:
            pass
        
        return False, None, 0.0


class FaceDetectorFactory:
    """Factory for creating and managing face detectors"""
    
    DETECTORS = {
        'mediapipe': MediaPipeFaceDetector,
        'blazeface': BlazeFaceFaceDetector,
        'dlib': DlibFaceDetector,
    }
    
    def __init__(self, primary_method: str = 'mediapipe', fallback_method: str = None, 
                 use_fallback: bool = True, config: Dict = None):
        """
        Initialize factory
        
        Args:
            primary_method: Primary detector ('mediapipe', 'blazeface', 'dlib')
            fallback_method: Fallback detector if primary fails
            use_fallback: Enable automatic fallback
            config: Configuration dict with detector configs
        """
        self.config = config or {}
        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.use_fallback = use_fallback
        
        self.primary_detector = None
        self.fallback_detector = None
        self.current_detector = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize primary and fallback detectors"""
        # Initialize primary
        try:
            detector_class = self.DETECTORS[self.primary_method]
            cfg = self.config.get(self.primary_method, {})
            self.primary_detector = detector_class(cfg)
            self.current_detector = self.primary_detector
            print(f"✓ Initialized primary detector: {self.primary_method}")
        except Exception as e:
            print(f"✗ Failed to initialize {self.primary_method}: {e}")
            self.primary_detector = None
        
        # Initialize fallback if specified
        if self.use_fallback and self.fallback_method:
            try:
                detector_class = self.DETECTORS[self.fallback_method]
                cfg = self.config.get(self.fallback_method, {})
                self.fallback_detector = detector_class(cfg)
                print(f"✓ Initialized fallback detector: {self.fallback_method}")
            except Exception as e:
                print(f"✗ Failed to initialize fallback {self.fallback_method}: {e}")
                self.fallback_detector = None
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[any], str, float]:
        """
        Detect face using primary or fallback detector
        
        Returns:
            (detected: bool, landmarks/roi: any, detector_name: str, confidence: float)
        """
        if self.primary_detector:
            detected, result, confidence = self.primary_detector.detect(frame)
            if detected:
                return True, result, self.primary_detector.name, confidence
        
        # Try fallback if primary failed
        if self.use_fallback and self.fallback_detector:
            print(f"Primary detector failed, trying fallback: {self.fallback_method}")
            detected, result, confidence = self.fallback_detector.detect(frame)
            if detected:
                return True, result, self.fallback_detector.name, confidence
        
        return False, None, "None", 0.0
    
    def switch_detector(self, method: str, config: Dict = None) -> bool:
        """
        Switch to a different detector
        
        Args:
            method: Detector method ('mediapipe', 'blazeface', 'dlib')
            config: Configuration for the detector
            
        Returns:
            bool: Success status
        """
        if method not in self.DETECTORS:
            print(f"Unknown detector method: {method}")
            return False
        
        try:
            cfg = config or self.config.get(method, {})
            detector_class = self.DETECTORS[method]
            new_detector = detector_class(cfg)
            self.primary_detector = new_detector
            self.primary_method = method
            print(f"✓ Switched to detector: {method}")
            return True
        except Exception as e:
            print(f"✗ Failed to switch to {method}: {e}")
            return False
    
    def get_available_detectors(self):
        """Get list of available detectors"""
        return list(self.DETECTORS.keys())
    
    def get_current_detector(self):
        """Get current active detector name"""
        return self.primary_method if self.primary_detector else None


def create_face_detector(config: Dict = None) -> FaceDetectorFactory:
    """
    Create face detector from config
    
    Args:
        config: Configuration dict (typically from config.yaml)
        
    Returns:
        FaceDetectorFactory instance
    """
    if config is None:
        config = {}
    
    face_config = config.get('face_detection', {})
    primary = face_config.get('method', 'mediapipe')
    fallback = face_config.get('fallback_method', 'dlib')
    use_fallback = face_config.get('use_fallback', True)
    
    # Extract individual detector configs
    detector_configs = {
        'mediapipe': face_config.get('mediapipe', {}),
        'blazeface': face_config.get('blazeface', {}),
        'dlib': face_config.get('dlib', {}),
    }
    
    return FaceDetectorFactory(
        primary_method=primary,
        fallback_method=fallback,
        use_fallback=use_fallback,
        config=detector_configs
    )
