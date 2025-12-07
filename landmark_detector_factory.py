#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Landmark Detector Factory

Factory pattern to support multiple landmark detection models:
- MediaPipe (468 landmarks, most accurate)
- dlib (68 landmarks, classical ML)

Enables easy switching and conversion between landmark formats.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, Dict, List


class MediaPipeLandmarkDetector:
    """MediaPipe Face Mesh landmark detector"""
    
    def __init__(self, config: Dict = None):
        """Initialize MediaPipe landmark detector"""
        self.mp_face_mesh = mp.solutions.face_mesh
        cfg = config or {}
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=cfg.get('max_num_faces', 1),
            refine_landmarks=cfg.get('refine_landmarks', True),
            min_detection_confidence=cfg.get('min_detection_confidence', 0.5),
            min_tracking_confidence=cfg.get('min_tracking_confidence', 0.5)
        )
        self.name = "MediaPipe"
        self.num_landmarks = 468
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, Optional[any]]:
        """
        Detect landmarks in frame
        
        Returns:
            (detected: bool, landmarks: mediapipe landmarks or None)
        """
        if frame.ndim == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            return True, results.multi_face_landmarks[0]
        return False, None
    
    def get_landmark_indices(self, part: str) -> List[int]:
        """Get MediaPipe landmark indices for specific face parts"""
        indices = {
            'left_eye': [33, 160, 158, 133, 153, 144],
            'right_eye': [362, 385, 387, 263, 373, 380],
            'mouth': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405],
            'left_eyebrow': [70, 63, 105, 66, 107],
            'right_eyebrow': [336, 296, 334, 293, 300],
            'nose': [1, 2, 3, 4, 5, 6, 8, 9, 10],
            'face_contour': [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                           397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                           172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
        }
        return indices.get(part, [])


class DlibLandmarkDetector:
    """dlib landmark detector"""
    
    def __init__(self, config: Dict = None):
        """Initialize dlib landmark detector"""
        try:
            import dlib
            from imutils import face_utils
        except ImportError:
            raise ImportError("dlib and imutils not installed. Install with: pip install dlib imutils")
        
        self.dlib = dlib
        self.face_utils = face_utils
        cfg = config or {}
        model_path = cfg.get('landmark_predictor_model', 'models/dlib_landmark_predictor.dat')
        self.return_format = cfg.get('return_format', 'mediapipe_compatible')
        
        try:
            self.predictor = dlib.shape_predictor(model_path)
            self.name = "dlib"
            self.num_landmarks = 68
        except Exception as e:
            raise IOError(f"Failed to load dlib landmark predictor: {e}")
    
    def detect(self, frame: np.ndarray, face_roi: any = None) -> Tuple[bool, Optional[any]]:
        """
        Detect landmarks using dlib
        
        Args:
            frame: Input frame (BGR or grayscale)
            face_roi: Face ROI from MediaPipe or dlib detector (dict with 'roi' key or dlib.rectangle)
            
        Returns:
            (detected: bool, landmarks: dlib.full_object_detection or converted format)
        """
        # Convert to grayscale if needed
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame
        
        try:
            # Get dlib rectangle object from various input formats
            if isinstance(face_roi, dict):
                # Check if dict contains dlib rectangle (from dlib detector)
                if 'rect' in face_roi:
                    face_rect = face_roi['rect']
                elif 'roi' in face_roi:
                    # From MediaPipe or other detectors
                    x, y, w, h = face_roi['roi']
                    # Create dlib rectangle with proper integer conversion
                    face_rect = self.dlib.rectangle(
                        int(max(0, x)), 
                        int(max(0, y)), 
                        int(min(frame_gray.shape[1], x + w)), 
                        int(min(frame_gray.shape[0], y + h))
                    )
                else:
                    return False, None
            else:
                # Assume it's already dlib.rectangle
                face_rect = face_roi
            
            # Predict landmarks using grayscale image
            dlib_landmarks = self.predictor(frame_gray, face_rect)
            
            if self.return_format == 'mediapipe_compatible':
                return True, self._convert_to_mediapipe_format(dlib_landmarks, frame_gray.shape)
            else:
                return True, dlib_landmarks
        except Exception as e:
            return False, None
    
    def _convert_to_mediapipe_format(self, dlib_landmarks, frame_shape: Tuple[int, int]) -> any:
        """
        Convert dlib 68 landmarks to MediaPipe-compatible format
        
        Creates a mock MediaPipe landmark object with dlib data
        """
        height, width = frame_shape
        
        class MockLandmark:
            def __init__(self, x, y):
                self.x = x / width
                self.y = y / height
                self.z = 0.0
        
        class MockLandmarkList:
            def __init__(self, landmarks):
                self.landmark = landmarks
        
        # Convert dlib 68 points to MediaPipe format (pad to 468)
        mediapipe_landmarks = []
        
        # Add dlib landmarks (positions 0-67)
        for i in range(len(dlib_landmarks.parts())):
            point = dlib_landmarks.parts()[i]
            mediapipe_landmarks.append(MockLandmark(float(point.x), float(point.y)))
        
        # Pad with zeros to reach 468 landmarks (MediaPipe format)
        for _ in range(468 - len(dlib_landmarks.parts())):
            mediapipe_landmarks.append(MockLandmark(0.0, 0.0))
        
        return MockLandmarkList(mediapipe_landmarks)
    
    def get_landmark_indices(self, part: str) -> List[int]:
        """Get dlib landmark indices for specific face parts (68-point model)"""
        indices = {
            'left_eye': [42, 43, 44, 45, 46, 47],
            'right_eye': [36, 37, 38, 39, 40, 41],
            'mouth': [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
            'left_eyebrow': [22, 23, 24, 25, 26],
            'right_eyebrow': [17, 18, 19, 20, 21],
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
            'face_contour': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        }
        return indices.get(part, [])


class LandmarkDetectorFactory:
    """Factory for creating and managing landmark detectors"""
    
    DETECTORS = {
        'mediapipe': MediaPipeLandmarkDetector,
        'dlib': DlibLandmarkDetector,
    }
    
    def __init__(self, primary_method: str = 'mediapipe', fallback_method: str = None,
                 use_fallback: bool = True, config: Dict = None):
        """
        Initialize factory
        
        Args:
            primary_method: Primary detector ('mediapipe', 'dlib')
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
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize primary and fallback detectors"""
        # Initialize primary
        try:
            detector_class = self.DETECTORS[self.primary_method]
            cfg = self.config.get(self.primary_method, {})
            self.primary_detector = detector_class(cfg)
            print(f"✓ Initialized primary landmark detector: {self.primary_method}")
        except Exception as e:
            print(f"✗ Failed to initialize {self.primary_method}: {e}")
            self.primary_detector = None
        
        # Initialize fallback if specified
        if self.use_fallback and self.fallback_method:
            try:
                detector_class = self.DETECTORS[self.fallback_method]
                cfg = self.config.get(self.fallback_method, {})
                self.fallback_detector = detector_class(cfg)
                print(f"✓ Initialized fallback landmark detector: {self.fallback_method}")
            except Exception as e:
                print(f"✗ Failed to initialize fallback {self.fallback_method}: {e}")
                self.fallback_detector = None
    
    def detect(self, frame: np.ndarray, face_roi: any = None) -> Tuple[bool, Optional[any], str]:
        """
        Detect landmarks using primary or fallback detector
        
        Args:
            frame: Input frame
            face_roi: Face ROI (required for dlib, optional for MediaPipe)
            
        Returns:
            (detected: bool, landmarks: any, detector_name: str)
        """
        if self.primary_detector:
            if isinstance(self.primary_detector, DlibLandmarkDetector):
                if face_roi is None:
                    # Try MediaPipe as fallback for landmark detection only
                    if self.fallback_detector and not isinstance(self.fallback_detector, DlibLandmarkDetector):
                        detected, result = self.fallback_detector.detect(frame)
                        if detected:
                            return True, result, self.fallback_detector.name
                else:
                    detected, result = self.primary_detector.detect(frame, face_roi)
                    if detected:
                        return True, result, self.primary_detector.name
            else:
                detected, result = self.primary_detector.detect(frame)
                if detected:
                    return True, result, self.primary_detector.name
        
        # Try fallback if primary failed
        if self.use_fallback and self.fallback_detector:
            if isinstance(self.fallback_detector, DlibLandmarkDetector):
                if face_roi is not None:
                    detected, result = self.fallback_detector.detect(frame, face_roi)
                    if detected:
                        return True, result, self.fallback_detector.name
            else:
                detected, result = self.fallback_detector.detect(frame)
                if detected:
                    return True, result, self.fallback_detector.name
        
        return False, None, "None"
    
    def switch_detector(self, method: str, config: Dict = None) -> bool:
        """
        Switch to a different detector
        
        Args:
            method: Detector method ('mediapipe', 'dlib')
            config: Configuration for the detector
            
        Returns:
            bool: Success status
        """
        if method not in self.DETECTORS:
            print(f"Unknown landmark detector method: {method}")
            return False
        
        try:
            cfg = config or self.config.get(method, {})
            detector_class = self.DETECTORS[method]
            new_detector = detector_class(cfg)
            self.primary_detector = new_detector
            self.primary_method = method
            print(f"✓ Switched to landmark detector: {method}")
            return True
        except Exception as e:
            print(f"✗ Failed to switch to {method}: {e}")
            return False
    
    def get_available_detectors(self):
        """Get list of available landmark detectors"""
        return list(self.DETECTORS.keys())
    
    def get_current_detector(self):
        """Get current active detector name"""
        return self.primary_method if self.primary_detector else None


def create_landmark_detector(config: Dict = None) -> LandmarkDetectorFactory:
    """
    Create landmark detector from config
    
    Args:
        config: Configuration dict (typically from config.yaml)
        
    Returns:
        LandmarkDetectorFactory instance
    """
    if config is None:
        config = {}
    
    landmark_config = config.get('landmark_detection', {})
    primary = landmark_config.get('method', 'mediapipe')
    fallback = landmark_config.get('fallback_method', 'dlib')
    use_fallback = landmark_config.get('use_fallback', True)
    
    # Extract individual detector configs
    detector_configs = {
        'mediapipe': landmark_config.get('mediapipe', {}),
        'dlib': landmark_config.get('dlib', {}),
    }
    
    return LandmarkDetectorFactory(
        primary_method=primary,
        fallback_method=fallback,
        use_fallback=use_fallback,
        config=detector_configs
    )
