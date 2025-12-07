#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Face Detector Module

Specialized detection for thermal infrared images.
Uses temperature-based face detection and landmark extraction adapted for thermal data.
"""

import cv2
import numpy as np
from collections import deque
import mediapipe as mp


class ThermalFaceDetector:
    """
    Detects faces in thermal images using:
    1. Temperature-based face detection (thermal signature of face is warmer than background)
    2. Contour analysis for face boundaries
    3. Facial feature localization using thermal gradients
    """
    
    def __init__(self, min_face_size=50, temperature_threshold=None, use_mediapipe_fallback=True):
        """
        Initialize thermal face detector
        
        Args:
            min_face_size: Minimum face size in pixels
            temperature_threshold: Temperature threshold for face detection (None = auto-calibrate)
            use_mediapipe_fallback: Fall back to MediaPipe if thermal detection fails
        """
        self.min_face_size = min_face_size
        self.temperature_threshold = temperature_threshold
        self.use_mediapipe_fallback = use_mediapipe_fallback
        
        # Initialize MediaPipe as fallback
        if use_mediapipe_fallback:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
        
        # Calibration for thermal image characteristics
        self.temp_calibration_frames = deque(maxlen=30)
        self.is_calibrated = False
        self.face_temp_baseline = 32.0  # Average human face temp in Celsius (camera dependent)
        
    def detect_face(self, frame):
        """
        Detect face in thermal frame
        
        Args:
            frame: Input frame (grayscale thermal, 8-bit or 16-bit)
            
        Returns:
            dict: {
                'detected': bool,
                'face_roi': (x, y, w, h) or None,
                'landmarks': landmarks or None,
                'confidence': float,
                'thermal_map': processed thermal map,
                'method': 'thermal' or 'mediapipe'
            }
        """
        result = {
            'detected': False,
            'face_roi': None,
            'landmarks': None,
            'confidence': 0.0,
            'thermal_map': None,
            'method': None
        }
        
        # Handle 16-bit thermal images
        if frame.dtype == np.uint16:
            frame_normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            frame_normalized = frame
        
        # Apply thermal-specific preprocessing
        thermal_map = self._preprocess_thermal(frame_normalized)
        result['thermal_map'] = thermal_map
        
        # Try thermal-based detection first
        face_roi, confidence = self._detect_face_thermal(thermal_map)
        
        if face_roi is not None and confidence > 0.5:
            result['detected'] = True
            result['face_roi'] = face_roi
            result['confidence'] = confidence
            result['method'] = 'thermal'
            
            # Extract landmarks from thermal ROI
            landmarks = self._extract_thermal_landmarks(thermal_map, face_roi)
            result['landmarks'] = landmarks
        
        # Fallback to MediaPipe if thermal detection failed
        elif self.use_mediapipe_fallback and frame.dtype == np.uint8:
            try:
                # Convert to RGB for MediaPipe (thermal is grayscale)
                rgb_frame = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2RGB)
                mp_results = self.face_mesh.process(rgb_frame)
                
                if mp_results.multi_face_landmarks:
                    result['detected'] = True
                    result['method'] = 'mediapipe'
                    result['confidence'] = 0.7
                    
                    # Extract bounding box from landmarks
                    landmarks_array = np.array([
                        [lm.x * frame.shape[1], lm.y * frame.shape[0]]
                        for lm in mp_results.multi_face_landmarks[0].landmark
                    ])
                    x, y, w, h = cv2.boundingRect(landmarks_array.astype(np.int32))
                    result['face_roi'] = (x, y, w, h)
                    
                    # Convert MediaPipe landmarks to thermal format
                    result['landmarks'] = self._convert_mediapipe_landmarks(
                        mp_results.multi_face_landmarks[0],
                        frame.shape,
                        (x, y, w, h)
                    )
            except Exception as e:
                print(f"MediaPipe fallback failed: {e}")
        
        return result
    
    def _preprocess_thermal(self, frame):
        """
        Preprocess thermal image for face detection
        
        Args:
            frame: 8-bit thermal image
            
        Returns:
            Processed thermal map
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        thermal_enhanced = clahe.apply(frame)
        
        # Apply bilateral filter to reduce noise while preserving edges
        thermal_filtered = cv2.bilateralFilter(thermal_enhanced, 9, 75, 75)
        
        return thermal_filtered
    
    def _detect_face_thermal(self, thermal_map):
        """
        Detect face using thermal characteristics
        
        Strategy: Face appears as a high-temperature (bright) region in thermal images
        
        Args:
            thermal_map: Preprocessed thermal image
            
        Returns:
            (face_roi, confidence): Face bounding box and confidence score
        """
        h, w = thermal_map.shape
        
        # Apply Gaussian blur for smoothing
        blurred = cv2.GaussianBlur(thermal_map, (21, 21), 0)
        
        # Threshold to find hot regions (face)
        # Use Otsu's thresholding for automatic threshold selection
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, 0.0
        
        # Find the largest contour (most likely the face)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Size validation
        if area < (self.min_face_size ** 2):
            return None, 0.0
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Aspect ratio check (face should be roughly square)
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 0.6 or aspect_ratio > 1.6:
            return None, 0.0
        
        # Calculate confidence based on area and circularity
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        confidence = min(1.0, circularity * 0.8 + 0.2)  # Faces aren't perfectly circular
        
        return (x, y, w, h), confidence
    
    def _extract_thermal_landmarks(self, thermal_map, face_roi):
        """
        Extract facial landmarks from thermal image
        Uses temperature gradients and structural analysis
        
        Args:
            thermal_map: Preprocessed thermal image
            face_roi: Face bounding box (x, y, w, h)
            
        Returns:
            dict with key thermal features (eyes, mouth approximations)
        """
        x, y, w, h = face_roi
        face_region = thermal_map[y:y+h, x:x+w]
        
        landmarks = {
            'face_roi': face_roi,
            'eye_regions': self._detect_thermal_eyes(face_region, (x, y)),
            'mouth_region': self._detect_thermal_mouth(face_region, (x, y)),
            'face_center': (x + w//2, y + h//2)
        }
        
        return landmarks
    
    def _detect_thermal_eyes(self, face_region, offset):
        """
        Detect eye regions in thermal image
        Eyes appear as cold spots (dark regions) in thermal images
        
        Args:
            face_region: Cropped face region
            offset: (x_offset, y_offset) for absolute coordinates
            
        Returns:
            dict with left and right eye regions
        """
        h, w = face_region.shape
        
        # Eyes are typically in upper 1/3 of face and appear as dark regions (cold)
        eye_region = face_region[:h//2, :]
        
        # Invert: eyes are COLD (dark in thermal), so find dark regions
        inverted = 255 - eye_region
        blurred = cv2.GaussianBlur(inverted, (15, 15), 0)
        
        # Find peaks (dark regions = high values in inverted)
        # Divide face into left and right halves to find eye centers
        left_half = blurred[:, :w//2]
        right_half = blurred[:, w//2:]
        
        # Find approximate eye positions
        left_eye_y, left_eye_x = np.unravel_index(left_half.argmax(), left_half.shape)
        right_eye_y, right_eye_x = np.unravel_index(right_half.argmax(), right_half.shape)
        right_eye_x += w // 2  # Adjust for right half offset
        
        x_off, y_off = offset
        
        eye_size = max(w//8, 15)  # Approximate eye region size
        
        return {
            'left': {
                'center': (x_off + left_eye_x, y_off + left_eye_y),
                'roi': (max(0, left_eye_x - eye_size), max(0, left_eye_y - eye_size),
                       eye_size * 2, eye_size * 2)
            },
            'right': {
                'center': (x_off + right_eye_x, y_off + right_eye_y),
                'roi': (max(0, right_eye_x - eye_size), max(0, right_eye_y - eye_size),
                       eye_size * 2, eye_size * 2)
            }
        }
    
    def _detect_thermal_mouth(self, face_region, offset):
        """
        Detect mouth region in thermal image
        Mouth appears with distinct thermal signature
        
        Args:
            face_region: Cropped face region
            offset: (x_offset, y_offset) for absolute coordinates
            
        Returns:
            dict with mouth region info
        """
        h, w = face_region.shape
        
        # Mouth is typically in lower 1/3 of face
        mouth_region = face_region[h*2//3:, :]
        
        blurred = cv2.GaussianBlur(mouth_region, (11, 11), 0)
        
        # Find approximate mouth center
        mouth_y, mouth_x = np.unravel_index(blurred.argmax(), blurred.shape)
        mouth_y += h * 2 // 3  # Adjust for offset
        
        x_off, y_off = offset
        mouth_size = max(w//6, 10)
        
        return {
            'center': (x_off + mouth_x, y_off + mouth_y),
            'roi': (max(0, mouth_x - mouth_size), max(0, mouth_y - mouth_size),
                   mouth_size * 2, mouth_size * 2)
        }
    
    def _convert_mediapipe_landmarks(self, mp_landmarks, frame_shape, face_roi):
        """
        Convert MediaPipe landmarks to thermal landmark format
        
        Args:
            mp_landmarks: MediaPipe NormalizedLandmarkList
            frame_shape: Shape of frame (height, width)
            face_roi: Face bounding box (x, y, w, h)
            
        Returns:
            dict with thermal landmark format
        """
        height, width = frame_shape
        x, y, w, h = face_roi
        
        # Convert landmarks to absolute coordinates
        landmarks_array = np.array([
            [lm.x * width, lm.y * height]
            for lm in mp_landmarks.landmark
        ])
        
        # MediaPipe facial landmarks indices (FACEMESH)
        # Left eye: 33, 160, 158, 133, 153, 144
        # Right eye: 263, 387, 385, 362, 382, 381
        # Mouth: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375
        
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [263, 387, 385, 362, 382, 381]
        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375]
        
        # Extract eye regions
        left_eye_pts = landmarks_array[left_eye_indices]
        right_eye_pts = landmarks_array[right_eye_indices]
        
        left_eye_center = left_eye_pts.mean(axis=0)
        right_eye_center = right_eye_pts.mean(axis=0)
        
        # Extract mouth region
        mouth_pts = landmarks_array[mouth_indices]
        mouth_center = mouth_pts.mean(axis=0)
        
        eye_size = 20
        mouth_size = 25
        
        return {
            'face_roi': face_roi,
            'eye_regions': {
                'left': {
                    'center': tuple(left_eye_center.astype(int)),
                    'roi': (
                        int(left_eye_center[0] - eye_size),
                        int(left_eye_center[1] - eye_size),
                        eye_size * 2,
                        eye_size * 2
                    )
                },
                'right': {
                    'center': tuple(right_eye_center.astype(int)),
                    'roi': (
                        int(right_eye_center[0] - eye_size),
                        int(right_eye_center[1] - eye_size),
                        eye_size * 2,
                        eye_size * 2
                    )
                }
            },
            'mouth_region': {
                'center': tuple(mouth_center.astype(int)),
                'roi': (
                    int(mouth_center[0] - mouth_size),
                    int(mouth_center[1] - mouth_size),
                    mouth_size * 2,
                    mouth_size * 2
                )
            },
            'face_center': (x + w//2, y + h//2)
        }
    
    def process_video_frame(self, frame):
        """
        Process a single frame from thermal video
        
        Args:
            frame: Single frame from thermal camera/video
            
        Returns:
            Detection result dict
        """
        return self.detect_face(frame)
