#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Drowsiness Detection - Test and Comparison Script

Compare different detection strategies and validate system performance.
Test your thermal images with all available strategies.
"""

import cv2
import numpy as np
import sys
import json
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

# Import thermal modules
from modules.thermal_detector import ThermalFaceDetector
from modules.thermal_eye_monitor import ThermalEyeMonitor
from modules.thermal_yawn_detector import ThermalYawnDetector
from modules.thermal_drowsiness_analyzer import ThermalDrowsinessAnalyzer


class ThermalDetectionTester:
    """Test and validate thermal detection strategies"""
    
    def __init__(self):
        self.face_detector = ThermalFaceDetector()
        self.strategies_eye = ['ear', 'thermal', 'temporal', 'hybrid']
        self.strategies_yawn = ['mar', 'thermal', 'temporal', 'contour', 'hybrid']
        
        # Calibration data for threshold option 3
        self.calibration_data = {
            'baseline_eye_closure': None,
            'baseline_yawn': None,
            'baseline_active': None,
            'calibrated': False,
            'calibration_source': 'none'  # Track which method: 'auto', 'manual', or 'none'
        }
    
    def analyze_eyes_only_threshold(self, eye_closure_score):
        """
        Threshold Option 1: Based only on eye closure
        Simple binary decision based on eye closure score
        
        Returns: drowsiness_result dict
        """
        threshold = 0.6  # Eye closure threshold
        drowsy = eye_closure_score >= threshold
        comparison = '>=' if drowsy else '<'
        
        return {
            'method': 'Eyes-Only Threshold',
            'drowsy': drowsy,
            'score': eye_closure_score,
            'threshold': threshold,
            'reasoning': f"Eye closure score {eye_closure_score:.3f} {comparison} {threshold}"
        }
    
    def get_calibration_status(self):
        """Return current calibration status"""
        if not self.calibration_data['calibrated']:
            return "NOT CALIBRATED - Using generic thresholds"
        
        source = self.calibration_data['calibration_source']
        if source == 'auto':
            return "CALIBRATED (Auto Mode) - Using video/image analysis"
        elif source == 'manual':
            return "CALIBRATED (Manual Mode) - Using manually selected min/max values"
        else:
            return "CALIBRATED (Unknown source)"
    
    def analyze_weighted_threshold(self, eye_closure_score, yawn_score):
        """
        Threshold Option 2: Based on weighted combination
        80% weight on eye closure (threshold 0.8)
        20% weight on yawn score (threshold 0.2)
        
        Returns: drowsiness_result dict
        """
        eye_threshold = 0.8
        yawn_threshold = 0.2
        
        # Normalized scoring
        eye_factor = 1.0 if eye_closure_score >= eye_threshold else 0.0
        yawn_factor = 1.0 if yawn_score >= yawn_threshold else 0.0
        
        # Weighted combination: 80% eyes, 20% yawn
        combined_score = (0.8 * eye_factor) + (0.2 * yawn_factor)
        drowsy = combined_score >= 0.5  # Threshold for combined score
        comparison = '>=' if drowsy else '<'
        
        return {
            'method': 'Weighted Threshold (80% Eye, 20% Yawn)',
            'drowsy': drowsy,
            'combined_score': combined_score,
            'eye_component': f"{eye_closure_score:.3f} (threshold: {eye_threshold})",
            'yawn_component': f"{yawn_score:.3f} (threshold: {yawn_threshold})",
            'reasoning': f"Combined score {combined_score:.3f} {comparison} 0.5"
        }
    
    def calibrate_baseline(self, source_path, calibration_type):
        """
        Calibrate baseline values from image or video
        
        Args:
            source_path: Path to calibration image or video
            calibration_type: 'active' (alert state), 'yawning', or 'closed_eyes'
        
        Returns: bool - success status
        """
        print(f"\n[CALIBRATION] Loading {calibration_type} baseline from: {source_path}")
        
        # Detect if it's a video or image
        source_path_obj = Path(source_path)
        is_video = source_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if is_video:
            return self._calibrate_from_video(source_path, calibration_type)
        else:
            return self._calibrate_from_image(source_path, calibration_type)
    
    def _calibrate_from_image(self, image_path, calibration_type):
        """Calibrate from single image"""
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Error: Could not load calibration image")
            return False
        
        # Detect face
        detection = self.face_detector.detect_face(frame)
        if not detection['detected']:
            print("[ERROR] Face detection failed during calibration")
            return False
        
        # Get measurements
        eye_monitor = ThermalEyeMonitor(strategy='thermal')
        eye_result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])
        
        yawn_detector = ThermalYawnDetector(strategy='thermal')
        yawn_result = yawn_detector.analyze_yawn_state(frame, detection['landmarks'], eye_result['combined_score'])
        
        # Store baseline
        baseline_data = {
            'eye_closure': eye_result['combined_score'],
            'yawn': yawn_result['combined_score'],
            'source_type': 'single_image',
            'frames_analyzed': 1
        }
        
        self._store_baseline(calibration_type, baseline_data)
        print(f"  [OK] {calibration_type.capitalize()} baseline (from image)")
        print(f"       Eye: {eye_result['combined_score']:.3f}, Yawn: {yawn_result['combined_score']:.3f}")
        
        return True
    
    def _calibrate_from_video(self, video_path, calibration_type):
        """Calibrate from video by analyzing multiple frames"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return False
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"  Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        print(f"  Sampling every 5th frame for faster analysis...")
        
        eye_scores = []
        yawn_scores = []
        frame_count = 0
        analyzed_count = 0
        
        # Sample frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample every 5th frame
            if frame_count % 5 != 0:
                continue
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            # Detect face
            detection = self.face_detector.detect_face(frame_gray)
            if not detection['detected']:
                continue
            
            # Get measurements
            eye_monitor = ThermalEyeMonitor(strategy='thermal')
            eye_result = eye_monitor.analyze_eye_state(frame_gray, detection['landmarks'])
            
            yawn_detector = ThermalYawnDetector(strategy='thermal')
            yawn_result = yawn_detector.analyze_yawn_state(frame_gray, detection['landmarks'], eye_result['combined_score'])
            
            eye_scores.append(eye_result['combined_score'])
            yawn_scores.append(yawn_result['combined_score'])
            analyzed_count += 1
            
            # Progress indicator
            if analyzed_count % 10 == 0:
                print(f"    Analyzed {analyzed_count} frames...")
        
        cap.release()
        
        if not eye_scores or not yawn_scores:
            print("[ERROR] Failed to extract measurements from video")
            return False
        
        # Calculate statistics
        baseline_data = {
            'eye_closure_mean': float(np.mean(eye_scores)),
            'eye_closure_min': float(np.min(eye_scores)),
            'eye_closure_max': float(np.max(eye_scores)),
            'eye_closure_std': float(np.std(eye_scores)),
            'yawn_mean': float(np.mean(yawn_scores)),
            'yawn_min': float(np.min(yawn_scores)),
            'yawn_max': float(np.max(yawn_scores)),
            'yawn_std': float(np.std(yawn_scores)),
            'source_type': 'video',
            'frames_analyzed': analyzed_count
        }
        
        # Use mean values as the baseline
        baseline_data['eye_closure'] = baseline_data['eye_closure_mean']
        baseline_data['yawn'] = baseline_data['yawn_mean']
        
        self._store_baseline(calibration_type, baseline_data)
        print(f"  [OK] {calibration_type.capitalize()} baseline (from {analyzed_count} video frames)")
        print(f"       Eye closure: {baseline_data['eye_closure']:.3f} (min: {baseline_data['eye_closure_min']:.3f}, max: {baseline_data['eye_closure_max']:.3f}, std: {baseline_data['eye_closure_std']:.3f})")
        print(f"       Yawn: {baseline_data['yawn']:.3f} (min: {baseline_data['yawn_min']:.3f}, max: {baseline_data['yawn_max']:.3f}, std: {baseline_data['yawn_std']:.3f})")
        
        return True
    
    def _store_baseline(self, calibration_type, baseline_data):
        """Store baseline data"""
        if calibration_type == 'active':
            self.calibration_data['baseline_active'] = baseline_data
        elif calibration_type == 'yawning':
            self.calibration_data['baseline_yawn'] = baseline_data
        elif calibration_type == 'closed_eyes':
            self.calibration_data['baseline_eye_closure'] = baseline_data
        
        # Check if all three baselines are available
        if (self.calibration_data['baseline_active'] is not None and
            self.calibration_data['baseline_yawn'] is not None and
            self.calibration_data['baseline_eye_closure'] is not None):
            self.calibration_data['calibrated'] = True
            self.calibration_data['calibration_source'] = 'auto'
            print("\n[OK] CALIBRATION COMPLETE - All baselines captured")
    
    def calibrate_manual_thresholds(self):
        """
        Manual calibration: Pick min/max values from specific images/videos
        
        Allows precise threshold definition:
        - Min eye closure: from alert/active images
        - Max eye closure: from eyes-closed images
        - Min mouth opening: from alert images
        - Max mouth opening: from yawning images
        """
        print("\n" + "="*60)
        print("MANUAL THRESHOLD CALIBRATION")
        print("="*60)
        print("\nThis mode lets you pick exact min/max values from images/videos")
        print("for precise threshold definition.\n")
        
        # Get MIN eye closure (alert state)
        print("[Step 1] MIN Eye Closure - Pick image/video with eyes OPEN")
        min_eye_path = input("  Enter path to min eye closure (alert state): ").strip()
        min_eye_closure = self._extract_measurement(min_eye_path, 'eye')
        
        # Get MAX eye closure (closed eyes)
        print("\n[Step 2] MAX Eye Closure - Pick image/video with eyes CLOSED")
        max_eye_path = input("  Enter path to max eye closure (eyes closed): ").strip()
        max_eye_closure = self._extract_measurement(max_eye_path, 'eye')
        
        # Get MIN mouth opening (alert state)
        print("\n[Step 3] MIN Mouth Opening - Pick image/video with mouth CLOSED")
        min_mouth_path = input("  Enter path to min mouth opening (alert state): ").strip()
        min_mouth_opening = self._extract_measurement(min_mouth_path, 'mouth')
        
        # Get MAX mouth opening (yawning)
        print("\n[Step 4] MAX Mouth Opening - Pick image/video with mouth OPEN (yawning)")
        max_mouth_path = input("  Enter path to max mouth opening (yawning): ").strip()
        max_mouth_opening = self._extract_measurement(max_mouth_path, 'mouth')
        
        # Store thresholds
        self.calibration_data['baseline_active'] = {
            'eye_closure': min_eye_closure,
            'yawn': min_mouth_opening,
            'source_type': 'manual_min'
        }
        self.calibration_data['baseline_eye_closure'] = {
            'eye_closure': max_eye_closure,
            'yawn': 0.0,
            'source_type': 'manual_max'
        }
        self.calibration_data['baseline_yawn'] = {
            'eye_closure': 0.0,
            'yawn': max_mouth_opening,
            'source_type': 'manual_max'
        }
        self.calibration_data['calibrated'] = True
        self.calibration_data['calibration_source'] = 'manual'
        
        print("\n[OK] MANUAL CALIBRATION COMPLETE")
        print(f"  Min Eye Closure: {min_eye_closure:.3f}")
        print(f"  Max Eye Closure: {max_eye_closure:.3f}")
        print(f"  Min Mouth Opening: {min_mouth_opening:.3f}")
        print(f"  Max Mouth Opening: {max_mouth_opening:.3f}")
    
    def _extract_measurement(self, source_path, measurement_type):
        """Extract single measurement from image or video"""
        source_path_obj = Path(source_path)
        is_video = source_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if is_video:
            return self._extract_from_video(source_path, measurement_type)
        else:
            return self._extract_from_image(source_path, measurement_type)
    
    def _extract_from_image(self, image_path, measurement_type):
        """Extract measurement from single image"""
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"  [ERROR] Could not load image: {image_path}")
            return 0.5
        
        detection = self.face_detector.detect_face(frame)
        if not detection['detected']:
            print(f"  [ERROR] Face not detected in {image_path}")
            return 0.5
        
        if measurement_type == 'eye':
            eye_monitor = ThermalEyeMonitor(strategy='thermal')
            eye_result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])
            value = eye_result['combined_score']
            print(f"  ✓ Eye closure: {value:.3f}")
            return value
        else:  # mouth
            yawn_detector = ThermalYawnDetector(strategy='thermal')
            eye_monitor = ThermalEyeMonitor(strategy='thermal')
            eye_result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])
            yawn_result = yawn_detector.analyze_yawn_state(frame, detection['landmarks'], eye_result['combined_score'])
            value = yawn_result['combined_score']
            print(f"  ✓ Mouth opening (yawn): {value:.3f}")
            return value
    
    def _extract_from_video(self, video_path, measurement_type):
        """Extract max measurement from video frames"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"  [ERROR] Could not open video: {video_path}")
            return 0.5
        
        measurements = []
        frame_count = 0
        analyzed_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % 5 != 0:  # Sample every 5th frame
                continue
            
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            detection = self.face_detector.detect_face(frame_gray)
            if not detection['detected']:
                continue
            
            if measurement_type == 'eye':
                eye_monitor = ThermalEyeMonitor(strategy='thermal')
                eye_result = eye_monitor.analyze_eye_state(frame_gray, detection['landmarks'])
                measurements.append(eye_result['combined_score'])
            else:  # mouth
                yawn_detector = ThermalYawnDetector(strategy='thermal')
                eye_monitor = ThermalEyeMonitor(strategy='thermal')
                eye_result = eye_monitor.analyze_eye_state(frame_gray, detection['landmarks'])
                yawn_result = yawn_detector.analyze_yawn_state(frame_gray, detection['landmarks'], eye_result['combined_score'])
                measurements.append(yawn_result['combined_score'])
            
            analyzed_count += 1
        
        cap.release()
        
        if not measurements:
            print(f"  [ERROR] Could not extract measurements from video")
            return 0.5
        
        # Return MAX value (for manual thresholds, we want the peak)
        max_value = max(measurements)
        mean_value = np.mean(measurements)
        print(f"  ✓ {measurement_type.capitalize()} - Max: {max_value:.3f}, Mean: {mean_value:.3f} (from {analyzed_count} frames)")
        return max_value
    
    def analyze_calibration_threshold(self, eye_closure_score, yawn_score):
        """
        Threshold Option 3: Based on calibration baseline
        Uses ratios from calibration data to determine drowsiness
        
        Returns: drowsiness_result dict
        """
        if not self.calibration_data['calibrated']:
            return {
                'method': 'Calibration-Based Threshold',
                'drowsy': False,
                'error': 'Not calibrated - please calibrate first',
                'reasoning': 'Calibration data not available'
            }
        
        # Get baselines
        active = self.calibration_data['baseline_active']
        yawning = self.calibration_data['baseline_yawn']
        closed = self.calibration_data['baseline_eye_closure']
        
        # Calculate ratios relative to baselines
        # If current eye_closure is between active and closed, interpolate
        eye_ratio = (eye_closure_score - active['eye_closure']) / (closed['eye_closure'] - active['eye_closure'] + 1e-6)
        eye_ratio = np.clip(eye_ratio, 0.0, 1.0)
        
        yawn_ratio = (yawn_score - active['yawn']) / (yawning['yawn'] - active['yawn'] + 1e-6)
        yawn_ratio = np.clip(yawn_ratio, 0.0, 1.0)
        
        # Combined score: weighted average
        combined_score = 0.7 * eye_ratio + 0.3 * yawn_ratio
        drowsy = combined_score >= 0.5
        comparison = '>=' if drowsy else '<'
        
        return {
            'method': 'Calibration-Based Threshold',
            'drowsy': drowsy,
            'combined_score': combined_score,
            'eye_ratio': eye_ratio,
            'yawn_ratio': yawn_ratio,
            'baselines': {
                'active': active,
                'yawning': yawning,
                'closed_eyes': closed
            },
            'reasoning': f"Combined ratio {combined_score:.3f} {comparison} 0.5 (Eye: {eye_ratio:.3f}, Yawn: {yawn_ratio:.3f})"
        }
    
    def test_single_image(self, image_path, show_results=True, threshold_option='all'):
        """
        Test detection on single thermal image
        
        Args:
            image_path: Path to thermal image
            show_results: Whether to display results
            threshold_option: Which threshold to use ('all', '1', '2', '3')
            
        Returns:
            dict with comprehensive analysis
        """
        print(f"\n{'='*80}")
        print(f"Testing: {image_path}")
        print(f"{'='*80}")
        
        # Load image
        frame = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if frame is None:
            print(f"Error: Could not load image {image_path}")
            return None
        
        print(f"Image shape: {frame.shape}, dtype: {frame.dtype}")
        
        # Detect face
        detection = self.face_detector.detect_face(frame)
        
        if not detection['detected']:
            print("[FAILED] FACE DETECTION FAILED")
            return {'detected': False}
        
        print(f"[OK] Face detected at ROI: {detection['face_roi']}")
        print(f"  Confidence: {detection['confidence']:.2f}")
        print(f"  Method: {detection['method']}")
        
        # Test all eye strategies
        print(f"\n{'Eye Detection Strategies':^80}")
        print("-" * 80)
        
        eye_results = {}
        eye_data = []
        
        for strategy in self.strategies_eye:
            eye_monitor = ThermalEyeMonitor(strategy=strategy)
            result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])
            eye_results[strategy] = result
            
            eye_data.append([
                strategy.upper(),
                f"{result['combined_score']:.3f}",
                f"{result['perclos']:.1%}",
                "[YES]" if result['eyes_closed'] else "[NO]",
                "[YES]" if result['sleep_detected'] else "[NO]"
            ])
        
        print(tabulate(
            eye_data,
            headers=['Strategy', 'Score', 'PERCLOS', 'Closed', 'Sleep'],
            tablefmt='grid'
        ))
        
        # Test all yawn strategies with eye closure score from THERMAL strategy (most accurate for closed eyes)
        print(f"\n{'Yawn Detection Strategies':^80}")
        print("-" * 80)
        
        yawn_results = {}
        yawn_data = []
        # Use THERMAL eye score for better detection of closed eyes (0 or 1.0 scores)
        eye_closure_score_thermal = eye_results['thermal']['combined_score']
        
        for strategy in self.strategies_yawn:
            yawn_detector = ThermalYawnDetector(strategy=strategy)
            result = yawn_detector.analyze_yawn_state(frame, detection['landmarks'], eye_closure_score_thermal)
            yawn_results[strategy] = result
            
            yawn_data.append([
                strategy.upper(),
                f"{result['combined_score']:.3f}",
                result['yawn_count'],
                "[YES]" if result['yawning'] else "[NO]",
                f"{result['yawn_rate']:.2f}/5min"
            ])
        
        print(tabulate(
            yawn_data,
            headers=['Strategy', 'Score', 'Count', 'Active', 'Rate'],
            tablefmt='grid'
        ))
        
        # Drowsiness analysis with hybrid strategies
        print(f"\n{'Drowsiness Analysis (Hybrid Eye + Hybrid Yawn)':^80}")
        print("-" * 80)
        
        analyzer = ThermalDrowsinessAnalyzer()
        
        # Create analysis input from hybrid results
        hybrid_eye = eye_results['hybrid']
        hybrid_yawn = yawn_results['hybrid']
        thermal_eye = eye_results['thermal']  # Use thermal for better closed-eye detection
        
        # Use thermal eye detection for overall eye_closure_score if it's higher
        # This captures cases where thermal clearly detects closure but hybrid doesn't
        eye_closure_for_analysis = max(hybrid_eye['combined_score'], thermal_eye['combined_score'])
        
        analysis = analyzer.analyze_drowsiness(
            eye_closure_score=eye_closure_for_analysis,
            yawn_score=hybrid_yawn['combined_score'],
            ear_score=eye_results['ear']['combined_score'],
            perclos=hybrid_eye['perclos'],
            blink_rate=hybrid_eye['blink_rate'],
            yawn_count=hybrid_yawn['yawn_count'],
            yawn_frequency=hybrid_yawn['yawn_rate']
        )
        
        print(f"Overall Drowsiness Score: {analysis['drowsiness_score']:.3f}")
        print(f"Drowsiness Level: {analysis['drowsiness_level']}")
        print(f"Alert Status: {analysis['alert_status']}")
        print(f"Confidence: {analysis['confidence']:.2f}")
        
        print(f"\nComponent Breakdown:")
        for factor, score in analysis['component_scores'].items():
            print(f"  {factor:30} {score:.3f}")
        
        print(f"\nContributing Factors:")
        for i, factor in enumerate(analysis['contributing_factors'], 1):
            print(f"  {i}. {factor}")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        # Threshold-based analysis
        print(f"\n{'='*80}")
        print(f"THRESHOLD-BASED DROWSINESS ANALYSIS")
        print(f"{'='*80}")
        
        threshold_results = []
        
        # Option 1: Eyes only
        if threshold_option in ['all', '1']:
            result1 = self.analyze_eyes_only_threshold(eye_closure_for_analysis)
            threshold_results.append(result1)
            print(f"\n[OPTION 1] Eyes-Only Threshold")
            print(f"  Eye Closure Score: {eye_closure_for_analysis:.3f}")
            print(f"  Threshold: {result1['threshold']}")
            print(f"  Status: {'DROWSY' if result1['drowsy'] else 'ALERT'}")
            print(f"  Reasoning: {result1['reasoning']}")
        
        # Option 2: Weighted (80% eye, 20% yawn)
        if threshold_option in ['all', '2']:
            result2 = self.analyze_weighted_threshold(eye_closure_for_analysis, hybrid_yawn['combined_score'])
            threshold_results.append(result2)
            print(f"\n[OPTION 2] Weighted Threshold (80% Eye Closure, 20% Yawn)")
            print(f"  Eye Closure Score: {eye_closure_for_analysis:.3f} (threshold: 0.8)")
            print(f"  Yawn Score: {hybrid_yawn['combined_score']:.3f} (threshold: 0.2)")
            print(f"  Combined Score: {result2['combined_score']:.3f}")
            print(f"  Status: {'DROWSY' if result2['drowsy'] else 'ALERT'}")
            print(f"  Reasoning: {result2['reasoning']}")
        
        # Option 3: Calibration-based
        if threshold_option in ['all', '3']:
            result3 = self.analyze_calibration_threshold(eye_closure_for_analysis, hybrid_yawn['combined_score'])
            threshold_results.append(result3)
            print(f"\n[OPTION 3] Calibration-Based Threshold")
            if 'error' in result3:
                print(f"  Status: {result3['error']}")
            else:
                print(f"  Eye Closure Ratio: {result3['eye_ratio']:.3f}")
                print(f"  Yawn Ratio: {result3['yawn_ratio']:.3f}")
                print(f"  Combined Score: {result3['combined_score']:.3f}")
                print(f"  Status: {'DROWSY' if result3['drowsy'] else 'ALERT'}")
                print(f"  Reasoning: {result3['reasoning']}")
        
        # Comparison table
        if len(threshold_results) > 1:
            print(f"\n{'Threshold Comparison':^80}")
            comparison_data = []
            for result in threshold_results:
                if 'error' not in result:
                    status = '[DROWSY]' if result['drowsy'] else '[ALERT]'
                    score_display = f"{result.get('combined_score', result.get('score', 0.0)):.3f}"
                else:
                    status = '[ERROR]'
                    score_display = 'N/A'
                
                comparison_data.append([
                    result['method'],
                    score_display,
                    status
                ])
            
            print(tabulate(
                comparison_data,
                headers=['Method', 'Score', 'Status'],
                tablefmt='grid'
            ))
        
        return analysis
    
    def test_video_stream(self, video_path, threshold_option='all'):
        """Test video file (all frames) with all strategies"""
        print(f"\n{'='*80}")
        print(f"TESTING VIDEO: {video_path}")
        print(f"{'='*80}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        print(f"Threshold Mode: {threshold_option}")
        if self.calibration_data['calibrated']:
            print(f"Calibration: {self.get_calibration_status()}")
        else:
            print(f"Calibration: NOT CALIBRATED - Using generic thresholds")
        print()
        
        frame_num = 0
        drowsy_count = 0
        
        # Track scores for all strategies
        eye_strategy_scores = {strategy: [] for strategy in self.strategies_eye}
        yawn_strategy_scores = {strategy: [] for strategy in self.strategies_yawn}
        
        # Create monitors for all strategies
        eye_monitors = {strategy: ThermalEyeMonitor(strategy=strategy) for strategy in self.strategies_eye}
        yawn_detectors = {strategy: ThermalYawnDetector(strategy=strategy) for strategy in self.strategies_yawn}
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Sample every 5th frame for analysis
            if frame_num % 5 != 0:
                continue
            
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            # Detect face
            detection = self.face_detector.detect_face(frame_gray)
            if not detection['detected']:
                continue
            
            # Analyze with all eye strategies
            eye_results = {}
            for strategy in self.strategies_eye:
                result = eye_monitors[strategy].analyze_eye_state(frame_gray, detection['landmarks'])
                eye_results[strategy] = result
                eye_strategy_scores[strategy].append(result['combined_score'])
            
            # Use thermal eye score for yawn analysis (most accurate for closed eyes)
            eye_closure_thermal = eye_results['thermal']['combined_score']
            
            # Analyze with all yawn strategies
            yawn_results = {}
            for strategy in self.strategies_yawn:
                result = yawn_detectors[strategy].analyze_yawn_state(frame_gray, detection['landmarks'], eye_closure_thermal)
                yawn_results[strategy] = result
                yawn_strategy_scores[strategy].append(result['combined_score'])
            
            # Use hybrid results for threshold analysis
            hybrid_eye = eye_results.get('hybrid', eye_results['ear'])
            hybrid_yawn = yawn_results.get('hybrid', yawn_results['mar'])
            eye_closure = hybrid_eye['combined_score']
            yawn_score = hybrid_yawn['combined_score']
            
            # Analyze thresholds
            status = 'ALERT'
            details = ""
            
            if threshold_option == 'all' or threshold_option == '1':
                result1 = self.analyze_eyes_only_threshold(eye_closure)
                if result1['drowsy']:
                    status = 'DROWSY'
                    drowsy_count += 1
                    details += f"[M1: DROWSY] "
            
            if threshold_option == 'all' or threshold_option == '2':
                result2 = self.analyze_weighted_threshold(eye_closure, yawn_score)
                if result2['drowsy']:
                    status = 'DROWSY'
                    drowsy_count += 1
                    details += f"[M2: DROWSY] "
            
            if threshold_option == 'all' or threshold_option == '3':
                if self.calibration_data['calibrated']:
                    result3 = self.analyze_calibration_threshold(eye_closure, yawn_score)
                    if result3['drowsy']:
                        status = 'DROWSY'
                        drowsy_count += 1
                        details += f"[M3: DROWSY] "
            
            # Print frame with hybrid results (more readable than all strategies)
            if frame_num % 50 == 0:  # Print every 50th frame analyzed
                print(f"Frame {frame_num:4d}/{total_frames} - {status:6s} | Eye(H): {eye_closure:.3f} | Yawn(H): {yawn_score:.3f} {details}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Video Analysis Complete:")
        print(f"Total frames analyzed: {frame_num}")
        print(f"Drowsy detections: {drowsy_count}")
        
        print(f"\n{'Eye Detection Strategy Statistics':^80}")
        print("-" * 80)
        eye_stats = []
        for strategy in self.strategies_eye:
            scores = eye_strategy_scores[strategy]
            if scores:
                eye_stats.append([
                    strategy.upper(),
                    f"{np.mean(scores):.3f}",
                    f"{np.min(scores):.3f}",
                    f"{np.max(scores):.3f}",
                    f"{np.std(scores):.3f}"
                ])
        
        if eye_stats:
            print(tabulate(
                eye_stats,
                headers=['Strategy', 'Mean', 'Min', 'Max', 'Std Dev'],
                tablefmt='grid'
            ))
        
        print(f"\n{'Yawn Detection Strategy Statistics':^80}")
        print("-" * 80)
        yawn_stats = []
        for strategy in self.strategies_yawn:
            scores = yawn_strategy_scores[strategy]
            if scores:
                yawn_stats.append([
                    strategy.upper(),
                    f"{np.mean(scores):.3f}",
                    f"{np.min(scores):.3f}",
                    f"{np.max(scores):.3f}",
                    f"{np.std(scores):.3f}"
                ])
        
        if yawn_stats:
            print(tabulate(
                yawn_stats,
                headers=['Strategy', 'Mean', 'Min', 'Max', 'Std Dev'],
                tablefmt='grid'
            ))
        print(f"{'='*80}")
    
    def test_webcam_stream(self, camera_id=0, threshold_option='all'):
        """Real-time webcam testing"""
        print(f"\n{'='*80}")
        print(f"WEBCAM REAL-TIME INFERENCE (Camera {camera_id})")
        print(f"{'='*80}")
        print(f"Press 'q' to quit, 'space' to pause")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return None
        
        frame_num = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_num += 1
            
            # Convert to grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            detection = self.face_detector.detect_face(frame_gray)
            
            status = "FACE DETECTED" if detection['detected'] else "NO FACE"
            
            # Display
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow(f'Thermal Drowsiness - Webcam {camera_id}', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Webcam inference complete. Frames processed: {frame_num}")


def main():
    """Main test routine"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test thermal drowsiness detection')
    parser.add_argument('--image', help='Test single image')
    parser.add_argument('--video', help='Test video file')
    parser.add_argument('--webcam', type=int, default=-1, help='Webcam index (0 for default camera). Use -1 to disable')
    parser.add_argument('--dir', help='Test directory of images')
    parser.add_argument('--threshold', choices=['all', '1', '2', '3'], default='all',
                        help='Threshold option: 1=Eyes-Only, 2=Weighted(80/20), 3=Calibration-Based')
    parser.add_argument('--calibrate-active', help='Auto calibration: path to active/alert state image or video')
    parser.add_argument('--calibrate-yawn', help='Auto calibration: path to yawning state image or video')
    parser.add_argument('--calibrate-closed', help='Auto calibration: path to eyes-closed state image or video')
    parser.add_argument('--manual-calibrate', action='store_true', help='Enter manual calibration mode (pick min/max from images)')
    parser.add_argument('--calibration-mode', choices=['auto', 'manual', 'none'], default='none',
                        help='Choose calibration mode: auto=use auto calibration, manual=use manual calibration, none=no calibration')
    parser.add_argument('--compare', action='store_true', help='Compare all strategies')
    
    args = parser.parse_args()
    
    tester = ThermalDetectionTester()
    
    # Handle calibration based on --calibration-mode flag
    if args.calibration_mode == 'manual' or args.manual_calibrate:
        print(f"\n{'='*80}")
        print("MANUAL CALIBRATION MODE")
        print(f"{'='*80}")
        tester.calibrate_manual_thresholds()
    
    elif args.calibration_mode == 'auto' or (args.calibrate_active or args.calibrate_yawn or args.calibrate_closed):
        print(f"\n{'='*80}")
        print("AUTO CALIBRATION MODE")
        print(f"{'='*80}")
        
        if args.calibrate_active:
            tester.calibrate_baseline(args.calibrate_active, 'active')
        if args.calibrate_yawn:
            tester.calibrate_baseline(args.calibrate_yawn, 'yawning')
        if args.calibrate_closed:
            tester.calibrate_baseline(args.calibrate_closed, 'closed_eyes')
    
    # Show calibration status before testing
    if tester.calibration_data['calibrated']:
        print(f"\n[STATUS] {tester.get_calibration_status()}")
    elif args.calibration_mode == 'none':
        print(f"\n[STATUS] NOT CALIBRATED - Using generic thresholds (Methods 1 & 2)")
    
    if args.image:
        result = tester.test_single_image(args.image, threshold_option=args.threshold)
    elif args.video:
        result = tester.test_video_stream(args.video, threshold_option=args.threshold)
    elif args.webcam >= 0:
        result = tester.test_webcam_stream(args.webcam, threshold_option=args.threshold)
    elif args.dir:
        # Test all images in directory
        image_dir = Path(args.dir)
        for image_file in sorted(image_dir.glob('*.png')):
            tester.test_single_image(str(image_file), threshold_option=args.threshold)
    else:
        # Interactive mode
        print("Thermal Detection Testing Tool")
        print("1. Calibrate system (3-point calibration)")
        print("2. Calibrate system (MANUAL - pick min/max from images)")
        print("3. Test single image")
        print("4. Test video file")
        print("5. Test webcam (real-time)")
        print("6. Test directory")
        print("7. Exit")
        
        choice = input("Select option: ").strip()
        
        if choice == '1':
            print("\n3-POINT CALIBRATION")
            print("-" * 40)
            active_path = input("Enter path to ACTIVE/ALERT state (image or video): ").strip()
            tester.calibrate_baseline(active_path, 'active')
            
            yawn_path = input("Enter path to YAWNING state (image or video): ").strip()
            tester.calibrate_baseline(yawn_path, 'yawning')
            
            closed_path = input("Enter path to CLOSED EYES state (image or video): ").strip()
            tester.calibrate_baseline(closed_path, 'closed_eyes')
        
        elif choice == '2':
            tester.calibrate_manual_thresholds()
        
        elif choice == '3':
            image_path = input("Enter image path: ").strip()
            threshold_choice = input("Select threshold option (all/1/2/3): ").strip()
            tester.test_single_image(image_path, threshold_option=threshold_choice)
        elif choice == '4':
            video_path = input("Enter video path: ").strip()
            threshold_choice = input("Select threshold option (all/1/2/3): ").strip()
            tester.test_video_stream(video_path, threshold_option=threshold_choice)
        elif choice == '5':
            webcam_id = input("Enter camera index (default 0): ").strip()
            webcam_id = int(webcam_id) if webcam_id else 0
            threshold_choice = input("Select threshold option (all/1/2/3): ").strip()
            tester.test_webcam_stream(webcam_id, threshold_option=threshold_choice)
        elif choice == '6':
            dir_path = input("Enter directory path: ").strip()
            threshold_choice = input("Select threshold option (all/1/2/3): ").strip()
            image_dir = Path(dir_path)
            for image_file in sorted(image_dir.glob('*.png')):
                tester.test_single_image(str(image_file), threshold_option=threshold_choice)


if __name__ == '__main__':
    main()
