#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermal Driver Drowsiness Detection System

Main entry point for thermal camera-based driver drowsiness detection.
Supports multiple detection strategies optimized for thermal infrared imaging.

Usage:
    python thermal_driver_drowsiness_detection.py --video <path_to_thermal_video>
    python thermal_driver_drowsiness_detection.py --camera <camera_index>
    python thermal_driver_drowsiness_detection.py --image <path_to_thermal_image>
"""

import cv2
import numpy as np
import time
import argparse
import os
from datetime import datetime
from pathlib import Path

# Import thermal-specific modules
from modules.thermal_detector import ThermalFaceDetector
from modules.thermal_eye_monitor import ThermalEyeMonitor
from modules.thermal_yawn_detector import ThermalYawnDetector
from modules.thermal_drowsiness_analyzer import ThermalDrowsinessAnalyzer
from modules.config_loader import ConfigLoader
from modules.alert_system import AlertSystem
from modules.report_generator import ReportGenerator


class ThresholdCalibrationEngine:
    """Threshold-based drowsiness logic and optional calibration"""

    def __init__(self, face_detector=None):
        # Reuse existing face detector when possible
        self.face_detector = face_detector or ThermalFaceDetector()

        # Calibration data for threshold option 3
        self.calibration_data = {
            'baseline_eye_closure': None,
            'baseline_yawn': None,
            'baseline_active': None,
            'calibrated': False,
            'calibration_source': 'none'
        }

    def analyze_eyes_only_threshold(self, eye_closure_score):
        """Threshold Option 1: Based only on eye closure"""
        threshold = 0.6
        drowsy = eye_closure_score >= threshold
        comparison = '>=' if drowsy else '<'

        return {
            'method': 'Eyes-Only Threshold',
            'drowsy': drowsy,
            'score': eye_closure_score,
            'threshold': threshold,
            'reasoning': f"Eye closure score {eye_closure_score:.3f} {comparison} {threshold}"
        }

    def analyze_weighted_threshold(self, eye_closure_score, yawn_score):
        """Threshold Option 2: Weighted combination of eye closure and yawn"""
        eye_threshold = 0.8
        yawn_threshold = 0.2

        eye_factor = 1.0 if eye_closure_score >= eye_threshold else 0.0
        yawn_factor = 1.0 if yawn_score >= yawn_threshold else 0.0

        combined_score = (0.8 * eye_factor) + (0.2 * yawn_factor)
        drowsy = combined_score >= 0.5
        comparison = '>=' if drowsy else '<'

        return {
            'method': 'Weighted Threshold (80% Eye, 20% Yawn)',
            'drowsy': drowsy,
            'combined_score': combined_score,
            'eye_component': f"{eye_closure_score:.3f} (threshold: {eye_threshold})",
            'yawn_component': f"{yawn_score:.3f} (threshold: {yawn_threshold})",
            'reasoning': f"Combined score {combined_score:.3f} {comparison} 0.5"
        }

    def get_calibration_status(self):
        """Return current calibration status as human-readable string"""
        if not self.calibration_data['calibrated']:
            return "NOT CALIBRATED - Using generic thresholds"

        source = self.calibration_data['calibration_source']
        if source == 'auto':
            return "CALIBRATED (Auto Mode) - Using video/image analysis"
        elif source == 'manual':
            return "CALIBRATED (Manual Mode) - Using manually selected min/max values"
        else:
            return "CALIBRATED (Unknown source)"

    def calibrate_baseline(self, source_path, calibration_type):
        """Calibrate baseline values from image or video"""
        print(f"\n[CALIBRATION] Loading {calibration_type} baseline from: {source_path}")

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
            print("Error: Could not load calibration image")
            return False

        detection = self.face_detector.detect_face(frame)
        if not detection['detected']:
            print("[ERROR] Face detection failed during calibration")
            return False

        eye_monitor = ThermalEyeMonitor(strategy='thermal')
        eye_result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])

        yawn_detector = ThermalYawnDetector(strategy='thermal')
        yawn_result = yawn_detector.analyze_yawn_state(frame, detection['landmarks'], eye_result['combined_score'])

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
        print("  Sampling every 5th frame for faster analysis...")

        eye_scores = []
        yawn_scores = []
        frame_count = 0
        analyzed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % 5 != 0:
                continue

            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame

            detection = self.face_detector.detect_face(frame_gray)
            if not detection['detected']:
                continue

            eye_monitor = ThermalEyeMonitor(strategy='thermal')
            eye_result = eye_monitor.analyze_eye_state(frame_gray, detection['landmarks'])

            yawn_detector = ThermalYawnDetector(strategy='thermal')
            yawn_result = yawn_detector.analyze_yawn_state(frame_gray, detection['landmarks'], eye_result['combined_score'])

            eye_scores.append(eye_result['combined_score'])
            yawn_scores.append(yawn_result['combined_score'])
            analyzed_count += 1

            if analyzed_count % 10 == 0:
                print(f"    Analyzed {analyzed_count} frames...")

        cap.release()

        if not eye_scores or not yawn_scores:
            print("[ERROR] Failed to extract measurements from video")
            return False

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

        baseline_data['eye_closure'] = baseline_data['eye_closure_mean']
        baseline_data['yawn'] = baseline_data['yawn_mean']

        self._store_baseline(calibration_type, baseline_data)
        print(f"  [OK] {calibration_type.capitalize()} baseline (from {analyzed_count} video frames)")
        print(f"       Eye closure: {baseline_data['eye_closure']:.3f} (min: {baseline_data['eye_closure_min']:.3f}, max: {baseline_data['eye_closure_max']:.3f}, std: {baseline_data['eye_closure_std']:.3f})")
        print(f"       Yawn: {baseline_data['yawn']:.3f} (min: {baseline_data['yawn_min']:.3f}, max: {baseline_data['yawn_max']:.3f}, std: {baseline_data['yawn_std']:.3f})")

        return True

    def _store_baseline(self, calibration_type, baseline_data):
        """Store baseline data and update calibration state"""
        if calibration_type == 'active':
            self.calibration_data['baseline_active'] = baseline_data
        elif calibration_type == 'yawning':
            self.calibration_data['baseline_yawn'] = baseline_data
        elif calibration_type == 'closed_eyes':
            self.calibration_data['baseline_eye_closure'] = baseline_data

        if (
            self.calibration_data['baseline_active'] is not None
            and self.calibration_data['baseline_yawn'] is not None
            and self.calibration_data['baseline_eye_closure'] is not None
        ):
            self.calibration_data['calibrated'] = True
            self.calibration_data['calibration_source'] = 'auto'
            print("\n[OK] CALIBRATION COMPLETE - All baselines captured")

    def calibrate_manual_thresholds(self):
        """Manual calibration using specific images/videos for min/max values"""
        print("\n" + "=" * 60)
        print("MANUAL CALIBRATION MODE")
        print("=" * 60)
        print("\nThis mode lets you pick exact min/max values from images/videos")
        print("for precise threshold definition.\n")

        print("[Step 1] MIN Eye Closure - Pick image/video with eyes OPEN")
        min_eye_path = input("  Enter path to min eye closure (alert state): ").strip()
        min_eye_closure = self._extract_measurement(min_eye_path, 'eye')

        print("\n[Step 2] MAX Eye Closure - Pick image/video with eyes CLOSED")
        max_eye_path = input("  Enter path to max eye closure (eyes closed): ").strip()
        max_eye_closure = self._extract_measurement(max_eye_path, 'eye')

        print("\n[Step 3] MIN Mouth Opening - Pick image/video with mouth CLOSED")
        min_mouth_path = input("  Enter path to min mouth opening (alert state): ").strip()
        min_mouth_opening = self._extract_measurement(min_mouth_path, 'mouth')

        print("\n[Step 4] MAX Mouth Opening - Pick image/video with mouth OPEN (yawning)")
        max_mouth_path = input("  Enter path to max mouth opening (yawning): ").strip()
        max_mouth_opening = self._extract_measurement(max_mouth_path, 'mouth')

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
            print(f"  \u2713 Eye closure: {value:.3f}")
            return value
        else:
            yawn_detector = ThermalYawnDetector(strategy='thermal')
            eye_monitor = ThermalEyeMonitor(strategy='thermal')
            eye_result = eye_monitor.analyze_eye_state(frame, detection['landmarks'])
            yawn_result = yawn_detector.analyze_yawn_state(frame, detection['landmarks'], eye_result['combined_score'])
            value = yawn_result['combined_score']
            print(f"  \u2713 Mouth opening (yawn): {value:.3f}")
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
            if frame_count % 5 != 0:
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
            else:
                yawn_detector = ThermalYawnDetector(strategy='thermal')
                eye_monitor = ThermalEyeMonitor(strategy='thermal')
                eye_result = eye_monitor.analyze_eye_state(frame_gray, detection['landmarks'])
                yawn_result = yawn_detector.analyze_yawn_state(frame_gray, detection['landmarks'], eye_result['combined_score'])
                measurements.append(yawn_result['combined_score'])

            analyzed_count += 1

        cap.release()

        if not measurements:
            print("  [ERROR] Could not extract measurements from video")
            return 0.5

        max_value = max(measurements)
        mean_value = np.mean(measurements)
        print(f"  \u2713 {measurement_type.capitalize()} - Max: {max_value:.3f}, Mean: {mean_value:.3f} (from {analyzed_count} frames)")
        return max_value

    def analyze_calibration_threshold(self, eye_closure_score, yawn_score):
        """Threshold Option 3: Based on calibration baseline"""
        if not self.calibration_data['calibrated']:
            return {
                'method': 'Calibration-Based Threshold',
                'drowsy': False,
                'error': 'Not calibrated - please calibrate first',
                'reasoning': 'Calibration data not available'
            }

        active = self.calibration_data['baseline_active']
        yawning = self.calibration_data['baseline_yawn']
        closed = self.calibration_data['baseline_eye_closure']

        eye_ratio = (eye_closure_score - active['eye_closure']) / (closed['eye_closure'] - active['eye_closure'] + 1e-6)
        eye_ratio = np.clip(eye_ratio, 0.0, 1.0)

        yawn_ratio = (yawn_score - active['yawn']) / (yawning['yawn'] - active['yawn'] + 1e-6)
        yawn_ratio = np.clip(yawn_ratio, 0.0, 1.0)

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


def create_parser():
    """Create argument parser"""
    parser = argparse.ArgumentParser(
        description='Thermal Driver Drowsiness Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Real-time from thermal camera
  python thermal_driver_drowsiness_detection.py --camera 0
  
  # From thermal video file
  python thermal_driver_drowsiness_detection.py --video thermal_video.mp4
  
  # Single thermal image analysis
  python thermal_driver_drowsiness_detection.py --image thermal_frame.png
  
  # With specific strategy
  python thermal_driver_drowsiness_detection.py --camera 0 --eye-strategy hybrid --yawn-strategy thermal
  
  # With audio alerts
  python thermal_driver_drowsiness_detection.py --camera 0 --alert
        """
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--camera', type=int, help='Camera index for thermal camera')
    input_group.add_argument('--video', type=str, help='Path to thermal video file')
    input_group.add_argument('--image', type=str, help='Path to thermal image file')
    
    # Detection strategies
    parser.add_argument(
        '--eye-strategy',
        choices=['ear', 'thermal', 'temporal', 'hybrid'],
        default='hybrid',
        help='Eye detection strategy (default: hybrid)'
    )
    parser.add_argument(
        '--yawn-strategy',
        choices=['mar', 'thermal', 'temporal', 'contour', 'hybrid'],
        default='hybrid',
        help='Yawn detection strategy (default: hybrid)'
    )
    
    # Output options
    parser.add_argument('--show', action='store_true', help='Display video with annotations')
    parser.add_argument('--record', action='store_true', help='Record output video')
    parser.add_argument('--report', action='store_true', default=True, help='Generate session report')
    parser.add_argument('--alert', action='store_true', help='Enable audio alerts')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--output-dir', type=str, default='thermal_results', help='Output directory')
    
    # Processing options
    parser.add_argument('--fps', type=int, default=15, help='Expected FPS for temporal analysis')
    parser.add_argument('--skip-frames', type=int, default=0, help='Process every Nth frame')

    # Threshold-based options (Eyes-Only / weighted / calibration-based)
    parser.add_argument(
        '--threshold',
        choices=['none', 'all', '1', '2', '3'],
        default='none',
        help='Threshold option: 1=Eyes-Only, 2=Weighted(80/20), 3=Calibration-Based, all=run all, none=disable'
    )
    parser.add_argument('--calibrate-active', help='Auto calibration: path to active/alert state image or video')
    parser.add_argument('--calibrate-yawn', help='Auto calibration: path to yawning state image or video')
    parser.add_argument('--calibrate-closed', help='Auto calibration: path to eyes-closed state image or video')
    parser.add_argument(
        '--manual-calibrate',
        action='store_true',
        help='Enter manual calibration mode (pick min/max from images)'
    )
    parser.add_argument(
        '--calibration-mode',
        choices=['auto', 'manual', 'none'],
        default='none',
        help='Calibration mode for threshold 3 (auto/manual/none)'
    )
    
    return parser


class ThermalDriverDrowsinessDetection:
    """Main system class for thermal drowsiness detection"""
    
    def __init__(self, args, config):
        """
        Initialize thermal detection system
        
        Args:
            args: Parsed command-line arguments
            config: Configuration object
        """
        self.args = args
        self.config = config
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        print("Initializing thermal detection system...")
        
        self.face_detector = ThermalFaceDetector(
            min_face_size=50,
            use_mediapipe_fallback=True
        )
        
        self.eye_monitor = ThermalEyeMonitor(
            strategy=args.eye_strategy,
            fps=args.fps
        )
        
        self.yawn_detector = ThermalYawnDetector(
            strategy=args.yawn_strategy,
            fps=args.fps
        )
        
        # Fixed: Pass config as dictionary
        config_dict = config.config if hasattr(config, 'config') else {}
        
        self.drowsiness_analyzer = ThermalDrowsinessAnalyzer(
            config=config_dict,
            eye_monitor=self.eye_monitor,
            yawn_detector=self.yawn_detector
        )

        self.alert_system = AlertSystem(enable_audio=args.alert)
        self.report_generator = ReportGenerator() if args.report else None

        # Threshold and calibration engine (for Eyes-Only and related options)
        self.threshold_engine = ThresholdCalibrationEngine(face_detector=self.face_detector)
        self.threshold_option = args.threshold

        # Handle calibration before starting processing, if requested
        if args.calibration_mode == 'manual' or args.manual_calibrate:
            print("\n" + "=" * 80)
            print("MANUAL CALIBRATION MODE")
            print("=" * 80)
            self.threshold_engine.calibrate_manual_thresholds()
        elif args.calibration_mode == 'auto' or (
            args.calibrate_active or args.calibrate_yawn or args.calibrate_closed
        ):
            print("\n" + "=" * 80)
            print("AUTO CALIBRATION MODE")
            print("=" * 80)
            if args.calibrate_active:
                self.threshold_engine.calibrate_baseline(args.calibrate_active, 'active')
            if args.calibrate_yawn:
                self.threshold_engine.calibrate_baseline(args.calibrate_yawn, 'yawning')
            if args.calibrate_closed:
                self.threshold_engine.calibrate_baseline(args.calibrate_closed, 'closed_eyes')

        if self.threshold_engine.calibration_data['calibrated']:
            print(f"\n[STATUS] {self.threshold_engine.get_calibration_status()}")
        elif args.calibration_mode == 'none' and self.threshold_option in ['all', '3']:
            print("\n[STATUS] NOT CALIBRATED - Using generic thresholds (Methods 1 & 2)")
        
        # Initialize video capture
        self.cap = None
        self.fps = args.fps
        self.frame_width = 640
        self.frame_height = 480
        self.out = None
        
        if args.camera is not None:
            print(f"Opening thermal camera {args.camera}...")
            self.cap = cv2.VideoCapture(args.camera)
        elif args.video:
            print(f"Opening thermal video: {args.video}")
            self.cap = cv2.VideoCapture(args.video)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or args.fps
        elif args.image:
            print(f"Loading thermal image: {args.image}")
            self.process_single_image(args.image)
            return
        
        if self.cap and self.cap.isOpened():
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if args.record:
                self._init_video_writer()
        else:
            raise RuntimeError("Failed to open thermal camera/video source")
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.skip_counter = 0
    
    def _init_video_writer(self):
        """Initialize video writer for output"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = self.output_dir / f'thermal_detection_{timestamp}.mp4'
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.out.isOpened():
            print(f"Warning: Failed to open video writer. Video recording disabled.")
            self.out = None
        else:
            print(f"Recording video to: {output_path}")
    
    def _convert_to_grayscale(self, frame):
        """
        Convert frame to grayscale if needed
        
        Args:
            frame: Input frame (can be BGR, grayscale, or 16-bit)
            
        Returns:
            Grayscale frame (8-bit or 16-bit)
        """
        if frame is None:
            return None
        
        # If already grayscale, return as is
        if len(frame.shape) == 2:
            return frame
        
        # If BGR/RGB, convert to grayscale
        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        
        return frame
    
    def process_single_image(self, image_path):
        """Process a single thermal image"""
        print(f"Processing single image: {image_path}")
        
        # Try to load as grayscale first
        frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if frame is None:
            # Try loading as color then convert
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not load image {image_path}")
                return
            frame = self._convert_to_grayscale(frame)
        
        result = self.process_frame(frame)
        
        # Display or save result
        if self.args.show:
            self._display_result(frame, result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            output_path = self.output_dir / f'result_{Path(image_path).stem}.png'
            annotated_frame = self._annotate_frame(frame.copy(), result)
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"Result saved to: {output_path}")
            
            # Print analysis
            self._print_analysis(result)
    
    def process_frame(self, frame):
        """
        Process single thermal frame
        
        Args:
            frame: Thermal frame (grayscale or 16-bit)
            
        Returns:
            dict with analysis results
        """
        # Ensure frame is grayscale
        gray_frame = self._convert_to_grayscale(frame)
        
        # Detect face
        detection_result = self.face_detector.detect_face(gray_frame)
        
        if not detection_result['detected']:
            return {'detected': False, 'frame': gray_frame}
        
        # Extract face region
        face_roi = detection_result['face_roi']
        landmarks = detection_result['landmarks']
        
        # Analyze eye state
        eye_result = self.eye_monitor.analyze_eye_state(gray_frame, landmarks)
        
        # Analyze yawn state - pass eye closure score from thermal strategy
        eye_closure_score = eye_result.get('combined_score', 0.0)
        yawn_result = self.yawn_detector.analyze_yawn_state(gray_frame, landmarks, eye_closure_score)
        
        # Perform drowsiness analysis
        drowsiness_result = self.drowsiness_analyzer.analyze_drowsiness(
            eye_closure_score=eye_result.get('combined_score', 0.0),
            yawn_score=yawn_result.get('combined_score', 0.0),
            ear_score=eye_result.get('ear', None),
            perclos=eye_result.get('perclos', None),
            blink_rate=eye_result.get('blink_rate', None),
            yawn_count=yawn_result.get('yawn_count', None),
            yawn_frequency=yawn_result.get('yawn_rate', None)
        )
        
        # Optional threshold-based analysis (Eyes-Only, weighted, calibration-based)
        threshold_results = []
        threshold_overall = None
        yawn_score = yawn_result.get('combined_score', 0.0)
        
        if self.threshold_option and self.threshold_option != 'none':
            if self.threshold_option in ['all', '1']:
                res1 = self.threshold_engine.analyze_eyes_only_threshold(eye_closure_score)
                threshold_results.append(res1)
            if self.threshold_option in ['all', '2']:
                res2 = self.threshold_engine.analyze_weighted_threshold(eye_closure_score, yawn_score)
                threshold_results.append(res2)
            if self.threshold_option in ['all', '3']:
                res3 = self.threshold_engine.analyze_calibration_threshold(eye_closure_score, yawn_score)
                threshold_results.append(res3)
        
            valid_results = [r for r in threshold_results if 'error' not in r]
            if valid_results:
                threshold_overall = any(r.get('drowsy', False) for r in valid_results)
        
        return {
            'detected': True,
            'frame': gray_frame,
            'detection': detection_result,
            'eye_analysis': eye_result,
            'yawn_analysis': yawn_result,
            'drowsiness_analysis': drowsiness_result,
            'threshold_analysis': threshold_results,
            'threshold_overall': threshold_overall
        }
    
    def run(self):
        """Main processing loop"""
        print("\nStarting thermal drowsiness detection...")
        print(f"Eye Strategy: {self.args.eye_strategy}")
        print(f"Yawn Strategy: {self.args.yawn_strategy}")
        if self.threshold_option and self.threshold_option != 'none':
            print(f"Threshold Mode: {self.threshold_option}")
            if self.threshold_engine.calibration_data['calibrated']:
                print(f"Calibration: {self.threshold_engine.get_calibration_status()}")
            else:
                print("Calibration: NOT CALIBRATED - Using generic thresholds")
        print(f"Press 'q' to quit, 's' to save analysis\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("\nEnd of video stream")
                    break
                
                # Handle frame skipping
                self.skip_counter += 1
                if self.skip_counter <= self.args.skip_frames:
                    continue
                self.skip_counter = 0
                
                self.frame_count += 1
                
                # Convert to grayscale for processing
                gray_frame = self._convert_to_grayscale(frame)
                
                # Process frame
                result = self.process_frame(gray_frame)
                
                if result['detected']:
                    if self.args.verbose:
                        self._print_analysis(result)
                    
                    # Record output (use original color frame for better visualization)
                    if self.out:
                        # Annotate the color frame if available, otherwise grayscale
                        display_frame = frame if len(frame.shape) == 3 else gray_frame
                        annotated_frame = self._annotate_frame(display_frame.copy(), result)
                        self.out.write(annotated_frame)
                    
                    # Display
                    if self.args.show:
                        # Use color frame for display if available
                        display_frame = frame if len(frame.shape) == 3 else gray_frame
                        self._display_result(display_frame, result)
                    
                    # Handle alerts
                    self._handle_alerts(result)
                
                # FPS calculation
                if self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed
                    print(f"Processed {self.frame_count} frames @ {fps:.1f} FPS", end='\r')
                
                # Keyboard handling
                if self.args.show:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        self._save_analysis()
        
        finally:
            self._cleanup()
    
    def _annotate_frame(self, frame, result):
        """Add annotations to frame"""
        if not result['detected']:
            # Ensure frame is in color for annotation
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return frame
        
        # Convert to BGR for display (if grayscale)
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw face ROI
        detection = result['detection']
        if detection.get('face_roi'):
            x, y, w, h = detection['face_roi']
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Get drowsiness analysis
        drowsiness = result['drowsiness_analysis']
        level = drowsiness['drowsiness_level']
        level_name = drowsiness['level_name']
        score = drowsiness['drowsiness_score']
        
        # Color based on level
        colors = {
            0: (0, 255, 0),      # ALERT - Green
            1: (0, 255, 255),    # PRE_ALERT - Yellow
            2: (0, 165, 255),    # SOFT - Orange
            3: (0, 0, 255),      # MEDIUM - Red
            4: (0, 0, 128)       # CRITICAL - Dark red
        }
        color = colors.get(level, (255, 255, 255))
        
        # Draw status
        cv2.putText(frame, f"Status: {level_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Score: {score:.2f}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw metrics
        eye_score = result['eye_analysis'].get('combined_score', 0)
        yawn_score = result['yawn_analysis'].get('combined_score', 0)
        perclos = result['eye_analysis'].get('perclos', 0)
        
        cv2.putText(frame, f"Eye: {eye_score:.2f} | Yawn: {yawn_score:.2f} | PERCLOS: {perclos:.2f}", 
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Optional threshold overlay
        threshold_overall = result.get('threshold_overall', None)
        if threshold_overall is not None and self.threshold_option and self.threshold_option != 'none':
            thresh_text = f"Thresh({self.threshold_option}): {'DROWSY' if threshold_overall else 'ALERT'}"
            cv2.putText(frame, thresh_text, (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Draw recommendations
        recommendations = drowsiness.get('recommendations', [])
        if recommendations:
            y_offset = 190 if threshold_overall is not None and self.threshold_option and self.threshold_option != 'none' else 150
            cv2.putText(frame, recommendations[0][:50], (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _display_result(self, frame, result):
        """Display frame with annotations"""
        annotated = self._annotate_frame(frame.copy(), result)
        
        # Convert to BGR if needed
        if len(annotated.shape) == 2:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_GRAY2BGR)
        
        cv2.imshow('Thermal Drowsiness Detection', annotated)
    
    def _handle_alerts(self, result):
        """Handle alert generation"""
        drowsiness_level = result['drowsiness_analysis']['drowsiness_level']
        
        if drowsiness_level >= 3:  # MEDIUM or CRITICAL
            recommendations = result['drowsiness_analysis']['recommendations']
            message = recommendations[0] if recommendations else "Drowsiness detected"
            self.alert_system.trigger_alert(
                level=drowsiness_level,
                message=message
            )
    
    def _print_analysis(self, result):
        """Print detailed analysis"""
        if not result['detected']:
            return
        
        drowsiness = result['drowsiness_analysis']
        print(f"\n{'='*60}")
        print(f"Frame {self.frame_count}")
        print(f"{'='*60}")
        print(f"Status: {drowsiness['level_name']}")
        print(f"Drowsiness Score: {drowsiness['drowsiness_score']:.3f}")
        print(f"\nComponent Scores:")
        for component, score in drowsiness['component_scores'].items():
            print(f"  {component}: {score:.3f}")
        print(f"\nContributing Factors:")
        for factor in drowsiness['contributing_factors']:
            print(f"  - {factor}")
        print(f"\nRecommendations:")
        for rec in drowsiness['recommendations'][:2]:
            print(f"  - {rec}")
        
        # Optional threshold-based summary
        threshold_results = result.get('threshold_analysis', [])
        if threshold_results:
            print(f"\nThreshold Analysis ({self.threshold_option}):")
            for res in threshold_results:
                status = 'DROWSY' if res.get('drowsy') else 'ALERT'
                method = res.get('method', 'Unknown')
                if 'error' in res:
                    print(f"  - {method}: ERROR - {res['error']}")
                else:
                    score_val = res.get('combined_score', res.get('score', 0.0))
                    print(f"  - {method}: {status} (score={score_val:.3f})")
    
    def _save_analysis(self):
        """Save current analysis"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'thermal_report_{timestamp}.txt'
        
        summary = self.drowsiness_analyzer.get_session_summary()
        
        with open(report_path, 'w') as f:
            f.write("THERMAL DROWSINESS DETECTION - SESSION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Session Duration: {summary['session_duration']:.1f} seconds\n")
            f.write(f"Total Frames: {summary['total_frames_processed']}\n")
            f.write(f"Alert Percentage: {summary['alert_percentage']:.1f}%\n")
            f.write(f"Drowsy Percentage: {summary['drowsy_percentage']:.1f}%\n")
            f.write(f"Critical Percentage: {summary['critical_percentage']:.1f}%\n")
            f.write(f"Average Drowsiness Score: {summary['average_drowsiness_score']:.3f}\n")
            f.write(f"\nOverall Assessment: {summary['overall_assessment']}\n")
        
        print(f"\nAnalysis saved to: {report_path}")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing complete. {self.frame_count} frames processed.")
        
        # Generate final report
        if self.report_generator:
            self._save_analysis()


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    if not os.path.exists(args.config):
        print(f"Warning: Config file not found: {args.config}")
        config = ConfigLoader()  # Use defaults
    else:
        config = ConfigLoader(args.config)
    
    # Ensure at least one input source
    if args.camera is None and args.video is None and args.image is None:
        print("Error: Must specify --camera, --video, or --image")
        parser.print_help()
        return
    
    # Enable display if interactive
    if args.show:
        args.show = True
    
    # Create and run system
    try:
        system = ThermalDriverDrowsinessDetection(args, config)
        
        if args.image:
            # Already processed in __init__
            pass
        else:
            system.run()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()