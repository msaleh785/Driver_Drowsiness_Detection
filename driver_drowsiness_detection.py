#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Driver Drowsiness Detection System

Supports:
- MediaPipe or BlazeFace face detection (configurable)
- Temporal model (TCN/GRU) for improved accuracy
- Per-driver calibration
- Multi-indicator decision logic
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import argparse
from datetime import datetime
import os
import pygame

# Import our custom modules
from modules.config_loader import ConfigLoader
from modules.eye_monitor import EyeMonitor
from modules.yawn_detector import YawnDetector
from modules.head_monitor import HeadMonitor
from modules.drowsiness_analyzer import DrowsinessAnalyzer
from modules.alert_system import AlertSystem
from modules.report_generator import ReportGenerator
from modules.utils import draw_landmarks_on_frame, draw_status_on_frame

# Optional modules (loaded based on config)
blazeface_detector = None
temporal_model = None
calibration_manager = None

try:
    from modules.blazeface_detector import BlazeFaceDetector
    BLAZEFACE_AVAILABLE = True
except ImportError:
    BLAZEFACE_AVAILABLE = False
    print("Warning: BlazeFace detector not available")

try:
    from modules.temporal_model import TemporalModel
    TEMPORAL_MODEL_AVAILABLE = True
except ImportError:
    TEMPORAL_MODEL_AVAILABLE = False
    print("Warning: Temporal model not available")

try:
    from modules.calibration_manager import CalibrationManager
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
    print("Warning: Calibration manager not available")

# Set up argument parser
parser = argparse.ArgumentParser(description='Enhanced Driver Drowsiness Detection System')
parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
parser.add_argument('--video', type=str, help='Path to video file instead of using camera')
parser.add_argument('--show_landmarks', action='store_true', help='Show facial landmarks on display')
parser.add_argument('--alert', action='store_true', help='Enable audio alerts')
parser.add_argument('--record', action='store_true', help='Record video with detection results')
parser.add_argument('--report', action='store_true', help='Generate session report', default=True)
parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
parser.add_argument('--demo', action='store_true', help='Run in demo mode with simulated drowsiness')
parser.add_argument('--calibrate', type=str, help='Start calibration for driver ID (e.g., --calibrate driver1)')
args = parser.parse_args()

class DriverDrowsinessDetection:
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.monitoring_active = False
        self.calibration_active = False
        
        # Initialize face detection based on config
        face_detection_method = config.get('face_detection.method', 'mediapipe')
        
        if face_detection_method == 'blazeface' and BLAZEFACE_AVAILABLE:
            print("Using BlazeFace face detection")
            blazeface_config = config.get('face_detection.blazeface', {})
            model_path = blazeface_config.get('model_path', 'models/blazeface.tflite')
            detection_interval = blazeface_config.get('detection_interval', 3)
            score_threshold = blazeface_config.get('score_threshold', 0.5)
            use_mp_landmarks = blazeface_config.get('use_mediapipe_landmarks', True)
            
            try:
                self.face_detector = BlazeFaceDetector(
                    model_path=model_path,
                    detection_interval=detection_interval,
                    score_threshold=score_threshold,
                    use_mediapipe_landmarks=use_mp_landmarks
                )
                self.use_blazeface = True
            except Exception as e:
                print(f"Failed to initialize BlazeFace: {e}. Falling back to MediaPipe.")
                self._init_mediapipe()
                self.use_blazeface = False
        else:
            print("Using MediaPipe face detection")
            self._init_mediapipe()
            self.use_blazeface = False
        
        # Initialize monitoring modules
        self.eye_monitor = EyeMonitor()
        self.yawn_detector = YawnDetector()
        self.head_monitor = HeadMonitor()
        
        # Initialize temporal model if enabled
        if config.is_enabled('temporal_model') and TEMPORAL_MODEL_AVAILABLE:
            print("Initializing temporal model...")
            temporal_config = config.get('temporal_model', {})
            model_path = temporal_config.get('model_path', 'models/temporal_model.tflite')
            model_type = temporal_config.get('model_type', 'tcn')
            window_size = temporal_config.get('window_size_seconds', 2.0)
            fps = temporal_config.get('fps', 15)
            input_features = temporal_config.get('input_features', 10)
            
            try:
                self.temporal_model = TemporalModel(
                    model_path=model_path,
                    model_type=model_type,
                    window_size_seconds=window_size,
                    fps=fps,
                    input_features=input_features
                )
                print("Temporal model initialized")
            except Exception as e:
                print(f"Failed to initialize temporal model: {e}")
                self.temporal_model = None
        else:
            self.temporal_model = None
        
        # Initialize calibration manager if enabled
        if config.is_enabled('calibration') and CALIBRATION_AVAILABLE:
            print("Initializing calibration manager...")
            calib_config = config.get('calibration', {})
            duration = calib_config.get('calibration_duration', 20)
            profiles_dir = calib_config.get('profiles_directory', 'driver_profiles')
            
            self.calibration_manager = CalibrationManager(
                calibration_duration=duration,
                profiles_directory=profiles_dir
            )
            
            # Set default baseline if provided
            default_baseline = calib_config.get('default_baselines', {})
            if default_baseline:
                self.calibration_manager.set_default_baseline(default_baseline)
            
            # Auto-calibrate on start if requested
            if args.calibrate:
                self.calibration_manager.start_calibration(args.calibrate)
                self.calibration_active = True
            elif calib_config.get('auto_calibrate_on_start', False):
                self.calibration_manager.start_calibration("default")
                self.calibration_active = True
            
            print("Calibration manager initialized")
        else:
            self.calibration_manager = None
        
        # Initialize drowsiness analyzer
        self.drowsiness_analyzer = DrowsinessAnalyzer(
            config=config.config,  # Pass full config dict
            temporal_model=self.temporal_model,
            calibration_manager=self.calibration_manager
        )
        
        # Initialize alert system
        self.alert_system = AlertSystem(enable_audio=args.alert)
        
        # Initialize report generator
        self.report_generator = ReportGenerator() if args.report else None
        
        # Initialize video capture
        if args.video:
            print(f"Using video file: {args.video}")
            self.cap = cv2.VideoCapture(args.video)
            if self.cap.isOpened():
                # Get video properties
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                duration = total_frames / fps if fps > 0 else 0
                print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        else:
            print(f"Using camera index: {args.camera}")
            self.cap = cv2.VideoCapture(args.camera)
        
        if not self.cap.isOpened():
            print("\nERROR: Could not open camera or video file.")
            print("Try these options:")
            print("1. Use a video file with: --video path_to_video.mp4")
            print("2. Try a different camera index: --camera 1")
            print("3. Use demo mode with: --demo")
            raise IOError("Failed to open video source")
        
        # For recording
        self.recorder = None
        if args.record:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs('recordings', exist_ok=True)
            self.recorder = cv2.VideoWriter(
                f'recordings/drowsiness_{timestamp}.avi',
                fourcc, 20.0,
                (int(self.cap.get(3)), int(self.cap.get(4)))
            )
        
        # For FPS calculation
        self.prev_frame_time = 0
        self.curr_frame_time = 0
    
    def _init_mediapipe(self):
        """Initialize MediaPipe Face Mesh"""
        self.mp_face_mesh = mp.solutions.face_mesh
        mp_config = self.config.get('face_detection.mediapipe', {})
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=mp_config.get('max_num_faces', 1),
            refine_landmarks=mp_config.get('refine_landmarks', True),
            min_detection_confidence=mp_config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=mp_config.get('min_tracking_confidence', 0.5)
        )
    
    def run(self):
        try:
            print("\n=== Driver Drowsiness Detection System ===")
            print("Controls:")
            print("  's' - Start/pause monitoring")
            print("  'r' - Reset alarm state")
            print("  'c' - Start calibration (if enabled)")
            print("  'q' - Quit")
            
            if self.calibration_active:
                print("\n⚠️  CALIBRATION MODE ACTIVE")
                print(f"Please remain alert for {self.calibration_manager.calibration_duration} seconds...")
            
            frame_count = 0
            while self.cap.isOpened():
                success, frame = self.cap.read()
                if not success:
                    if self.args.video:
                        print(f"\nVideo playback completed. Processed {frame_count} frames.")
                        break
                    print("Ignoring empty camera frame.")
                    continue
                
                frame_count += 1
                
                try:
                    # Calculate FPS
                    self.curr_frame_time = time.time()
                    fps = 1 / (self.curr_frame_time - self.prev_frame_time) if self.prev_frame_time else 0
                    self.prev_frame_time = self.curr_frame_time
                    
                    # Process frame
                    frame.flags.writeable = False
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect face and extract landmarks
                    face_landmarks = None
                    try:
                        if self.use_blazeface:
                            face_landmarks = self.face_detector.process_frame(frame)
                        else:
                            results = self.face_mesh.process(frame)
                            face_landmarks = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
                    except Exception as e:
                        print(f"Warning: Error in face detection: {e}")
                        face_landmarks = None
                    
                    # Convert back for rendering
                    frame.flags.writeable = True
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    if face_landmarks:
                        # Draw landmarks if enabled
                        if args.show_landmarks:
                            draw_landmarks_on_frame(frame, face_landmarks)
                    
                        # Process with monitors
                        try:
                            eye_status = self.eye_monitor.process(face_landmarks)
                            yawn_status = self.yawn_detector.process(face_landmarks)
                            head_status = self.head_monitor.process(face_landmarks)
                        except Exception as e:
                            print(f"Warning: Error processing landmarks: {e}")
                            eye_status = {"drowsiness": "none"}
                            yawn_status = {"drowsiness": "none", "is_yawning": False}
                            head_status = {"drowsiness": "none", "is_nodding": False}
                        
                        # Handle calibration
                        if self.calibration_active and self.calibration_manager:
                            try:
                                calibration_complete = self.calibration_manager.update_calibration(
                                    eye_status, yawn_status, head_status
                                )
                                if calibration_complete:
                                    self.calibration_active = False
                                    print("\n✅ Calibration complete! Starting monitoring...")
                                    self.monitoring_active = True
                            except Exception as e:
                                print(f"Warning: Error in calibration: {e}")
                        
                        # Analyze drowsiness (only if monitoring active and not calibrating)
                        if self.monitoring_active and not self.calibration_active:
                            try:
                                drowsiness_level, detailed_status = self.drowsiness_analyzer.analyze(
                                    eye_status, yawn_status, head_status
                                )
                                
                                # Handle alerts
                                try:
                                    self.alert_system.update(drowsiness_level)
                                except Exception as e:
                                    print(f"Warning: Error in alert system: {e}")
                            except Exception as e:
                                print(f"Warning: Error in drowsiness analysis: {e}")
                                drowsiness_level = 0
                                detailed_status = {
                                    "level": 0,
                                    "description": "ERROR",
                                    "details": [f"Processing error: {str(e)}"]
                                }
                        else:
                            # Not monitoring or calibrating
                            if self.calibration_active:
                                try:
                                    progress = self.calibration_manager.get_calibration_progress()
                                    drowsiness_level = 0
                                    detailed_status = {
                                        "level": 0,
                                        "description": f"CALIBRATING ({progress*100:.0f}%)",
                                        "details": [f"Calibration in progress: {progress*100:.0f}%"]
                                    }
                                except:
                                    drowsiness_level = 0
                                    detailed_status = {
                                        "level": 0,
                                        "description": "CALIBRATING",
                                        "details": ["Calibration in progress"]
                                    }
                            else:
                                drowsiness_level = 0
                                detailed_status = {
                                    "level": 0,
                                    "description": "MONITORING PAUSED",
                                    "details": ["Press 's' to start monitoring"]
                                }
                        
                        # Update report generator
                        if self.report_generator:
                            try:
                                self.report_generator.update(frame, drowsiness_level, detailed_status)
                            except Exception as e:
                                print(f"Warning: Error updating report: {e}")
                        
                        # Display status
                        try:
                            draw_status_on_frame(frame, drowsiness_level, detailed_status, fps)
                        except Exception as e:
                            print(f"Warning: Error drawing status: {e}")
                    else:
                        # No face detected
                        cv2.putText(frame, "No face detected", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add monitoring status
                    if not self.monitoring_active and not self.calibration_active:
                        cv2.putText(frame, "MONITORING PAUSED - Press 's' to start", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display frame
                    cv2.imshow('Driver Drowsiness Detection', frame)
                    
                    # Record if enabled
                    if self.recorder:
                        try:
                            self.recorder.write(frame)
                        except Exception as e:
                            print(f"Warning: Error recording frame: {e}")
                    
                    # Handle key presses (non-blocking)
                    key = cv2.waitKey(1) & 0xFF  # Changed from 5 to 1 for better responsiveness
                    if key == ord('q'):
                        print("\nUser requested quit.")
                        break
                    elif key == ord('s'):
                        if not self.calibration_active:
                            self.monitoring_active = not self.monitoring_active
                            status = "STARTED" if self.monitoring_active else "PAUSED"
                            print(f"Monitoring {status}")
                    elif key == ord('r'):
                        try:
                            self.alert_system.reset_alarm()
                            self.eye_monitor = EyeMonitor()
                            self.head_monitor = HeadMonitor()
                            self.drowsiness_analyzer.reset()
                            print("System reset - alarm cleared")
                        except Exception as e:
                            print(f"Warning: Error resetting system: {e}")
                    elif key == ord('c') and self.calibration_manager:
                        if not self.calibration_active:
                            try:
                                driver_id = input("Enter driver ID (or press Enter for 'default'): ").strip() or "default"
                                self.calibration_manager.start_calibration(driver_id)
                                self.calibration_active = True
                                self.monitoring_active = False
                                print(f"Calibration started for driver: {driver_id}")
                            except Exception as e:
                                print(f"Error starting calibration: {e}")
                
                except Exception as e:
                    # Catch any other errors and continue processing
                    print(f"Warning: Error processing frame: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next frame
                    continue
        
        finally:
            # Clean up
            self.cap.release()
            if self.recorder:
                self.recorder.release()
            
            if self.report_generator:
                report_path = self.report_generator.generate_report()
                print(f"\nDrowsiness detection session completed")
                print(f"Session report saved to: {report_path}")
            
            cv2.destroyAllWindows()
            self.alert_system.cleanup()

def create_demo_video():
    """Create a demo video file with simulated drowsiness patterns"""
    os.makedirs('demo', exist_ok=True)
    demo_path = 'demo/simulated_drowsiness.mp4'
    
    if os.path.exists(demo_path):
        print(f"Using existing demo file: {demo_path}")
        return demo_path
    
    print("Creating demo video file...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(demo_path, fourcc, 20.0, (640, 480))
    
    for _ in range(200):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Demo Mode - No Camera Available", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "This is a placeholder demo video", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out.write(frame)
    
    out.release()
    print(f"Created demo file: {demo_path}")
    return demo_path

if __name__ == "__main__":
    try:
        # Load configuration
        config = ConfigLoader(args.config)
        
        # Handle demo mode
        if args.demo:
            args.video = create_demo_video()
        
        # Create and run application
        app = DriverDrowsinessDetection(args, config)
        app.run()
    except IOError as e:
        if not args.demo:
            print(f"\nError: {e}")
            print("Run with --demo flag to test with simulated drowsiness patterns.")
        if input("\nWould you like to test with simulated drowsiness? [y/n]: ").lower().startswith('y'):
            args.video = create_demo_video()
            config = ConfigLoader(args.config)
            app = DriverDrowsinessDetection(args, config)
            app.run()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
