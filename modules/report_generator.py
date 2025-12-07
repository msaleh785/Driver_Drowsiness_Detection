#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report Generator Module

This module tracks drowsiness events during a session and generates
summary reports with statistics and event captures.
"""

import os
import time
import json
import csv
import cv2
import numpy as np
from datetime import datetime
from collections import deque

class ReportGenerator:
    def __init__(self, save_dir="reports"):
        """
        Initialize the report generator
        
        Args:
            save_dir: Directory to save reports and event captures
        """
        # Create directories if they don't exist
        self.save_dir = save_dir
        self.images_dir = os.path.join(save_dir, "images")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Session info
        self.session_start_time = time.time()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Drowsiness event tracking
        self.drowsiness_events = []
        self.current_event = None
        self.frame_buffer = deque(maxlen=10)  # Store last 10 frames for context
        
        # Statistics (updated for 5-level system: 0=Alert, 1=Pre-alert, 2=Soft, 3=Medium, 4=Critical)
        self.alert_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.total_frames = 0
        self.drowsy_frames = 0
        
        # Thresholds for new events
        self.NEW_EVENT_COOLDOWN = 10  # seconds between events
        self.last_event_time = 0
    
    def update(self, frame, drowsiness_level, detailed_status):
        """
        Update the report with current frame information
        
        Args:
            frame: Current video frame
            drowsiness_level: Current drowsiness level (0-4)
            detailed_status: Dict with detailed status information
        """
        try:
            # Add frame to buffer
            if frame is not None:
                self.frame_buffer.append(frame.copy())
        except Exception as e:
            # If frame copy fails, skip it
            pass
        
        # Update statistics
        self.total_frames += 1
        if drowsiness_level > 0:
            self.drowsy_frames += 1
            
        # Get current time - needed for all drowsiness levels
        current_time = time.time()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Track significant drowsiness events (level 2, 3, and 4)
        if drowsiness_level >= 2:  # Soft, Medium, or Critical drowsiness
            # Ensure drowsiness_level is within valid range
            drowsiness_level = min(4, max(0, drowsiness_level))
            self.alert_counts[drowsiness_level] += 1
            
            # Check if we should create a new event
            if (self.current_event is None and 
                current_time - self.last_event_time > self.NEW_EVENT_COOLDOWN):
                # Start new drowsiness event
                level_names = {2: "SOFT", 3: "MODERATE", 4: "CRITICAL"}
                level_name = level_names.get(drowsiness_level, "MODERATE")
                
                self.current_event = {
                    "start_time": current_time,
                    "start_time_str": timestamp_str,
                    "max_level": drowsiness_level,
                    "level_name": level_name,
                    "details": [detailed_status.get("description", "Unknown")],
                    "image_path": self._save_event_image(frame, drowsiness_level)
                }
                print(f"\n[{timestamp_str}] {self.current_event['level_name']} drowsiness event detected")
            elif self.current_event is not None:
                # Update existing event
                prev_level = self.current_event["max_level"]
                self.current_event["max_level"] = max(
                    prev_level, drowsiness_level
                )
                
                # Update level name if escalated
                level_names = {2: "SOFT", 3: "MODERATE", 4: "CRITICAL"}
                if drowsiness_level >= 3 and prev_level < 3:
                    self.current_event["level_name"] = level_names.get(drowsiness_level, "MODERATE")
                    print(f"\n[{timestamp_str}] Drowsiness escalated to {self.current_event['level_name']} level")
                elif drowsiness_level == 4 and prev_level < 4:
                    self.current_event["level_name"] = "CRITICAL"
                    print(f"\n[{timestamp_str}] Drowsiness escalated to CRITICAL level")
                
                # Add new details if they're different
                description = detailed_status.get("description", "Unknown")
                if description not in self.current_event["details"]:
                    self.current_event["details"].append(description)
        
        # If drowsiness event is over
        elif self.current_event is not None:
            # Add end time to event
            self.current_event["end_time"] = current_time
            self.current_event["duration"] = current_time - self.current_event["start_time"]
            
            # Add to events list
            self.drowsiness_events.append(self.current_event)
            self.current_event = None
            self.last_event_time = current_time
    
    def _save_event_image(self, frame, level):
        """Save an image from the drowsiness event"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"event_{self.session_id}_{timestamp}_level{level}.jpg"
            filepath = os.path.join(self.images_dir, filename)
            
            # Add annotations
            annotated_frame = frame.copy()
            level_names = {
                0: "Alert",
                1: "Pre-Alert",
                2: "Soft Alert",
                3: "Moderate Drowsiness",
                4: "Critical Drowsiness"
            }
            level_name = level_names.get(level, f"Level {level}")
            cv2.putText(
                annotated_frame, 
                f"DROWSINESS EVENT: {level_name}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
            
            # Save image
            cv2.imwrite(filepath, annotated_frame)
            return filepath
        except Exception as e:
            print(f"Warning: Could not save event image: {e}")
            return ""
    
    def generate_report(self):
        """
        Generate a summary report at the end of the session
        
        Returns:
            str: Path to the generated report
        """
        # If there's an active event, close it
        if self.current_event is not None:
            self.current_event["end_time"] = time.time()
            self.current_event["duration"] = self.current_event["end_time"] - self.current_event["start_time"]
            
            # Ensure level_name is present
            if "level_name" not in self.current_event:
                level_names = {2: "SOFT", 3: "MODERATE", 4: "CRITICAL"}
                max_level = self.current_event.get("max_level", 3)
                self.current_event["level_name"] = level_names.get(max_level, "MODERATE")
                
            self.drowsiness_events.append(self.current_event)
        
        # Calculate session duration
        session_duration = time.time() - self.session_start_time
        
        # Prepare report data
        report_data = {
            "session_id": self.session_id,
            "start_time": datetime.fromtimestamp(self.session_start_time).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": session_duration,
            "total_frames": self.total_frames,
            "alert_statistics": {
                "alert_frames": self.total_frames - self.drowsy_frames,
                "drowsy_frames": self.drowsy_frames,
                "drowsy_percentage": (self.drowsy_frames / max(self.total_frames, 1)) * 100,
                "soft_alerts": self.alert_counts[2],
                "moderate_alerts": self.alert_counts[3],
                "critical_alerts": self.alert_counts[4],
                "total_significant_alerts": self.alert_counts[2] + self.alert_counts[3] + self.alert_counts[4]
            },
            "drowsiness_events": []
        }
        
        # Add events to report
        for event in self.drowsiness_events:
            # Ensure level_name is present
            if "level_name" not in event:
                level_names = {2: "SOFT", 3: "MODERATE", 4: "CRITICAL"}
                max_level = event.get("max_level", 3)
                event["level_name"] = level_names.get(max_level, "MODERATE")
            
            # Ensure all required fields are present
            event_data = {
                "start_time": datetime.fromtimestamp(event.get("start_time", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.fromtimestamp(event.get("end_time", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": event.get("duration", 0),
                "max_level": event.get("max_level", 0),
                "level_name": event.get("level_name", "UNKNOWN"),
                "details": event.get("details", []),
                "image_path": event.get("image_path", "")
            }
            report_data["drowsiness_events"].append(event_data)
        
        # Save report as JSON
        json_report_path = os.path.join(self.save_dir, f"report_{self.session_id}.json")
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        
        # Save CSV summary
        csv_report_path = os.path.join(self.save_dir, f"summary_{self.session_id}.csv")
        with open(csv_report_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Driver Drowsiness Detection Session Summary"])
            writer.writerow(["Session ID", self.session_id])
            writer.writerow(["Start Time", report_data["start_time"]])
            writer.writerow(["End Time", report_data["end_time"]])
            writer.writerow(["Duration (seconds)", f"{session_duration:.1f}"])
            writer.writerow([])
            writer.writerow(["Alert Statistics"])
            writer.writerow(["Total Frames Processed", report_data["total_frames"]])
            writer.writerow(["Drowsy Percentage", f"{report_data['alert_statistics']['drowsy_percentage']:.2f}%"])
            writer.writerow(["Soft Alert Events", report_data["alert_statistics"]["soft_alerts"]])
            writer.writerow(["Moderate Drowsiness Events", report_data["alert_statistics"]["moderate_alerts"]])
            writer.writerow(["Critical Drowsiness Events", report_data["alert_statistics"]["critical_alerts"]])
            writer.writerow(["Total Significant Events", report_data["alert_statistics"]["total_significant_alerts"]])
            writer.writerow([])
            writer.writerow([])
            writer.writerow(["Significant Drowsiness Events (Levels 2-3 Only)"])
            writer.writerow(["Start Time", "End Time", "Duration (s)", "Level", "Details", "Screenshot"])
            
            for event in report_data["drowsiness_events"]:
                writer.writerow([
                    event["start_time"],
                    event["end_time"],
                    f"{event['duration_seconds']:.2f}",
                    event["level_name"],  # Use descriptive level name instead of number
                    "; ".join(event["details"]),
                    os.path.basename(event["image_path"])
                ])
            
            # Print summary to console
            print(f"\n==== Session Summary ====")
            print(f"Duration: {session_duration:.1f} seconds")
            print(f"Soft Alert Events: {report_data['alert_statistics']['soft_alerts']}")
            print(f"Moderate Drowsiness Events: {report_data['alert_statistics']['moderate_alerts']}")
            print(f"Critical Drowsiness Events: {report_data['alert_statistics']['critical_alerts']}")
            print(f"Total Significant Events: {report_data['alert_statistics']['total_significant_alerts']}")
        
        return json_report_path