#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for the Driver Drowsiness Detection System
"""

import cv2
import numpy as np
import mediapipe as mp
import time

def draw_landmarks_on_frame(frame, face_landmarks):
    """
    Draw facial landmarks on the frame for visualization
    
    Args:
        frame: Video frame
        face_landmarks: MediaPipe face landmarks
    """
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh
    
    # Drawing specifications
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
    
    # Draw the face mesh
    mp_drawing.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=drawing_spec)
    
    # Highlight eye landmarks
    left_eye_indices = [362, 385, 387, 263, 373, 380]
    right_eye_indices = [33, 160, 158, 133, 153, 144]
    
    # Draw eye landmarks with a different color
    for idx in left_eye_indices + right_eye_indices:
        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    
    # Highlight mouth landmarks
    mouth_indices = [13, 14, 78, 80, 81, 82, 178, 87, 88, 91, 95, 185]
    
    # Draw mouth landmarks with a different color
    for idx in mouth_indices:
        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    # Highlight key head-pose landmarks (same indices used in HeadMonitor)
    head_indices = [1, 10, 152, 33, 263, 61, 291]  # nose, forehead, chin, eyes, mouth corners
    for idx in head_indices:
        x = int(face_landmarks.landmark[idx].x * frame.shape[1])
        y = int(face_landmarks.landmark[idx].y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

def draw_status_on_frame(frame, drowsiness_level, detailed_status, fps):
    """
    Draw drowsiness status information on the frame
    
    Args:
        frame: Video frame
        drowsiness_level: Current drowsiness level (0-3)
        detailed_status: Dict with detailed status information
        fps: Current frames per second
    """
    h, w = frame.shape[:2]
    
    # Draw background rectangle for status display
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
    
    # Display FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display drowsiness level (updated for 5-level system)
    drowsiness_level = min(4, max(0, drowsiness_level))  # Clamp to valid range
    level_color = get_level_color(drowsiness_level)
    level_descriptions = [
        "ALERT",
        "PRE-ALERT",
        "SOFT ALERT - Level 2",
        "MODERATE DROWSINESS - Level 3",
        "CRITICAL DROWSINESS - Level 4 ALARM"
    ]
    if drowsiness_level < len(level_descriptions):
        cv2.putText(frame, level_descriptions[drowsiness_level], (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
    else:
        cv2.putText(frame, f"DROWSINESS LEVEL {drowsiness_level}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, level_color, 2)
    
    # Display eye status if available (but not yawn info)
    details = detailed_status.get('details', [])
    if details and len(details) > 0:
        eye_info = [d for d in details if "Eyes" in d or "PERCLOS" in d]
        if eye_info:
            cv2.putText(frame, eye_info[0], (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        elif len(details) > 0:
            # Show first detail if available
            cv2.putText(frame, details[0][:50], (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show yawn status line from details if present
        yawn_info = [d for d in details if ("Yawning" in d) or ("yawn" in d.lower())]
        if yawn_info:
            cv2.putText(frame, yawn_info[0][:50], (10, 75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Additionally, show a simple yawning percentage based on yawn drowsiness level
    yawn_status = detailed_status.get('yawn_status', {})
    if yawn_status:
        yawn_level = yawn_status.get('drowsiness', 'none')
        if yawn_level == 'high':
            yawn_percent = 100
        elif yawn_level == 'medium':
            yawn_percent = 70
        elif yawn_level == 'low':
            yawn_percent = 40
        else:
            yawn_percent = 0

        if yawn_percent > 0:
            cv2.putText(frame, f"Yawn: {yawn_percent}%", (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show per-component scores (eye, yawn, head, total, ear_norm, mar_norm)
    scores = detailed_status.get('scores', {})
    if scores:
        eye_s = scores.get('eye', 0.0)
        yawn_s = scores.get('yawn', 0.0)
        head_s = scores.get('head', 0.0)
        total_s = scores.get('total', 0.0)
        ear_n = scores.get('ear_norm', 0.0)
        mar_n = scores.get('mar_norm', 0.0)
        text = f"E:{eye_s:.2f} Y:{yawn_s:.2f} H:{head_s:.2f} T:{total_s:.2f} EARn:{ear_n:.2f} MARn:{mar_n:.2f}"
        cv2.putText(frame, text, (10, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # If critical drowsiness (level 4), make it very noticeable
    if drowsiness_level >= 3:  # Level 3 or 4
        # Flash the screen with red border
        thickness = int(time.time() * 5) % 10 + 5  # Pulsating thickness
        color = (0, 0, 255) if drowsiness_level == 4 else (0, 100, 255)  # Red for critical, orange for moderate
        cv2.rectangle(frame, (0, 0), (w, h), color, thickness)

def get_level_color(drowsiness_level):
    """
    Get color based on drowsiness level (updated for 5-level system)
    
    Args:
        drowsiness_level: Current drowsiness level (0-4)
        
    Returns:
        tuple: BGR color value
    """
    drowsiness_level = min(4, max(0, drowsiness_level))  # Clamp to valid range
    if drowsiness_level == 0:
        return (0, 255, 0)    # Green - Alert
    elif drowsiness_level == 1:
        return (0, 255, 255)  # Yellow - Pre-alert
    elif drowsiness_level == 2:
        return (0, 200, 255)  # Light Orange - Soft alert
    elif drowsiness_level == 3:
        return (0, 165, 255)  # Orange - Moderate drowsiness
    else:  # level 4
        return (0, 0, 255)    # Red - Critical drowsiness