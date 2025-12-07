#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Alert System Module

This module handles alerts and warnings based on detected drowsiness levels.
It can provide visual and audio alerts of varying intensity.
"""

import time
import os
# COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
# import pygame
import threading

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO = None
    GPIO_AVAILABLE = False

if GPIO_AVAILABLE:
    # Use Broadcom pin-numbering
    GPIO.setmode(GPIO.BCM)
    BUZZER_PIN = 2
    GPIO.setup(BUZZER_PIN, GPIO.OUT)
    GPIO.output(BUZZER_PIN, GPIO.LOW)
else:
    BUZZER_PIN = None

class AlertSystem:
    # Alert sound files (mapped to drowsiness levels)
    # Level 0 = Alert (no sound)
    # Level 1 = Pre-alert (slight_alert.wav)
    # Level 2 = Soft (slight_alert.wav)
    # Level 3 = Medium (moderate_alert.wav)
    # Level 4 = Critical (severe_alert.wav)
    ALERT_SOUNDS = {
        1: "assets/slight_alert.wav",   # Pre-alert
        2: "assets/slight_alert.wav",   # Soft
        3: "assets/moderate_alert.wav", # Medium
        4: "assets/severe_alert.wav"    # Critical
    }
    
    # Alert states
    STATE_NORMAL = 0   # No alert
    STATE_ALERTING = 1 # Alert in progress
    STATE_ALARMING = 2 # Continuous alarm until reset
    
    def __init__(self, enable_audio=True):
        """
        Initialize the alert system
        
        Args:
            enable_audio: Whether to enable audio alerts
        """
        self.enable_audio = enable_audio
        self.current_level = 0
        self.last_alert_time = {1: 0, 2: 0, 3: 0, 4: 0}
        self.alert_cooldown = {1: 10, 2: 5, 3: 3, 4: 0}  # Cooldown periods in seconds (level 4 has no cooldown)
        self.consecutive_alerts = 0
        self.alarm_state = self.STATE_NORMAL
        self.severe_alert_start_time = 0
        self.warning_shown = set()  # Track which warnings we've already shown
        
        # Initialize pygame for audio
        # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
        # if self.enable_audio:
        #     pygame.mixer.init()
        #     
        #     # Ensure assets directory exists
        #     os.makedirs("assets", exist_ok=True)
        #     
        #     # Create default alert sounds if they don't exist
        #     self._create_default_alert_sounds()
    
    def _create_default_alert_sounds(self):
        """Create default alert sound files if they don't exist"""
        # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
        # # This is a placeholder - in a real app, you would include actual sound files
        # # For now, we'll just check if the files exist
        # for level, filename in self.ALERT_SOUNDS.items():
        #     if not os.path.exists(filename):
        #         if level not in self.warning_shown:
        #             print(f"Warning: Alert sound file {filename} not found (will use system beep for level {level})")
        #             self.warning_shown.add(level)
        pass  # Sound disabled for PC - uncomment above when moving to RPi4
    
    def _should_trigger_alert(self, level):
        """Check if an alert should be triggered based on cooldown"""
        current_time = time.time()
        
        # Check cooldown for this level
        if current_time - self.last_alert_time[level] < self.alert_cooldown[level]:
            return False
        
        # Update last alert time
        self.last_alert_time[level] = current_time
        return True
    
    def _play_alert_sound(self, level):
        """Play the alert sound for the specified level"""
        # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
        # if not self.enable_audio:
        #     return
        #     
        # sound_file = self.ALERT_SOUNDS.get(level)
        # if not sound_file or not os.path.exists(sound_file):
        #     # Only show warning once per level
        #     if level not in self.warning_shown:
        #         print(f"Warning: Alert sound file for level {level} not available (using system beep)")
        #         self.warning_shown.add(level)
        #     # Use system beep as fallback
        #     try:
        #         import sys
        #         if sys.platform == 'win32':
        #             import winsound
        #             winsound.Beep(800 if level == 1 else (1000 if level == 2 else 1200), 200)
        #         else:
        #             print('\a', end='', flush=True)  # ASCII bell
        #     except:
        #         pass
        #     return
        #     
        # # Play sound in a separate thread to avoid blocking
        # def play_sound():
        #     try:
        #         sound = pygame.mixer.Sound(sound_file)
        #         sound.play()
        #     except Exception as e:
        #         # If sound file is corrupted, use system beep
        #         if level not in self.warning_shown:
        #             print(f"Error playing alert sound: {e} (using system beep instead)")
        #             self.warning_shown.add(level)
        #         try:
        #             import sys
        #             if sys.platform == 'win32':
        #                 import winsound
        #                 winsound.Beep(800 if level == 1 else (1000 if level == 2 else 1200), 200)
        #             else:
        #                 print('\a', end='', flush=True)
        #         except:
        #             pass
        #             
        # threading.Thread(target=play_sound).start()
        pass  # Sound disabled for PC - uncomment above when moving to RPi4
    
    def update(self, drowsiness_level):
        """
        Update the alert system with the current drowsiness level
        
        Args:
            drowsiness_level: Current drowsiness level (0-3)
        """
        # If we were alarming and have now dropped below level 4 (critical), turn buzzer off
        if GPIO_AVAILABLE and self.alarm_state == self.STATE_ALARMING and drowsiness_level != 4:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            self.alarm_state = self.STATE_NORMAL

        # Check if level has changed
        level_changed = drowsiness_level != self.current_level
        self.current_level = drowsiness_level
        
        # Critical drowsiness (level 4): latch buzzer on continuously
        if drowsiness_level == 4:
            if self.alarm_state != self.STATE_ALARMING:
                # Enter continuous alarm state
                self.alarm_state = self.STATE_ALARMING
                self.severe_alert_start_time = time.time()
                print("CRITICAL DROWSINESS DETECTED - CONTINUOUS ALARM ACTIVATED")
                print("Press 'r' to reset the alarm")
                if GPIO_AVAILABLE:
                    GPIO.output(BUZZER_PIN, GPIO.HIGH)    # ← turn buzzer on and leave it on
                # start playing the severe sound in loop
                # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
                # self._play_alert_sound(4)
            # (optional) replay sound every few seconds
            # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
            # elif time.time() - self.last_alert_time[4] > 3:
            #     self.last_alert_time[4] = time.time()
            #     self._play_alert_sound(4)

        elif drowsiness_level == 0:
            # If level 0 and not in latch alarm, reset counters
            if self.alarm_state != self.STATE_ALARMING:
                self.consecutive_alerts = 0

        else:  # moderate or slight alert (1 or 2)
            if self.alarm_state != self.STATE_ALARMING and self._should_trigger_alert(drowsiness_level):
                self.alarm_state = self.STATE_ALERTING
                # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
                # self._play_alert_sound(drowsiness_level)
                self.consecutive_alerts += 1

    def reset_alarm(self):
        """Reset the alarm state and turn the buzzer off"""
        # Turn buzzer off immediately
        if GPIO_AVAILABLE:
            GPIO.output(BUZZER_PIN, GPIO.LOW)
        self.alarm_state = self.STATE_NORMAL
        self.consecutive_alerts = 0
        self.current_level = 0
        print("Alert system reset")

    
    def cleanup(self):
        """Clean up resources"""
        # COMMENTED OUT FOR PC - UNCOMMENT WHEN MOVING TO RPI4
        # if self.enable_audio:
        #     pygame.mixer.quit()
        pass  # Sound disabled for PC - uncomment above when moving to RPi4

