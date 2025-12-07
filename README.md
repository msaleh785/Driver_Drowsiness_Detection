# Driver_Drowsiness_Detection
Driver Drowsiness Detection Project
# Driver Drowsiness Detection (RGB + Thermal)

Real‑time driver drowsiness detection system using RGB and thermal cameras.

The system monitors:
- **Eyes** – Eye Aspect Ratio (EAR), PERCLOS, blink rate, sleep detection  
- **Mouth** – Mouth Aspect Ratio (MAR), yawn duration, yawn frequency  
- **Head** – Head pose (pitch/yaw), nodding, improper head position  

It fuses these indicators into a **5‑level alert**:
0 = Alert, 1 = Pre‑alert, 2 = Soft drowsiness, 3 = Medium, 4 = Critical.

---

## Features

- **RGB pipeline**
  - MediaPipe Face Mesh (or BlazeFace + landmarks) for 468‑point landmarks  
  - EyeMonitor: EAR, PERCLOS, blink detection, sleep detection  
  - YawnDetector: MAR‑based yawn detection with duration + frequency  
  - HeadMonitor: pitch/yaw, nodding, head position  
  - DrowsinessAnalyzer: multi‑indicator fusion with weighted scores

- **Thermal pipeline**
  - Thermal face/eye/mouth detection  
  - Multiple strategies (threshold‑based, temporal, hybrid)  
  - ThermalDrowsinessAnalyzer with weighted fusion of eye, mouth, thermal trend

- **Alerts & reporting**
  - Audio and visual alerts (configurable)  
  - 5‑level alert display overlaid on video  
  - Per‑component scores (`E, Y, H, T, EARn, MARn`) for debugging  
  - JSON/CSV session reports with event timelines

---

## Project structure

```text
.
├── driver_drowsiness_detection.py       # Main RGB pipeline
├── thermal_driver_drowsiness_detection.py  # Main thermal pipeline
├── thermal_test_strategies.py           # Thermal evaluation and strategy tester
├── modules/
│   ├── eye_monitor.py                   # EAR, blinks, PERCLOS, sleep detection
│   ├── yawn_detector.py                 # MAR, yawn duration & frequency
│   ├── head_monitor.py                  # Head pose & nodding
│   ├── drowsiness_analyzer.py           # RGB drowsiness fusion (5 levels)
│   ├── thermal_eye_monitor.py           # Thermal eye monitoring
│   ├── thermal_yawn_detector.py         # Thermal yawn/mouth monitoring
│   ├── thermal_drowsiness_analyzer.py   # Thermal drowsiness fusion
│   └── alert_system.py                  # Audio/visual alerts
├── models/                              # BlazeFace, dlib, temporal model, etc.
├── Testing Videos and Images/           # Example RGB/thermal videos & images
├── config.yaml                          # Main RGB configuration
├── config_thermal.yaml                  # Thermal configuration
├── README_THERMAL.md                    # Extra thermal documentation
└── reports/                             # Generated session reports









Configuration
Main configuration files:

config.yaml
 – RGB pipeline settings
indicator weights (eye_weight, yawn_weight, head_weight)
EAR/MAR thresholds, PERCLOS window, alert settings, report options
config_thermal.yaml
 – thermal strategies, thresholds, and camera options
You can adjust:

Which face detector to use (MediaPipe vs BlazeFace)
Whether to use temporal model (if TFLite model available)
Audio on/off, sound files, screenshot/report saving, etc.



Drowsiness levels
The system outputs 5 levels:

Level 0 – ALERT: normal state
Level 1 – PRE‑ALERT: early risk (slightly higher blinks/yawns)
Level 2 – SOFT: mild drowsiness
Level 3 – MEDIUM: strong drowsiness (long closures, many yawns)
Level 4 – CRITICAL: likely micro‑sleep / eyes closed for long time
