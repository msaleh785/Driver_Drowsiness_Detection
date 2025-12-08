# Driver Drowsiness Detection System (RGB + Thermal)

A real-time driver drowsiness detection system using computer vision and machine learning to monitor driver alertness through facial analysis. This project includes both **RGB-based detection** and **novel thermal imaging-based detection** approaches.

## 🎯 Project Overview

This system monitors drivers in real-time by analyzing:
- **Eye Closure** - Detects prolonged eye closure and calculates Eye Aspect Ratio (EAR)
- **Yawning** - Identifies yawning patterns through Mouth Aspect Ratio (MAR)
- **Head Position** - Tracks head movements, tilting, and nodding (optional)

The system provides multi-level alerts based on drowsiness severity and generates detailed session reports.

---

## 🌡️ **Novel Thermal Drowsiness Detection (NEW)**

### What Makes This Novel?

The **Thermal Drowsiness Detection** system represents a novel approach to drowsiness detection by leveraging thermal imaging instead of traditional RGB cameras. This innovation offers several advantages:

**Key Innovations:**
- **Privacy-Preserving:** Thermal imaging doesn't capture identifiable facial features, addressing privacy concerns
- **Low-Light Performance:** Works in complete darkness and varying lighting conditions
- **Temperature-Based Analysis:** Detects physiological changes associated with drowsiness (eye temperature variations, yawning heat signatures)
- **Multi-Metric Approach:** Combines multiple thermal-based metrics for robust detection

### Thermal System Components

The thermal detection system (`Thermal_Drowsiness/` directory) includes three specialized modules:

#### 1. **Eye Drowsiness Detection** (`eye_drowsiness_video.py`)

Analyzes thermal signatures of eyes using multiple metrics:

**Metrics Implemented:**
- **EAR (Eye Aspect Ratio):** Geometric eye opening measurement
- **EPR (Eye Perimeter Ratio):** Perimeter-to-width ratio
- **EAR (Eye Area Ratio):** Normalized polygon area
- **EIR (Eye Intensity Ratio):** Thermal intensity analysis
- **Combined Score:** Weighted combination with temporal smoothing

**Intensity Analysis Methods:**
- `region`: Average intensity over entire eye region
- `center`: Focused intensity at eye center (more sensitive)

**Usage:**
```bash
# Basic eye drowsiness detection
python Thermal_Drowsiness/eye_drowsiness_video.py --input thermal_video.mp4 --output eye_output.mp4

# With EAR threshold and metrics display
python Thermal_Drowsiness/eye_drowsiness_video.py \
    --input thermal_video.mp4 \
    --output eye_output.mp4 \
    --ear-thresh 0.20 \
    --show-metrics

# Using combined metrics with intensity analysis
python Thermal_Drowsiness/eye_drowsiness_video.py \
    --input thermal_video.mp4 \
    --output eye_output.mp4 \
    --combined-thresh 0.6 \
    --intensity center \
    --show-metrics
```

#### 2. **Mouth/Yawn Detection** (`mouth_drowsiness_video.py`)

Detects yawning through thermal mouth analysis:

**Metrics Implemented:**
- **MAR (Mouth Aspect Ratio):** Vertical-to-horizontal mouth opening ratio
- **MOR (Mouth Opening Ratio):** Height-to-width ratio
- **MAP (Mouth Area Percentage):** Normalized mouth area
- **MIT (Mouth Intensity Total):** Average thermal intensity
- **Combined Score:** Weighted multi-metric analysis

**Features:**
- 20-point outer lip contour tracking
- Temporal smoothing (10-frame window)
- Adjustable sensitivity thresholds

**Usage:**
```bash
# Basic mouth/yawn detection
python Thermal_Drowsiness/mouth_drowsiness_video.py \
    --input thermal_video.mp4 \
    --output mouth_output.mp4 \
    --mar-thresh 0.6

# With combined metrics
python Thermal_Drowsiness/mouth_drowsiness_video.py \
    --input thermal_video.mp4 \
    --output mouth_output.mp4 \
    --combined-thresh 0.45 \
    --show-metrics
```

#### 3. **Combined Eye-Mouth Analysis** (`eye_mouth_combined_video.py`)

Integrates both eye closure and yawning detection:

**Features:**
- Simultaneous eye and mouth monitoring
- Independent threshold configuration for each
- Real-time metric visualization
- Combined drowsiness scoring

**Usage:**
```bash
# Full thermal drowsiness detection
python Thermal_Drowsiness/eye_mouth_combined_video.py \
    --input thermal_video.mp4 \
    --output combined_output.mp4 \
    --eye-combined-thresh 0.6 \
    --eye-intensity center \
    --mouth-combined-thresh 0.45 \
    --show-metrics
```

### Thermal System Requirements

```bash
# Install thermal-specific dependencies
cd Thermal_Drowsiness
pip install -r requirements.txt

# Dependencies include:
# - opencv-python
# - mediapipe (for facial landmark detection)
# - numpy
# - scipy
# - imutils
```

### Thermal Detection Parameters

**Configurable Thresholds:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--ear-thresh` | 0.25 | Eye Aspect Ratio threshold for eye closure |
| `--eye-combined-thresh` | 0.60 | Combined eye metrics threshold |
| `--mar-thresh` | 0.60 | Mouth Aspect Ratio threshold for yawning |
| `--mouth-combined-thresh` | 0.45 | Combined mouth metrics threshold |
| `--intensity` | region | Intensity analysis method (region/center) |
| `--show-metrics` | False | Display real-time metrics overlay |

**Intensity Analysis:**
- `region`: Analyzes average intensity over entire facial region (robust)
- `center`: Focuses on central eye/mouth area (more sensitive to changes)

### Thermal vs RGB Comparison

| Feature | RGB Detection | Thermal Detection (Novel) |
|---------|--------------|---------------------------|
| Privacy | Captures identifiable features | Anonymous thermal signatures |
| Lighting | Requires adequate lighting | Works in complete darkness |
| Performance | Excellent in daylight | Consistent in all conditions |
| Physiological Data | Limited | Temperature variations detected |
| Cost | Standard cameras | Thermal camera required |
| Processing | Real-time capable | Real-time capable |

### Research Applications

The thermal drowsiness detection system enables research in:
- **Privacy-preserving monitoring systems**
- **Multi-modal drowsiness detection** (RGB + Thermal fusion)
- **Physiological drowsiness indicators** (temperature-based)
- **Low-light autonomous vehicle safety**
- **Healthcare monitoring** (non-invasive patient observation)

---

## 🏗️ System Architecture

### Core Modules (Required for RGB Detection)

The `driver_drowsiness_detection.py` uses the following essential modules:

#### **Required Modules:**
1. **`config_loader.py`** - Loads and manages YAML configuration
2. **`eye_monitor.py`** - Monitors eye closure and calculates EAR
3. **`yawn_detector.py`** - Detects yawning through MAR calculation
4. **`head_monitor.py`** - Tracks head position and movements
5. **`drowsiness_analyzer.py`** - Combines indicators into drowsiness levels
6. **`alert_system.py`** - Manages audio/visual alerts
7. **`report_generator.py`** - Creates session reports
8. **`utils.py`** - Display and rendering utilities

#### **Optional Modules:**
- **`blazeface_detector.py`** - Alternative face detection (optional)
- **`temporal_model.py`** - TCN/GRU temporal analysis (optional)
- **`calibration_manager.py`** - Per-driver calibration (optional)

## 📋 Prerequisites

### Hardware Requirements
- **Raspberry Pi 4** (2GB RAM minimum, 4GB+ recommended)
- **Raspberry Pi Camera Module** (v2 or compatible USB webcam)
- **MicroSD Card** (16GB minimum, Class 10)
- **Power Supply** (5V 3A official adapter)
- **Optional:** Speaker/buzzer for audio alerts

### Software Requirements
- **Raspberry Pi OS** (64-bit recommended)
- **Python 3.8+**
- **Camera enabled** in Raspberry Pi configuration

## 🚀 Installation Guide for Raspberry Pi

### Step 1: Prepare Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv git
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtgui4 libqt4-test
sudo apt install -y libportaudio2 portaudio19-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
```

### Step 2: Enable Camera

```bash
# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable

# Reboot
sudo reboot
```

### Step 3: Install Performance Optimization (Recommended)

```bash
# Increase swap size for MediaPipe compilation
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Change CONF_SWAPSIZE=100 to CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Step 4: Clone Repository

```bash
# Create project directory
mkdir -p ~/projects
cd ~/projects

# Clone or copy project files
git clone <your-repo-url> DDD
# OR copy files manually to ~/projects/DDD

cd DDD
```

### Step 5: Download Required Models

The repository includes empty `models/` folders. You need to download the required open-source models and place them in the specified directories.

#### RGB Detection Models (Place in `models/` folder)

**1. MediaPipe Face Mesh (Automatically downloaded)**
- MediaPipe will automatically download its models on first run
- No manual download required

**2. BlazeFace TFLite Model (Optional - for alternative face detection)**
- **Download:** [BlazeFace Short Range Model](https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite)
- **Rename to:** `blazeface.tflite`
- **Place in:** `models/blazeface.tflite`

**3. Dlib Face Detector (Optional)**
- **Download:** [Dlib SVM Model](http://dlib.net/files/mmod_human_face_detector.dat.bz2)
- **Extract and rename to:** `dlib_face_detector.svm`
- **Place in:** `models/dlib_face_detector.svm`

**4. U-Net Model (Optional - for temporal analysis)**
- **Download:** Pre-trained U-Net model from [Keras Applications](https://keras.io/api/applications/)
- **Alternative:** Train your own using `modules/train_temporal_model.py`
- **Place in:** `models/unet_model.h5`

#### Thermal Detection Models (Place in `Thermal_Drowsiness/models/` folder)

**MediaPipe Face Mesh for Thermal**
- MediaPipe will automatically download models on first run
- No manual download required
- Models work with grayscale thermal images

```bash
# Create models directories if they don't exist
mkdir -p models
mkdir -p Thermal_Drowsiness/models

# Download BlazeFace (Optional)
cd models
wget https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite -O blazeface.tflite

# Download Dlib (Optional)
wget http://dlib.net/files/mmod_human_face_detector.dat.bz2
bunzip2 mmod_human_face_detector.dat.bz2
mv mmod_human_face_detector.dat dlib_face_detector.svm

cd ..
```

**Note:** The core system uses MediaPipe Face Mesh which downloads automatically. BlazeFace and Dlib models are optional alternatives for face detection.

### Step 6: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv ddd_env

# Activate environment
source ddd_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 7: Install Dependencies

```bash
# Install from requirements file
pip install -r requirements.txt

# This will install:
# - opencv-python (4.8.1.78)
# - mediapipe (0.10.8)
# - numpy (1.24.3)
# - pygame (2.5.2)
# - PyYAML (6.0.1)
# - scipy (1.10.1)
# - psutil (5.9.6)
```

**Note:** Installation may take 15-30 minutes on Raspberry Pi due to compilation.

### Step 8: Verify Installation

```bash
# Test camera access
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"

# Test MediaPipe
python3 -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"

# Test camera capture
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera:', cap.isOpened()); cap.release()"
```

### Step 9: Configure System

```bash
# Copy and edit configuration (if needed)
cp config.yaml config_custom.yaml
nano config_custom.yaml

# Adjust parameters:
# - Detection thresholds
# - Alert settings
# - Weight distribution
```

---

## 🎮 Running the System

### Basic Usage

```bash
# Activate environment
source ddd_env/bin/activate

# Run with default camera (index 0)
python driver_drowsiness_detection.py --camera 0

# Run with landmarks visualization
python driver_drowsiness_detection.py --camera 0 --show_landmarks

# Run with audio alerts
python driver_drowsiness_detection.py --camera 0 --alert

# Run with head movement detection
python driver_drowsiness_detection.py --camera 0 --include_head
```

### Advanced Options

```bash
# Use video file for testing
python driver_drowsiness_detection.py --video path/to/video.mp4

# Enable recording
python driver_drowsiness_detection.py --camera 0 --record

# Custom configuration
python driver_drowsiness_detection.py --camera 0 --config config_custom.yaml

# Full featured run
python driver_drowsiness_detection.py --camera 0 --show_landmarks --alert --include_head --record
```

### Runtime Controls

While the system is running:
- **'s'** - Start/pause monitoring
- **'r'** - Reset alarm state
- **'c'** - Start calibration (if enabled)
- **'q'** - Quit application

## 📊 Understanding the Output

### Drowsiness Levels

| Level | Description | Action |
|-------|-------------|---------|
| 0 | Alert | No warnings |
| 1 | Pre-Alert | Monitoring closely |
| 2 | Soft Drowsiness | Gentle warning |
| 3 | Medium Drowsiness | Moderate alert |
| 4 | Critical Drowsiness | Urgent alert |

### Display Information

The system displays:
- **EAR (Eye Aspect Ratio)** - Left, Right, Average
- **MAR (Mouth Aspect Ratio)** - Current percentage
- **Yawn Count** - Frequency in last 60 seconds
- **Yawn Timer** - Duration of current yawn
- **Head Position** - Yaw, Pitch, Roll angles (if enabled)
- **Combined Score** - Overall drowsiness percentage
- **Weight Breakdown** - Component contributions (Eyes:60%, Yawn:25%, Head:15%)

### Reports

Session reports are saved in `reports/` directory with:
- Session duration and statistics
- Frame-by-frame analysis
- Alert history
- Timestamp: `report_YYYYMMDD_HHMMSS.json`

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Detection thresholds
eye_monitor:
  ear_threshold: 0.25
  eye_closure_threshold: 0.3

yawn_detector:
  mar_threshold: 0.6
  continuous_yawn_threshold: 5.0

head_monitor:
  deviation_threshold: 0.35  # ~20 degrees

# Weight distribution
drowsiness:
  weights:
    eye_closure: 0.60
    yawning: 0.25
    head_movement: 0.15
```

## 🐛 Troubleshooting

### Camera Issues

```bash
# Check camera connection
vcgencmd get_camera

# Expected output: supported=1 detected=1

# Test camera directly
raspistill -o test.jpg

# Check camera index
ls -l /dev/video*
```

### Performance Issues

```bash
# Reduce camera resolution in code
# Edit driver_drowsiness_detection.py
# Add after camera initialization:
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Monitor CPU usage
htop

# Check temperature
vcgencmd measure_temp
```

### Import Errors

```bash
# Ensure virtual environment is activated
source ddd_env/bin/activate

# Reinstall packages
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

### Memory Errors

```bash
# Increase swap size (see Step 3)
# Close other applications
# Use lighter MediaPipe model (edit config.yaml)
```

## 📁 Project Structure

```
DDD/
├── driver_drowsiness_detection.py    # Main entry point (RGB)
├── config.yaml                        # Configuration file
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── modules/                           # Core modules (RGB)
│   ├── config_loader.py              # Configuration management
│   ├── eye_monitor.py                # Eye closure detection
│   ├── yawn_detector.py              # Yawn detection
│   ├── head_monitor.py               # Head position tracking
│   ├── drowsiness_analyzer.py        # Drowsiness analysis
│   ├── alert_system.py               # Alert management
│   ├── report_generator.py           # Report generation
│   └── utils.py                      # Utilities
├── models/                            # Pre-trained models
│   └── blazeface.tflite              # Face detection (optional)
├── reports/                           # Session reports (auto-generated)
├── recordings/                        # Video recordings (if enabled)
├── driver_profiles/                   # Calibration profiles (optional)
└── Thermal_Drowsiness/                # **NOVEL: Thermal detection system**
    ├── eye_drowsiness_video.py       # Thermal eye closure detection
    ├── mouth_drowsiness_video.py     # Thermal yawn detection
    ├── eye_mouth_combined_video.py   # Combined thermal analysis
    ├── requirements.txt               # Thermal-specific dependencies
    └── models/                        # Thermal model storage
```

## 🔬 Technical Details

### Detection Algorithms

**Eye Aspect Ratio (EAR):**
- Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
- Threshold: < 0.25 indicates eye closure
- Uses MediaPipe landmarks: 33, 133, 159, 145, 362, 386

**Mouth Aspect Ratio (MAR):**
- Formula: MAR = (||upper-lower||) / (||left-right||)
- Threshold: > 0.6 indicates yawning
- Uses MediaPipe landmarks: 13, 14, 82, 312, 87, 317, 61, 291

**Head Pose:**
- Extracted from MediaPipe 3D landmarks
- Pitch (nodding), Yaw (turning), Roll (tilting)
- Threshold: > 0.35 radians (~20°) deviation

### Scoring System

```
Combined Score = (Eye_Score × 0.60) + (Yawn_Score × 0.25) + (Head_Score × 0.15)

Dynamic adjustment:
- If Eye_Score > 0.5, Yawn_Weight reduced to 0.15
- Head detection is optional (--include_head flag)
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Test thoroughly on Raspberry Pi
4. Submit pull request with detailed description

## 📄 License

[Add your license information here]

## 📞 Support

For issues and questions:
- Check troubleshooting section
- Review configuration documentation
- Contact: [Your contact information]

## 🙏 Acknowledgments

- MediaPipe team for facial landmark detection
- OpenCV community for computer vision tools
- Research papers on drowsiness detection algorithms
- Thermal imaging community for advancing privacy-preserving monitoring technologies

---

## 📚 Quick Start Guide

### For RGB Detection (Standard):
```bash
source ddd_env/bin/activate  # Linux/Mac
python driver_drowsiness_detection.py --camera 0 --show_landmarks --alert
```

### For Thermal Detection (Novel):
```bash
cd Thermal_Drowsiness
pip install -r requirements.txt
python eye_mouth_combined_video.py --input your_thermal_video.mp4 --output output.mp4 --show-metrics
```

---

**Version:** 1.0  
**Last Updated:** December 2025  
**Platform:** Raspberry Pi 4 with Python 3.8+ (RGB) | Any system with thermal camera support (Thermal)
