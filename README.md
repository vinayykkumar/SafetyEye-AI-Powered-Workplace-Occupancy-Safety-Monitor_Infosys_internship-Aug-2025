---
title: Construction Site Safety Monitor
emoji: 🚧
colorFrom: orange
colorTo: red
sdk: streamlit
sdk_version: 1.50.0
app_file: construction_safety_app.py
pinned: false
license: mit
---

# Construction Site Safety Monitor

A Streamlit web application that uses YOLO models to monitor construction site safety by detecting workers and their personal protective equipment (PPE) in real-time video feeds.

## Features

- **Real-time Video Processing**: Upload and process construction site videos
- **PPE Detection**: Detects hard hats, safety vests, masks, and identifies missing equipment
- **Live Alerts**: Generates instant alerts when workers are not wearing required safety equipment
- **Safety Analytics**: Provides comprehensive statistics and violation reports
- **Model Selection**: Choose between YOLOv8n and YOLO11n models
- **Interactive Dashboard**: User-friendly interface with real-time monitoring

## Detected Classes

The app can detect the following 10 classes:
1. **Hardhat** (Green boxes)
2. **Mask** (Yellow boxes)
3. **NO-Hardhat** (Red boxes - triggers HIGH priority alert)
4. **NO-Mask** (Magenta boxes - triggers LOW priority alert)
5. **NO-Safety Vest** (Orange boxes - triggers MEDIUM priority alert)
6. **Person** (Cyan boxes)
7. **Safety Cone** (White boxes)
8. **Safety Vest** (Blue boxes)
9. **Machinery** (Gray boxes)
10. **Vehicle** (Pink boxes)

## Alert System

- 🔴 **High Priority**: Missing Hard Hat
- 🟡 **Medium Priority**: Missing Safety Vest
- 🟢 **Low Priority**: Missing Mask

## Installation

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv construction_safety_env
   source construction_safety_env/bin/activate  # On Windows: construction_safety_env\Scripts\activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
   
   **Note**: This project uses NumPy 1.x for compatibility with PyTorch and OpenCV. The requirements.txt includes version constraints to prevent compatibility issues.

4. **Ensure you have the trained models**:
   Make sure the following model files are in the `outputs/` folder:
   - `yolov8n.pt`
   - `yolo11n.pt`

## Usage

1. **Activate your virtual environment** (if using one):
   ```bash
   source construction_safety_env/bin/activate  # On Windows: construction_safety_env\Scripts\activate
   ```

2. **Start the Streamlit app**:
   ```bash
   streamlit run construction_safety_app.py
   ```

3. **Open your web browser** and navigate to the displayed URL (usually `http://localhost:8501`)

4. **Configure the app**:
   - Select your preferred model (YOLOv8n or YOLO11n) from the sidebar
   - Adjust the confidence threshold if needed (default: 0.5)

5. **Upload a video**:
   - Click "Browse files" and select a construction site video
   - Supported formats: MP4, AVI, MOV, MKV

6. **Start processing**:
   - Click "Start Processing" to begin analysis
   - Monitor real-time alerts in the right panel
   - View processed frames with bounding boxes and labels

7. **Review results**:
   - Check the safety alerts panel for violations
   - Review summary statistics including total persons detected, violation rates, and more

## File Structure

```
dataset11/
├── construction_safety_app.py    # Main Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                    # This file
├── data.yaml                    # Dataset configuration
├── outputs/                     # Trained models directory
│   ├── yolov8n.pt              # YOLOv8 nano model
│   └── yolo11n.pt              # YOLO11 nano model
├── train/                       # Training dataset
├── valid/                       # Validation dataset
└── test/                        # Test dataset
```

## Technical Details

- **Framework**: Streamlit for web interface
- **Model**: YOLO (You Only Look Once) for object detection
- **Video Processing**: OpenCV for video handling and frame processing
- **Real-time Analysis**: Frame-by-frame processing with bounding box visualization
- **Alert Generation**: Automatic detection of safety violations with severity classification

## Customization

You can customize the app by:

1. **Adjusting confidence thresholds** in the sidebar
2. **Modifying alert priorities** in the `analyze_safety_violations()` function
3. **Adding new detection classes** by updating the `class_names` dictionary
4. **Changing bounding box colors** in the `get_class_colors()` function

## Performance Tips

- For large videos, the app processes every 10th frame for display to maintain performance
- Adjust the confidence threshold to balance between detection accuracy and processing speed
- Use smaller video files for faster processing during testing

## Troubleshooting

1. **NumPy compatibility errors**: If you see "RuntimeError: Numpy is not available" or "_ARRAY_API not found":
   - Ensure you're using NumPy 1.x (not 2.x)
   - Recreate your virtual environment and reinstall from requirements.txt
   - The requirements.txt includes version constraints to prevent this issue

2. **Model loading errors**: Ensure the model files exist in the `outputs/` folder
3. **Video upload issues**: Check that your video format is supported (MP4, AVI, MOV, MKV)
4. **Memory issues**: Try processing smaller video files or reduce the confidence threshold
5. **Slow processing**: Lower the confidence threshold or process fewer frames

## Requirements

- Python 3.8+ (tested with Python 3.12)
- At least 4GB RAM (8GB recommended for large videos)
- CUDA-compatible GPU (optional, for faster processing)

### Key Dependencies:
- **NumPy 1.x** (1.21.0 to <2.0.0) - Version 2.x causes compatibility issues
- **OpenCV 4.8.1.78** - Stable version compatible with NumPy 1.x
- **PyTorch 2.2.0+** - For deep learning model inference
- **Streamlit 1.28.0+** - Web application framework

## License

This project is for educational and safety monitoring purposes in construction environments.