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

## 🚧 Construction Site Safety Monitor

A comprehensive Streamlit web application that leverages custom-trained YOLOv8 models to monitor construction site safety by detecting workers and their personal protective equipment (PPE) in real-time video feeds and images.

## 🚀 Key Features

### Custom Trained Model

- **Specialized Training**: YOLOv8 model trained on construction site safety dataset for 100 epochs
- **Enhanced Accuracy**: Superior detection performance for construction-specific objects and safety violations
- **PPE Optimization**: Fine-tuned for detecting hard hats, safety vests, masks, and safety violations
- **Real-world Performance**: Trained on authentic construction site imagery for practical applications

### Core Functionality

- **📹 Real-time Video Processing**: Upload and analyze construction site videos
- **🦺 PPE Detection**: Comprehensive detection of safety equipment and violations
- **🚨 Live Alert System**: Instant notifications when workers lack required safety equipment
- **📊 Safety Analytics**: Detailed statistics and violation reports with exportable data
- **🎯 Dual Model Support**: Custom trained model (recommended) and base YOLOv8 options
- **💻 Interactive Dashboard**: Intuitive web interface with real-time monitoring capabilities
- **🖼️ Image Annotation Tool**: Standalone application for single image analysis and manual annotation

## 🎯 Detection Capabilities

The application can detect and classify the following 10 object types:

| Class | Color Code | Priority Level |
|-------|------------|----------------|
| **Hardhat** | 🟢 Green | Normal |
| **Mask** | 🟡 Yellow | Normal |
| **NO-Hardhat** | 🔴 Red | HIGH Alert |
| **NO-Mask** | 🟣 Magenta | LOW Alert |
| **NO-Safety Vest** | 🟠 Orange | MEDIUM Alert |
| **Person** | 🔵 Cyan | Normal |
| **Safety Cone** | ⚪ White | Normal |
| **Safety Vest** | 🔵 Blue | Normal |
| **Machinery** | ⚫ Gray | Normal |
| **Vehicle** | 🩷 Pink | Normal |

## Alert System

- 🔴 **High Priority**: Missing Hard Hat
- 🟡 **Medium Priority**: Missing Safety Vest
- 🟢 **Low Priority**: Missing Mask

## 📦 Installation

### Prerequisites

- Python 3.8+ (tested with Python 3.12)
- At least 4GB RAM (8GB recommended for large videos)
- CUDA-compatible GPU (optional, for faster processing)

### Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/saketh-005/construction-safety-monitor.git
   cd construction-safety-monitor
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv construction_safety_env
   source construction_safety_env/bin/activate  # On Windows: construction_safety_env\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Project Structure

``` bash
construction-safety-monitor/
├── construction_safety_app.py    # Main Streamlit application
├── image_annotation_app.py       # Image annotation tool
├── training_notebook.ipynb       # Training and EDA notebook
├── requirements.txt              # Python dependencies
├── config/
│   ├── ppe_data.yaml            # Dataset configuration
│   └── training_args.yaml       # Training parameters
├── models/
│   ├── construction_best.pt     # Custom trained model (recommended)
│   └── yolov8n.pt              # Base YOLOv8 model
└── training_logs/
    └── results.csv             # Training metrics and results
```

## 🚀 Usage

### Main Application (Video Processing)

1. **Start the application**:

   ```bash
   streamlit run construction_safety_app.py
   ```

2. **Open your browser** and navigate to the displayed URL (typically `http://localhost:8501`)

3. **Configure settings**:
   - Select your preferred model (custom trained recommended)
   - Adjust confidence threshold (default: 0.5)
   - Set alert sensitivity levels

4. **Upload video**:
   - Supported formats: MP4, AVI, MOV, MKV
   - Maximum file size: 200MB

5. **Process and monitor**:
   - Click "Start Processing" to begin real-time analysis
   - View live alerts and safety violations
   - Export statistics and reports

### Image Annotation Tool

1. **Launch the annotation tool**:

   ```bash
   streamlit run image_annotation_app.py
   ```

2. **Upload single images** for detailed analysis and manual annotation
3. **Review detection results** with bounding boxes and confidence scores
4. **Export annotated images** for documentation and training

### Training Notebook

Explore the comprehensive training process:

```bash
jupyter notebook training_notebook.ipynb
```

- **Exploratory Data Analysis (EDA)** of construction safety dataset
- **Model training pipeline** with YOLOv8
- **Performance metrics** and visualization
- **Dataset preparation** techniques

## 📊 Model Performance

Our custom-trained model achieves excellent performance metrics:

- **Training Duration**: 100 epochs with early stopping
- **Dataset**: Construction site safety images with PPE annotations
- **mAP@0.5**: 0.809 (80.9% mean Average Precision)
- **Precision**: 89.9%
- **Recall**: 73.1%
- **Model Size**: Optimized for real-time inference

Detailed training logs and metrics are available in `training_logs/results.csv`.

## 🛠️ Configuration

Customize the application behavior through configuration files:

- **`config/ppe_data.yaml`**: Dataset paths and class definitions
- **`config/training_args.yaml`**: Model training parameters

Modify these files to adapt the system to your specific requirements.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

**Disclaimer**: This application is designed for educational and safety monitoring purposes in construction environments. Users should ensure compliance with local safety regulations and standards.

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for the base object detection framework
- **Streamlit** for the intuitive web application framework
- **Roboflow** for construction site safety datasets
- **OpenCV** for computer vision processing capabilities

---

## 💙 Built with dedication for construction site safety

For questions or support, please open an issue in the GitHub repository.
