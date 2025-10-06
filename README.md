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

### Analysis Modes

#### 📹 Video Analysis

- **Real-time Processing**: Upload and analyze construction site videos with live frame-by-frame updates
- **Live Monitoring Dashboard**: Compact view showing video feed, alerts, and statistics simultaneously
- **Full Video Analysis**: Process entire videos and generate comprehensive reports
- **Progress Tracking**: Real-time progress bar and frame-by-frame status updates

#### 🖼️ Image Analysis

- **Single Image Processing**: Upload and analyze individual images (JPG/PNG/BMP/TIFF)
- **Instant Detection**: Quick PPE violation detection on static images
- **Adjustable Confidence**: Fine-tune detection sensitivity for images
- **Visual Results**: Annotated images with bounding boxes and labels

#### 📊 Violation History

- **Master CSV Database**: Persistent storage of all violation data across sessions
- **Historical Analytics**: View and analyze violation trends over time
- **Advanced Filtering**: Filter by date, severity, session, and more
- **Interactive Charts**: Visualizations showing severity distribution and violation trends
- **Multi-Format Export**: Download filtered or complete data as CSV or interactive HTML reports

### Core Functionality

- **🦺 PPE Detection**: Comprehensive detection of safety equipment and violations
- **🚨 Smart Alert System**: Real-time violation alerts with severity-based prioritization
- **📊 Advanced Analytics**: Detailed statistics, KPIs, and safety compliance metrics
- **🎯 Dual Model Support**: Custom trained model (recommended) and base YOLOv8 options
- **💻 Interactive Dashboard**: Intuitive web interface with real-time monitoring capabilities
- **� Enhanced Vehicle Filtering**: Advanced multi-method filtering to reduce false positives from vehicles and machinery
- **👨‍💻 Developer Mode**: Technical metrics including FPS, detection stats, and processing information

### Email Notification System

- **📧 Real-time Alerts**: Instant email notifications when violations are detected (with frame images)
- **📄 CSV Summary Reports**: Email delivery of detailed violation data in CSV format
- **📊 HTML Reports**: Beautiful, interactive HTML reports with embedded graphs and analytics
- **⚙️ Queue-based Processing**: Non-blocking email system for smooth performance
- **📈 Email Status Tracking**: Monitor sent/failed emails with detailed logs

### Report Generation

- **📊 CSV Reports**: Comprehensive violation data in spreadsheet format
- **📄 HTML Reports**: Interactive reports featuring:
  - Executive summary with key metrics
  - Violation breakdown by severity
  - Interactive graphs (pie charts, timelines, compliance trends)
  - Searchable and sortable violation tables
  - Key performance indicators
  - Safety recommendations
  - Top violation types analysis
- **📥 Multiple Export Options**: Download filtered or complete datasets
- **🎨 Professional Styling**: Modern, responsive design with print support

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

```bash
construction-safety-monitor/
├── construction_safety_app.py      # Main Streamlit application
├── training_notebook.ipynb         # Training and EDA notebook
├── requirements.txt                # Python dependencies
├── violation_history_master.csv    # Master violation history database
├── config/
│   ├── ppe_data.yaml              # Dataset configuration
│   └── training_args.yaml         # Training parameters
├── models/
│   ├── construction_best.pt       # Custom trained model (recommended)
│   └── yolov8n.pt                 # Base YOLOv8 model
└── training_logs/
    └── results.csv                 # Training metrics and results
```

## 🚀 Usage

### Main Application

1. **Start the application**:

   ```bash
   streamlit run construction_safety_app.py
   ```

2. **Open your browser** and navigate to the displayed URL (typically `http://localhost:8501`)

3. **Select Analysis Mode**:

   Choose from three modes in the sidebar:

   - **📹 Video Analysis**: Process construction site videos
   - **🖼️ Image Analysis**: Analyze single images
   - **📊 Violation History**: View historical violation data

### Video Analysis Mode

1. **Configure settings**:
   - Select your preferred model (custom trained recommended)
   - Adjust confidence threshold (default: 0.4)
   - Enable Developer Mode for technical metrics (optional)

2. **Configure Email Notifications** (optional):
   - Enable email notifications checkbox
   - Enter recipient email address
   - Choose notification type:
     - 🚨 Real-time Alerts: Instant emails with frame images when violations detected
     - 📊 CSV Summary: Email CSV report after analysis
     - 📄 HTML Report: Email interactive HTML report with graphs

3. **Upload and Process Video**:
   - Supported formats: MP4, AVI, MOV, MKV
   - Choose processing mode:
     - **Live Frame Updates**: See violations in real-time with live dashboard
     - **Full Video Analysis**: Process entire video, then view results

4. **View Results**:
   - Real-time alerts and statistics
   - Detailed analytics with expandable graphs
   - Download reports (CSV or HTML format)

### Image Analysis Mode

1. **Upload Image**:
   - Supported formats: JPG, PNG, BMP, TIFF
   - Adjust detection confidence threshold

2. **View Results**:
   - Annotated image with bounding boxes
   - Violation summary table
   - Safety compliance metrics

3. **Download Results**:
   - Save annotated image
   - Export violation data as CSV

### Violation History Dashboard

1. **View Overall Statistics**:
   - Total violations across all sessions
   - Total analysis sessions
   - High severity violation count
   - Latest session date

2. **Apply Filters**:
   - Filter by date
   - Filter by severity (High/Medium/Low)
   - Filter by session ID

3. **Visualize Data**:
   - Severity distribution chart
   - Violations by date timeline

4. **Download Reports**:
   - 📊 Filtered CSV: Download filtered violations
   - 📄 Filtered HTML: Interactive report of filtered data
   - 📁 Full CSV: Download complete violation history
   - 📄 Full HTML: Interactive report of all violations

### Developer Mode Features

Enable Developer Mode in the sidebar to access:

- **JSON Configuration**: View model and processing configuration
- **Per-Frame Metrics**:
  - Detection statistics (persons, violations, filtered detections)
  - Processing information (frame dimensions, processing time)
  - Model metrics (FPS, inference speed)
- **Violation Details**: Expandable panel with detailed violation information

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

## 🛠️ Advanced Features

### Enhanced Vehicle Filtering

The system uses multiple aggressive filtering methods to reduce false positives from vehicles and machinery:

1. **Direct Overlap Detection**: IoU-based overlap checking with low threshold
2. **Expanded Zone Filtering**: 40% larger detection zones around vehicles/machinery
3. **Containment Analysis**: Checks if violations are inside super-expanded vehicle areas (60% larger)
4. **Size-based Filtering**: Filters out detections too large to be person-related

### Spatial PPE Association

Advanced spatial heuristics to accurately associate PPE with workers:

- **Hardhat Detection**: Expects detection at top of person box (upper ~45%)
- **Vest Detection**: Looks for detection in middle-lower torso region
- **Mask Detection**: Expects detection around face region (upper ~50%)
- **Fallback IoU**: Uses Intersection over Union when spatial heuristics don't match

### Violation Deduplication

Intelligent deduplication system prevents counting the same violation multiple times:

- Person-based violations tracked per individual
- NO-* detections filtered against existing person violations
- Spatial overlap checking to avoid duplicate alerts
- Confidence-based filtering (threshold: 0.6 for NO-* detections)

### Email Queue System

Thread-safe, non-blocking email processing:

- Background worker thread handles email sending
- Queue-based system prevents UI blocking
- Real-time status tracking (sent/failed counts)
- Recent email logs for debugging
- Automatic worker shutdown on completion

### Report Analytics

Comprehensive analytics included in reports:

- **Risk Assessment**: Calculated risk scores and severity rates
- **Compliance Metrics**: Safety compliance percentages and trends
- **Violation Density**: Violations per worker analysis
- **Temporal Analysis**: First/last violation tracking
- **Hotspot Detection**: Identifies frames with most violations
- **Recommendations**: Automated safety improvement suggestions

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
- **Matplotlib** for data visualization and graph generation
- **Pandas** for data manipulation and analysis

---

## 💙 Built with dedication for construction site safety

**Recent Updates:**

- ✅ Added Image Analysis mode for single image processing
- ✅ Implemented Violation History dashboard with master CSV database
- ✅ Added email notification system with real-time alerts and HTML reports
- ✅ Enhanced vehicle filtering with multi-method approach
- ✅ Implemented spatial PPE association for accurate detection
- ✅ Added Developer Mode with technical metrics
- ✅ Created interactive HTML reports with embedded graphs
- ✅ Improved UI with expandable sections and compact layouts

For questions or support, please open an issue in the GitHub repository.
