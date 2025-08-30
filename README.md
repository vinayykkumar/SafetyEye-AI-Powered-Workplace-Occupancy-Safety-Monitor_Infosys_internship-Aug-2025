# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

 - **Author:** Shally Katariya Group 1
 - **Context:** Infosys Internship Project, August 2025  

---

## 📌 Project Overview  

**SafetyEye** is an AI system designed for **real-time workplace safety monitoring**. It leverages **computer vision** and a **YOLOv8 object detection model** to analyze video feeds and ensure workers are wearing required **Personal Protective Equipment (PPE)** — specifically **hardhats** and **safety vests**.  

The goal is simple but critical:  
- Automate safety compliance checks.  
- Reduce workplace accidents.  
- Provide a scalable, high-performance monitoring solution.  

The project also includes a **full data processing pipeline** to prepare a robust, clean dataset for training the detection model.  

---

## 🚀 Key Features  

- **Real-Time Detection:** Identifies `hardhat`, `vest`, and `worker` classes instantly from video feeds.  
- **Compliance Logic:** Built-in rule engine checks if each detected worker has the required PPE (hardhat + vest).  
- **Advanced Data Pipeline:**  
  - Data cleaning  
  - Visual deduplication  
  - Augmentation (flip, rotate, brightness, etc.)  
  - Train/validation/test split  
- **GPU Accelerated:** Harnesses CUDA-enabled GPUs for fast **training** and **inference**.  

---

## 📂 Project Structure  

```plaintext
SafetyEye-AI/
├── .vscode/
├── alerts/
├── dashboard/
├── data/
│   ├── processed/
│   │   ├── augmented_data/
│   │   ├── final_dataset/
│   │   └── ppe_dataset/
│   └── raw/
├── data_preprocessing/
│   ├── __pycache__/
│   ├── data_augmentation.py
│   ├── duplicate_data.py
│   ├── preprocess.py
│   └── resize_normalise.py
├── detection/
├── docs/
├── model_training/
├── models/
├── notebooks/
├── outputs/
├── predictions/
├── runs/
├── safetyeye_env/
├── src/
├── test_images/
├── .dockerfile
├── .gitignore
├── README.md
└── yolov8n.pt
```

## ⚙️ Setup and Installation

### Prerequisites

 - Python 3.10+
 - NVIDIA GPU with CUDA drivers installed

### Installation Steps

Clone Repository
```
git clone https://github.com/vinayykkumar/SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025.git
cd SafetyEye-AI
```

### Create Virtual Environment
```
python -m venv safetyeye_env
```

### Windows:
```
safetyeye_env\Scripts\activate
```

### macOS/Linux:
```
source safetyeye_env/bin/activate
```

### Install Dependencies

First, install PyTorch with CUDA support from the official site.
Then, install the rest:
```
pip install ultralytics albumentations opencv-python-headless imagehash matplotlib tqdm
```

### Verify Installation
```
yolo check
```

## 🔄 Data Processing Pipeline

### Deduplication (duplicate_data.py)

 - Uses image hashing to detect and remove visually similar images.
 - Prevents bias from redundant data.

### Preparation (preprocess.py)

 - Cleans corrupted data and missing labels.
 - Splits into train, validation, and test sets.

### Augmentation (data_augmentation.py)

 - Applies random transformations: flip, rotate, brightness change, etc.
 - Increases dataset diversity.

### Final Assembly

 - Combines clean original data with augmented samples.
 - Produces the master dataset (final_dataset/).

## ✅ Milestone 1 Completed (Week 1–2):

 - Dataset downloaded and explored (Roboflow Construction Site Safety Dataset).
 - Data cleaned, deduplicated, and formatted into YOLOv8-compatible format.
 - Train/val/test splits created.
 - Development + training environment set up with all required dependencies.

## 🖥️ How to Use

### 1. Training the Model
```
yolo task=detect mode=train model=yolov8n.pt data=final_dataset.yaml epochs=100 imgsz=640 batch=16 name=yolov8n_final_model
```
 - final_dataset.yaml = configuration file for master dataset

### 2. Running Validation
```
yolo task=detect mode=val model=runs/detect/yolov8n_final_model/weights/best.pt data=final_dataset.yaml
```

### 3. Making Predictions
```
python predictions/predict.py
```
 - Runs inference on a sample image.
 - Prints a compliance summary (who’s safe, who’s not).

## 🔮 Future Work

 - Dashboard Development: Build a real-time web UI (Streamlit/Flask).
 - Alert System: Automated notifications for detected violations.
 - Model Optimization: Experiment with larger YOLOv8 variants (yolov8s, yolov8m) + hyperparameter tuning.
 - Expand PPE Classes: Add detection for safety glasses, gloves, and ear protection.
