# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

---

## 🚀 Overview

SafetyEye is an AI-based system built using YOLOv8 to monitor workplace occupancy and PPE compliance in real time. It detects missing safety gear such as helmets, masks, and vests, tracks workers, and visualizes data through a Streamlit dashboard.

---

## 🔑 Features

* **PPE Detection:** Detects Hardhat, Mask, and Safety Vest usage.
* **Violation Alerts:** Instantly flags workers missing essential PPE.
* **Occupancy Monitoring:** Counts and tracks people in different zones.
* **Dashboard Analytics:** Displays live video feed, violation logs, and compliance trends.

---

## 🛠️ Tech Stack

Python • OpenCV • Ultralytics YOLOv8 • PyTorch • Streamlit • Albumentations • scikit-learn

---

## 🏗️ Setup

```bash
git clone <repo-url>
cd SafetyEye
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

**Dataset:** Construction Site Safety Image Dataset (Kaggle) → place in `dataset/raw/`.

---

## ⚙️ Usage

```bash
python src/preprocess.py             # Prepare data
python src/train.py --epochs 50      # Train YOLOv8
python src/realtime_detection.py     # Run real-time detection
streamlit run src/dashboard_app.py   # Launch dashboard
```

---

## 📂 Structure

```
├─ dataset/ (raw + processed)
├─ src/
│  ├─ preprocess.py
│  ├─ train.py
│  ├─ realtime_detection.py
│  └─ dashboard_app.py
└─ outputs/, runs/, requirements.txt
```

---

## 📊 Milestones

### 🧩 Milestone 1: Data Preparation & Setup

* Uploaded **`src/preprocess.py`** — handles data cleaning, augmentation, and YOLOv8-compatible structure creation.
* Processed Construction Site Safety Dataset from Kaggle and organized it into `dataset/processed/`.
* Configured virtual environment and installed all dependencies.

### 🤖 Milestone 2: Model Training & Validation

* Uploaded **`src/train.py`** and **`src/validate.py`** — used to train and evaluate YOLOv8.
* Fine-tuned model to detect helmets, masks, and vests.
* Achieved reliable detection accuracy on validation set (saved model in `runs/detect/`).

### 🎥 Milestone 3: Real-Time Detection & Alerts

* Uploaded **`src/realtime_detection.py`** — performs live video inference.
* Integrated PPE rule logic (`NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`) and violation alerting.
* Processes webcam or video feeds and saves annotated results in `outputs/`.

### 📊 Milestone 4: Dashboard Integration

* Uploaded **`src/dashboard_app.py`** — Streamlit dashboard for monitoring and analytics.
* Displays live video, violation logs, charts, and summary statistics.
* Added CSV export and unique violation tracking using ByteTrack-based IDs.

---

## ⚙️ Config Summary

Violation Classes: `{'NO-Hardhat','NO-Mask','NO-Safety Vest'}`
Confidence Threshold: `0.5`
Alert Cooldown: `5s`

---

## 📽️ Demo

Dashboard & detection demo: [CSE Dashboard](https://drive.google.com/file/d/1yi4jGCHp2YO9nfXNRqf1-9mjJ2yhnyXO/view?usp=sharing)

---
