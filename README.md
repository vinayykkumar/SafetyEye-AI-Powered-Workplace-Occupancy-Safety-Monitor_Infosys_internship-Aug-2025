# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

### 👩‍💻 Developed by: *Sudeshna Sarkar*  
*Group 4 | Infosys Internship Project – August 2025*  

---

## 📘 Overview  
*SafetyEye* is a *real-time AI-powered monitoring system* designed to detect workplace safety violations such as *no helmet, no vest, or no mask* from live camera feeds or uploaded video files.  
The system uses *YOLOv8* for object detection, a *Streamlit dashboard* for visualization, and an *automated email alert* system for reporting violations.

---

## ⚙ Features  

✅ *Live Camera Monitoring* – Detects violations in real time using your webcam.  
✅ *Video Upload Mode* – Allows detection from uploaded video files.  
✅ *Snapshot-based Detection* – Extracts and analyzes snapshots when video is uploaded.  
✅ *Confidence Threshold Slider* – Lets you adjust YOLO detection sensitivity.  
✅ *Violation Logs* – Displays all detected violations with timestamps.  
✅ *Bar Chart Visualization* – Summarizes violation counts clearly.  
✅ *Email Notification* – Automatically sends an email summary of violations after detection stops.  
✅ *Data Persistence* – Keeps logs and charts visible even after stopping detection.  

---

## 🧠 Tech Stack  

| Component | Technology Used |
|------------|----------------|
| *Frontend* | Streamlit |
| *Backend / AI Model* | YOLOv8 (Ultralytics) |
| *Programming Language* | Python |
| *Email Automation* | smtplib (Gmail App Password) |
| *Visualization* | Matplotlib, Pandas |
| *Environment* | Virtual Environment (venv) |

---

## 🚀 How It Works  

1. *Choose Input Source:*  
   - Select either *Camera* or *Video File* in the sidebar.  

2. *Start Detection:*  
   - Click *▶ Start Detection* to begin monitoring.  
   - The dashboard displays the live video, violation logs, and bar chart side-by-side.  

3. *Stop Detection:*  
   - Click *⏹ Stop Detection*.  
   - The system retains logs and charts for review.  

4. *Email Summary:*  
   - Once stopped, a summarized report of violations with timestamps is automatically emailed.  

---

## 📊 Output Example  
<img width="737" height="346" alt="image" src="https://github.com/user-attachments/assets/f0733ccc-b9f9-4ad6-81d1-e1ee9f255159" />

---

## 🗂 Folder Structure
```
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
│   ├── _pycache_/
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
