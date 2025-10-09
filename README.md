# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

### 💻 Developed by: *Sudeshna Sarkar*  
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

## 🎬 Demo Instructions

This demo shows the *real-time safety monitoring system* in action.  

### 1️⃣ Live Camera Detection
- Click *▶ Start Detection* to start the camera feed.  
- The system detects *violations* like *no helmet, no mask, no vest* in real-time.  
- Violations are highlighted with *red bounding boxes, and safe items in **green*.  
- Logs and a *bar chart* are updated live.  
- Click *⏹ Stop Detection* to stop, while *logs and charts remain visible*.  
- An *email summary* is sent automatically with the violations detected.

### 2️⃣ Video File Detection
- Upload a *video file* (mp4, avi, mov).  
- The system takes *3 snapshots* from the video and detects violations.  
- Violations on snapshots are highlighted with bounding boxes.  
- Logs and charts are displayed similar to the live camera feed.  
- Email summary is sent automatically with detected violations.

### 3️⃣ Confidence Threshold
- Use the *slider in the sidebar* to adjust YOLO detection *confidence threshold*.  
- Lower threshold (0.2–0.3) is suitable for detecting subtle or small violations.  

### ✅ Features Observed During Demo
- Real-time detection on camera and uploaded videos  
- Violation logs with timestamps  
- Bar chart visualization  
- Email notification of main violations only  
<img width="1918" height="863" alt="Screenshot 2025-10-09 205114" src="https://github.com/user-attachments/assets/563e4c67-8408-4cbd-a60f-d0619cdef5d3" />
*Note:* This system is designed for demonstration and testing purposes during the Infosys Internship project, August 2025.

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
│   ├── val/images/labels/labels.cache
│   ├── train/images/labels/labels.cache
│   └── test/images/labels/labels.cache
├── detection/
├── docs/
├── model_training/
├── models/
├── notebooks/
├── logs/
├── predictions/
├── runs/
├── safetyeye_env/
├── src/ "contains preprocessing and augmented files"
├── test_images/
├── .dockerfile
├── .gitignore
├── README.md
└── yolov8n.pt
```

---

## 📌 Conclusion

*SafetyEye-AI* is an advanced AI-driven safety monitoring system designed to detect workplace compliance violations such as missing helmets, masks, and safety vests. The system supports both live camera feeds and video uploads, providing real-time dashboards, violation logs, and visual summaries. Detected violations can trigger **automated email notifications*, ensuring timely reporting and enhanced workplace safety.

This project demonstrates the seamless integration of computer vision, real-time analytics and interactive visualization, offering an end-to-end solution for monitoring and enforcing safety protocols effectively.

---


