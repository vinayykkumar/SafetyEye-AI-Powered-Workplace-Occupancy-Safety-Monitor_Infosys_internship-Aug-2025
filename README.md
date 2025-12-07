SafetyEye – AI-Powered Workplace PPE Monitoring System

SafetyEye is an AI-powered computer vision system designed to monitor workplace safety using CCTV or live video streams.
It automatically detects Personal Protective Equipment (PPE) such as Helmets, Safety Vests, Masks, and highlights safety violations in real-time.
The system logs detected violations and can generate alarms, helping industries maintain compliance and prevent accidents.


Key Features

-> Real-time PPE detection for:

Helmet
Safety Vest
Mask
Machinery (Proximity Awareness)
No Helmet
No Vest
No Mask
Safety Cone

-> Draws bounding boxes with class labels for each worker/action
-> Alerts when safety violations are detected
-> Stores violation logs for record keeping and audit
-> Supports CCTV monitoring & recorded video analysis
-> Lightweight deployment — runs on GPU or CPU

-> Tech Stack
Component	Technology
Programming Language	Python
Object Detection	YOLOV8 (You Only Look Once)
Computer Vision	OpenCV
Deep Learning	PyTorch / TensorFlow (depending on training pipeline)
Deployment	Docker (optional)
Logging/Storage	Database / CSV / Local logs


-> System Architecture
Video Stream / CCTV / File
        ↓
   Frame Processing
        ↓
   YOLO-based PPE Detection
        ↓
Violation Classification
        ↓
Alert System + Violation Log
        ↓
Dashboard / Monitoring Interface (optional)

-> Folder Structure

├── models/                 → Trained YOLO models
├── src/                    → Core source code
├── outputs/                → Annotated images / video results
├── data/                   → Dataset (if included)
├── requirements.txt        → Dependencies
├── Dockerfile              → Container deployment file
└── README.md               → Documentation



-> System Output

Event	Action
PPE present	Green bounding box
PPE missing	Red bounding box + Alert
Violation	Logged to database / CSV / local log


-> Applications

Construction industry
Manufacturing and heavy machinery plants
Mining and oil & gas
Electric utility workforce safety
Smart CCTV monitoring for industrial safety

-> Limitations

Accuracy decreases in:
Low-light environments
Camera glare / blur
Workers partially covered / occluded
Dataset must be expanded for better generalization

-> Future Scope

Multi-camera workplace analytics dashboard
Safety behavior scoring & compliance report
Integration with attendance and worker ID tracking
Edge AI deployment on NVIDIA Jetson / IoT devices
Audio-based alerts on factory floor

-> Contributors
Developed by: Revanth Kumar Surisetti
Mentor: Infosys Employer
Infosys Internship – August 2025
