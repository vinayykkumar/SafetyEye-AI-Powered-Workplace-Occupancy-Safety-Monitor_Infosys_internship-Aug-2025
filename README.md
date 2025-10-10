# SafetyEye-AI-Powered-Workplace-Occupancy-Safety-Monitor_Infosys_internship-Aug-2025

📌 Project Overview:
--------------------
SafetyEye is an AI-powered safety monitoring system designed to improve compliance and safety standards at construction sites and industrial workplaces. Using real-time computer vision and deep learning, the system detects the presence (or absence) of essential PPE (Personal Protective Equipment) like helmets and vests.

It provides a live monitoring dashboard, flags safety violations, and maintains automated logs for safety audits and analysis.

🎯 Objectives:
--------------
- Detect PPE like helmets and safety vests from images and videos
- Monitor workplace in real-time using webcam or video feed
- Flag safety violations automatically (e.g., missing helmet)
- Display live detection on an interactive dashboard
- Log violation data and generate compliance analytics

🔄 Project Workflow:
---------------------
The project was executed in 4 major milestones:

🟩 **Milestone 1: Data Preparation & Environment Setup 
- Downloaded construction site safety image dataset
- Labeled and formatted data in YOLOv8-compatible format
- Split data into training, validation, and testing sets
- Set up Python environment with PyTorch, OpenCV, Ultralytics
- Defined rule logic (e.g., "no helmet = violation")

🟦 **Milestone 2: Model Training & PPE Detection 
- Trained YOLOv8 model using annotated dataset
- Performed data augmentation to improve robustness
- Validated and tested model for precision and recall
- Tuned hyperparameters for better accuracy
- Evaluated with mAP, precision, and recall metrics

🟧 **Milestone 3: Real-Time Detection & Alert System 
- Developed real-time detection using live video stream
- Overlayed bounding boxes and labels on video feed
- Implemented logic for identifying violations
- Integrated alert system (console logs, email notifications)
- Stored violations in CSV/database format for tracking

🟨 **Milestone 4: Dashboard Development & Final Integration 
- Built an interactive dashboard using Streamlit
- Displayed live detection feed with overlays
- Added compliance statistics (charts, tables)
- Retrieved and stored violation logs
- Conducted full system testing (accuracy, latency)
- Prepared final report, codebase, and walkthrough video

🔧 Technical Requirements:
---------------------------
**Programming Language & Tools:**
- Python 3.11
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV
- Streamlit
- NumPy, Pandas
- Matplotlib or Plotly


**Optional Tools:**
- Git (version control)
- Streamlit Cloud / Heroku / AWS (for deployment)
- VS Code or PyCharm (IDE)


📈 Dashboard Features:

Live video feed with detection overlay

Real-time safety violation alerts

Charts showing PPE compliance statistics

Violation logs with timestamps

Exportable reports for safety audits

📦 requirements.txt Sample:
ultralytics

torch

opencv-python

streamlit

numpy

pandas

matplotlib

plotly

requests

💡 Future Enhancements:

Expand to detect more PPE types (gloves, goggles)

Multi-camera support across multiple locations

Face recognition for worker identification

SMS/Email/Push notification alerts

Mobile app interface for supervisors

🧪 Testing Checklist:

✔️ Model tested with multiple lighting conditions
✔️ Verified detection accuracy for helmets and vests
✔️ Dashboard tested for real-time performance
✔️ Alerts triggered correctly on rule violations
✔️ Logs created and stored automatically

📝 Deliverables:

Cleaned dataset (YOLOv8 format)

Trained YOLOv8 model

Real-time detection system

Streamlit dashboard with analytics

Violation logging system.
