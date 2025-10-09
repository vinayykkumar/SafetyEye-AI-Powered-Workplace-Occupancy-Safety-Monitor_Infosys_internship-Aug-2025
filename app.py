import streamlit as st
import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import matplotlib.pyplot as plt
import tempfile
import random
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = r"C:\safetyeye\runs\helmet_augmented_final\yolo_augmented_m_fresh\weights\best.pt"
EMAIL = "minimeeee44@gmail.com"          # sender email
APP_PASSWORD = "nptn heyq glcp iltg"     # Gmail App Password
TO_EMAIL = "minimeeee44@gmail.com"       # receiver email

# -----------------------------
# LOAD MODEL
# -----------------------------
st.set_page_config(page_title="Safety Monitoring Dashboard", layout="wide")
st.title("Live Safety Monitoring Dashboard")
model = YOLO(MODEL_PATH)

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Settings")
source_option = st.sidebar.radio("Select Input Source", ["Camera", "Video File"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
show_overlay = st.sidebar.checkbox("Show Overlays", value=True)

if source_option == "Camera":
    camera_index = st.sidebar.number_input("Camera Index", min_value=0, value=0)
else:
    uploaded_video = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# -----------------------------
# SESSION STATE
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "violation_log" not in st.session_state:
    st.session_state.violation_log = pd.DataFrame(columns=["Timestamp", "Violation"])

if "snapshot_done" not in st.session_state:
    st.session_state.snapshot_done = False

# -----------------------------
# START / STOP BUTTONS
# -----------------------------
col1, col2 = st.columns(2)
start_detection = col1.button("▶ Start Detection")
stop_detection = col2.button("⏹ Stop Detection")

if start_detection:
    st.session_state.running = True
    st.session_state.snapshot_done = False

if stop_detection:
    st.session_state.running = False

# -----------------------------
# DISPLAY AREA
# -----------------------------
video_frame = st.empty()
log_col, chart_col = st.columns([2, 1])
log_table = log_col.empty()
bar_chart = chart_col.empty()
snapshot_images_placeholder = st.empty()

# -----------------------------
# VIDEO / CAMERA CAPTURE
# -----------------------------
cap = None
if source_option == "Camera":
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error("Cannot open camera.")
        st.stop()
else:
    if uploaded_video is not None:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name
        cap = cv2.VideoCapture(temp_video_path)
    else:
        st.warning("Please upload a video to start detection.")
        st.stop()

# -----------------------------
# DETECTION LOOP
# -----------------------------
try:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            if source_option == "Camera":
                st.warning("Failed to capture frame from camera.")
            else:
                st.warning("End of video reached.")
            break

        results = model(frame, conf=confidence_threshold)

        # Process detections
        for r in results:
            boxes = r.boxes
            for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                label = r.names[int(cls_id)]
                if label in ["no_helmet", "no_mask", "no_vest"]:
                    # Log violation
                    st.session_state.violation_log = pd.concat([
                        st.session_state.violation_log,
                        pd.DataFrame([{
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Violation": label
                        }])
                    ], ignore_index=True)

                    if show_overlay:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    if show_overlay:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display live frame
        video_frame.image(frame, channels="BGR", width=480)

        # Update logs and bar chart
        if not st.session_state.violation_log.empty:
            log_table.dataframe(st.session_state.violation_log)
            bar_data = st.session_state.violation_log["Violation"].value_counts()
            fig, ax = plt.subplots(figsize=(3,3))
            ax.bar(bar_data.index, bar_data.values, color=["#FF0000", "#FFA500", "#FF69B4"])
            ax.set_ylabel("Count")
            ax.set_title("Violations")
            bar_chart.pyplot(fig)

        # For uploaded video, take 3 snapshots once
        if source_option == "Video File" and not st.session_state.snapshot_done:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            snapshot_indices = sorted(random.sample(range(total_frames), min(3, total_frames)))
            snapshots = []

            for idx in snapshot_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, snap_frame = cap.read()
                if not ret:
                    continue
                snap_results = model(snap_frame, conf=confidence_threshold)
                # Draw detections
                for r in snap_results:
                    boxes = r.boxes
                    for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
                        label = r.names[int(cls_id)]
                        if label in ["no_helmet", "no_mask", "no_vest"]:
                            st.session_state.violation_log = pd.concat([
                                st.session_state.violation_log,
                                pd.DataFrame([{
                                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "Violation": label
                                }])
                            ], ignore_index=True)

                            if show_overlay:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(snap_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(snap_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        else:
                            if show_overlay:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(snap_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(snap_frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                snapshots.append(snap_frame)
            # Show snapshots
            for i, snap in enumerate(snapshots):
                snapshot_images_placeholder.image(snap, caption=f"Snapshot {i+1}", width=320)
            st.session_state.snapshot_done = True

except KeyboardInterrupt:
    pass
finally:
    if cap:
        cap.release()

# -----------------------------
# AFTER STOP DETECTION
# -----------------------------
st.session_state.running = False
st.success("Detection stopped ✅")

# Show final logs and bar chart
log_col, chart_col = st.columns([2, 1])
with log_col:
    st.subheader("Violation Logs")
    log_table.dataframe(st.session_state.violation_log)

with chart_col:
    st.subheader("Violation Summary")
    if not st.session_state.violation_log.empty:
        bar_data = st.session_state.violation_log["Violation"].value_counts()
        fig, ax = plt.subplots(figsize=(3,3))
        ax.bar(bar_data.index, bar_data.values, color=["#FF0000", "#FFA500", "#FF69B4"])
        ax.set_ylabel("Count")
        ax.set_title("Violations")
        bar_chart.pyplot(fig)

# -----------------------------
# SEND EMAIL
# -----------------------------
if not st.session_state.violation_log.empty:
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL
        msg['To'] = TO_EMAIL
        msg['Subject'] = f"Safety Violation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        unique_violations = st.session_state.violation_log["Violation"].unique()
        body = "\n".join([f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {v}" for v in unique_violations])
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL, APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        st.success("Violation summary email sent ✅")
    except Exception as e:
        st.error(f"Failed to send email: {e}")
else:
    st.info("No violations detected. No email sent.")