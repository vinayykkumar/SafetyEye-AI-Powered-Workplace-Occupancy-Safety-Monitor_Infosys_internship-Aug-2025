import streamlit as st
import cv2
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import numpy as np
import time
import os
from io import BytesIO, StringIO

# ---------- CONFIG ----------
MODEL_PATH = r"D:\SAFETYEYE 2\source_file\best.pt"  # Update your model path here
VIOLATION_CLASSES = ["no-helmet", "no-vest"]
LOG_FILE = "violations.csv"

# ---------- PAGE SETUP ----------
st.set_page_config(
    page_title="🛡️ SafetyEye Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ SafetyEye Real-Time Safety Monitoring")

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------- SESSION STATE ----------
if "running" not in st.session_state:
    st.session_state.running = False
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "violations_log" not in st.session_state:
    if os.path.exists(LOG_FILE):
        st.session_state.violations_log = pd.read_csv(LOG_FILE)
    else:
        st.session_state.violations_log = pd.DataFrame(columns=["Timestamp", "Violation"])
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "file_type" not in st.session_state:
    st.session_state.file_type = None  # 'video' or 'image'

# ---------- SIDEBAR ----------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Video or Image", type=["mp4", "mov", "avi", "jpg", "jpeg", "png"])

start_btn = st.sidebar.button("▶️ Start Detection")
stop_btn = st.sidebar.button("⏹️ Stop Detection")
reset_btn = st.sidebar.button("🗑️ Reset Logs")
download_btn = st.sidebar.button("⬇️ Download Logs CSV")

st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown(
    """
    - Upload a video or image file to analyze.
    - Click **Start Detection** to begin.
    - Use **Stop Detection** to pause.
    - Use **Reset Logs** to clear violation data.
    - Download logs anytime.
    """
)

if reset_btn:
    st.session_state.violation_count = 0
    st.session_state.total_frames = 0
    st.session_state.violations_log = pd.DataFrame(columns=["Timestamp", "Violation"])
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    st.sidebar.success("Logs reset!")

if start_btn:
    if st.session_state.uploaded_file is not None:
        st.session_state.running = True
    else:
        st.sidebar.warning("Please upload a video or image first.")

if stop_btn:
    st.session_state.running = False

# Download logs CSV
if download_btn:
    csv_buffer = StringIO()
    st.session_state.violations_log.to_csv(csv_buffer, index=False)
    st.sidebar.download_button(
        label="Download violation logs CSV",
        data=csv_buffer.getvalue(),
        file_name="violation_logs.csv",
        mime="text/csv"
    )

# Handle file upload
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext in ["mp4", "mov", "avi"]:
        st.session_state.file_type = "video"
    else:
        st.session_state.file_type = "image"
    st.sidebar.success(f"Loaded {uploaded_file.name} ({st.session_state.file_type})")

# ---------- LAYOUT ----------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Live Video Feed / Image")
    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Compliance Rate")
    compliance_placeholder = st.empty()

    st.subheader("🚨 Violation Alerts")
    alert_placeholder = st.empty()

    st.subheader("📈 Violation Counts")
    chart_placeholder = st.empty()

    st.subheader("📋 Violation Logs (Recent 10)")
    log_placeholder = st.empty()

# ---------- LOGGING FUNCTION ----------
def log_violation(vtype):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame({"Timestamp": [timestamp], "Violation": [vtype]})
    st.session_state.violations_log = pd.concat(
        [new_entry, st.session_state.violations_log], ignore_index=True
    )
    st.session_state.violations_log.to_csv(LOG_FILE, index=False)

# ---------- VIDEO PROCESSING FUNCTION ----------
def process_video(video_bytes):
    cap = cv2.VideoCapture()
    cap.open(video_bytes)

    violation_counts = {v: 0 for v in VIOLATION_CLASSES}

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.info("🔁 Video ended.")
            st.session_state.running = False
            break

        st.session_state.total_frames += 1

        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot()

        detected_violations = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            if class_name in VIOLATION_CLASSES:
                detected_violations.append(class_name)

        if detected_violations:
            unique_violations = set(detected_violations)
            for v in unique_violations:
                violation_counts[v] += 1
                log_violation(v)
            st.session_state.violation_count += 1
            alert_placeholder.error(f"🚨 Violations Detected: {', '.join(unique_violations)}")
        else:
            alert_placeholder.success("✅ All safety gear detected.")

        compliance = 100 - (st.session_state.violation_count / st.session_state.total_frames) * 100
        compliance_placeholder.metric("Compliance Rate", f"{compliance:.2f}%")

        frame_placeholder.image(annotated_frame, channels="BGR")

        chart_data = pd.DataFrame({
            "Violation": list(violation_counts.keys()),
            "Count": list(violation_counts.values())
        })
        chart_placeholder.bar_chart(chart_data.set_index("Violation"))

        if not st.session_state.violations_log.empty:
            log_placeholder.dataframe(st.session_state.violations_log.head(10))
        else:
            log_placeholder.info("No violations recorded yet.")

        time.sleep(0.03)

    cap.release()

# ---------- IMAGE PROCESSING FUNCTION ----------
def process_image(image_bytes):
    # Read image bytes to numpy array
    image = np.frombuffer(image_bytes.read(), np.uint8)
    frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

    results = model.predict(frame, verbose=False)
    annotated_frame = results[0].plot()

    detected_violations = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        if class_name in VIOLATION_CLASSES:
            detected_violations.append(class_name)

    if detected_violations:
        unique_violations = set(detected_violations)
        for v in unique_violations:
            log_violation(v)
        alert_placeholder.error(f"🚨 Violations Detected: {', '.join(unique_violations)}")
    else:
        alert_placeholder.success("✅ All safety gear detected.")

    compliance = 100 - (len(st.session_state.violations_log) / 1) * 100
    compliance_placeholder.metric("Compliance Rate (Image)", f"{compliance:.2f}%")

    frame_placeholder.image(annotated_frame, channels="BGR")

    # Show logs
    if not st.session_state.violations_log.empty:
        log_placeholder.dataframe(st.session_state.violations_log.head(10))
    else:
        log_placeholder.info("No violations recorded yet.")

# ---------- MAIN RUN ----------
if st.session_state.uploaded_file:
    if st.session_state.file_type == "video":
        # Need to save uploaded video temporarily because cv2.VideoCapture doesn't accept bytes directly
        tfile = st.session_state.uploaded_file
        with open("temp_video.mp4", "wb") as f:
            f.write(tfile.getbuffer())

        if st.session_state.running:
            process_video("temp_video.mp4")
        else:
            st.info("▶️ Click Start Detection to begin video processing.")
            # Show first frame preview
            cap = cv2.VideoCapture("temp_video.mp4")
            ret, frame = cap.read()
            if ret:
                frame_placeholder.image(frame, channels="BGR")
            cap.release()

    elif st.session_state.file_type == "image":
        if st.session_state.running:
            process_image(st.session_state.uploaded_file)
        else:
            st.info("▶️ Click Start Detection to analyze image.")
            # Preview uploaded image
            image_bytes = st.session_state.uploaded_file
            image = np.frombuffer(image_bytes.read(), np.uint8)
            frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
            frame_placeholder.image(frame, channels="BGR")

else:
    st.info("Upload a video or image file from the sidebar to begin.")