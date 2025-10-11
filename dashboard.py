import streamlit as st
import cv2
from ultralytics import YOLO
import os
import numpy as np
import pandas as pd
import tempfile

MODEL_PATH = "../models/trained_model.pt"
model = YOLO(MODEL_PATH)
SCREENSHOT_DIR = "violation_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

st.title("🦺 Construction Site Safety Monitor")
uploaded_video = st.file_uploader("istockphoto-1036333520-640_adpp_is (online-video-cutter.com).mp4", type=["mp4", "avi", "mov", "mkv"])

if "total_frames" not in st.session_state:
    st.session_state.total_frames = 0
if "violations_detected" not in st.session_state:
    st.session_state.violations_detected = 0
if "violation_log_list" not in st.session_state:
    st.session_state.violation_log_list = []
if "workers_processed" not in st.session_state:
    st.session_state.workers_processed = 0
if "last_worker_count" not in st.session_state:
    st.session_state.last_worker_count = 0

def process_uploaded_video(video_bytes):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_bytes)
    cap = cv2.VideoCapture(tfile.name)
    frame_no = 0
    violation_log = []
    total_viol = 0
    worker_count_total = 0
    violation_type_counts = {}  # Global cooldown dict
    MAX_SAME_VIOL = 3
    MAX_FRAMES = 12

    WORKER_LABELS = ["worker", "person", "hardhat"]
    progress = st.progress(0)
    frame_display = st.empty()
    info_box = st.empty()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = 0
    while processed_frames < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        processed_frames += 1
        results = model(frame)
        detected_classes = results[0].boxes.cls.cpu().numpy() if results[0].boxes else []
        detected_names = [model.names[int(c)] for c in detected_classes]
        person_count = sum(detected_names.count(w) for w in WORKER_LABELS)
        st.session_state.last_worker_count = person_count
        worker_count_total += person_count

        # Violation logic
        violations = []
        has_helmet = "helmet" in detected_names or "hardhat" in detected_names
        has_vest = "vest" in detected_names
        if person_count > 0:
            if detected_names.count("hardhat") < person_count:
                violations.append("Missing Helmet")
            if detected_names.count("vest") < person_count:
                violations.append("Missing Vest")
        # COOLDOWN LOGIC
        for v in violations:
            key = v
            count = violation_type_counts.get(key, 0)
            if count < MAX_SAME_VIOL:
                violation_log.append({"Frame": frame_no, "Type": v})
                violation_type_counts[key] = count + 1
                total_viol += 1

        annotated = results[0].plot()
        frame_display.image(annotated, channels="BGR", caption=f"Frame {frame_no}, Workers: {person_count}")
        progress.progress(processed_frames / MAX_FRAMES)
        info_box.write(f"Detected Workers: {person_count}, Violations: {len(violations)} in this frame")

    cap.release()
    st.session_state.total_frames = processed_frames
    st.session_state.violations_detected = total_viol
    st.session_state.violation_log_list = violation_log
    st.session_state.workers_processed = worker_count_total

if uploaded_video is not None:
    st.info("Processing your video (only first 12 frames)...")
    process_uploaded_video(uploaded_video.read())
    st.success("Processing complete.")

st.subheader("Live Statistics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Frames", st.session_state.get('total_frames', 0))
col2.metric("Violations", st.session_state.get('violations_detected', 0))
col3.metric("Workers Detected (last frame)", st.session_state.get('last_worker_count', 0))


st.subheader("Violation Log")
df = pd.DataFrame(st.session_state.get('violation_log_list', []))
if not df.empty:
    st.dataframe(df)
