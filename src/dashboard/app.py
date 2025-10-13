# streamlit_dashboard_modern.py
import streamlit as st
import cv2
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import time
import os
from datetime import datetime
import plotly.express as px
import threading
import queue
import numpy as np

# -----------------------------
# 1️⃣ App Configuration & Constants
# -----------------------------
MODEL_PATH = r"C:\Users\mkr19\Desktop\SafetyEye\src\real_time_detection\bestt.pt"
LOG_FILE = "violations_log.csv"
SNAPSHOT_DIR = "snapshots"
FRAME_QUEUE_MAXSIZE = 4  # small queue to keep latest frames

st.set_page_config(page_title="🦺 SafetyEye Dashboard", layout="wide")
st.markdown("""
    <h1 style='text-align:center; color:#FF4C4C; font-family:Arial Black;'>🦺 SafetyEye - Real-Time Worker Monitoring</h1>
""", unsafe_allow_html=True)

# Ensure directories and log exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Timestamp", "ViolationType", "Confidence", "SnapshotPath"]).to_csv(LOG_FILE, index=False)

# -----------------------------
# 2️⃣ Load YOLO Model (cached)
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(MODEL_PATH)

# -----------------------------
# 3️⃣ Session State & Data
# -----------------------------
if 'violations_df' not in st.session_state:
    try:
        st.session_state.violations_df = pd.read_csv(LOG_FILE)
    except Exception:
        st.session_state.violations_df = pd.DataFrame(columns=["Timestamp", "ViolationType", "Confidence", "SnapshotPath"])
if 'latest_snapshot' not in st.session_state:
    st.session_state.latest_snapshot = None
if 'system_running' not in st.session_state:
    st.session_state.system_running = False
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
if 'reader_thread' not in st.session_state:
    st.session_state.reader_thread = None

# -----------------------------
# 4️⃣ Sidebar / Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    video_source = st.text_input("Video Source (path or 0 for webcam)", "sample1.mp4")
    confidence_threshold = st.slider("Global Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    per_class_thresholds = {
        "helmet": st.slider("Helmet detection threshold", 0.0, 1.0, 0.5, 0.01),
        "vest": st.slider("Vest detection threshold", 0.0, 1.0, 0.6, 0.01),
        "person": st.slider("Person detection threshold", 0.0, 1.0, 0.4, 0.01),
    }
    st.markdown("---")
    if st.button("Start System"):
        st.session_state.system_running = True
    if st.button("Stop System"):
        st.session_state.system_running = False

    st.markdown("---")
    st.header("📥 Logs")
    try:
        with open(LOG_FILE, "rb") as f:
            st.download_button("Download Violation Log", f, file_name="violations_log.csv", key="log_download")
    except FileNotFoundError:
        st.warning("Log file not found.")

# -----------------------------
# 5️⃣ Top-level KPIs & Layout
# -----------------------------
tabs = st.tabs(["Live Feed", "Analytics", "Settings"])
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.metric("Total Violations", int(len(st.session_state.violations_df)))
with kpi_col2:
    st.metric("No Helmet", int(len(st.session_state.violations_df[st.session_state.violations_df['ViolationType'] == 'no_helmet'])))
with kpi_col3:
    st.metric("No Vest", int(len(st.session_state.violations_df[st.session_state.violations_df['ViolationType'] == 'no_vest'])))

# -----------------------------
# 6️⃣ Video reader thread (non-blocking)
# -----------------------------
class VideoReaderThread(threading.Thread):
    def __init__(self, source, frame_q, stop_flag):
        super().__init__(daemon=True)
        self.source = source
        self.frame_q = frame_q
        self.stop_flag = stop_flag
        self.cap = None

    def run(self):
        try:
            self.cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else self.source)
            if not self.cap.isOpened():
                st.error("Error: Could not open video source.")
                return
            while not self.stop_flag.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop_flag.set()
                    break
                # keep latest frame only
                if self.frame_q.full():
                    try:
                        _ = self.frame_q.get_nowait()
                    except Exception:
                        pass
                self.frame_q.put(frame)
                time.sleep(0.01)
        finally:
            if self.cap:
                self.cap.release()

# -----------------------------
# 7️⃣ Processing & logging utilities
# -----------------------------
def append_log_row(row: dict):
    df_row = pd.DataFrame([row])
    try:
        df_row.to_csv(LOG_FILE, mode='a', header=False, index=False)
    except Exception as e:
        st.error(f"Error writing to log file: {e}")
    st.session_state.violations_df = pd.concat([st.session_state.violations_df, df_row], ignore_index=True)

def save_snapshot_and_crop(frame_bgr: np.ndarray, box_xyxy, violation_type):
    x1, y1, x2, y2 = [int(coord) for coord in box_xyxy]
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    crop = frame_bgr[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else frame_bgr
    timestamp = datetime.now()
    filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{violation_type}.jpg"
    path = os.path.join(SNAPSHOT_DIR, filename)
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    Image.fromarray(img_rgb).save(path, quality=85)
    return path

def process_frame(frame, model, conf_thresh, per_class_thresholds):
    results = model.predict(frame, conf=conf_thresh, verbose=False)
    annotated_frame = results[0].plot()
    violations_detected = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        conf = float(box.conf[0])

        # Apply per-class thresholds
        if label == "vest" and conf < per_class_thresholds.get("vest", 0.6):
            label = "no_vest"
        elif label == "helmet" and conf < per_class_thresholds.get("helmet", 0.5):
            label = "no_helmet"
        elif label == "person" and conf < per_class_thresholds.get("person", 0.4):
            continue
        else:
            if label in ["vest", "helmet"]:
                continue

        if label in ["no_helmet", "no_vest"]:
            violations_detected.append({'type': label, 'conf': conf, 'box': box.xyxy[0]})

        print(f"Detected {label} with confidence {conf:.2f}")

    return annotated_frame, violations_detected

# -----------------------------
# 8️⃣ Live Feed Tab
# -----------------------------
with tabs[0]:
    st.subheader("Live Video Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

    stop_flag = threading.Event()

    # Start or stop reader thread
    if st.session_state.system_running and (st.session_state.reader_thread is None or not st.session_state.reader_thread.is_alive()):
        stop_flag.clear()
        st.session_state.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        reader = VideoReaderThread(video_source, st.session_state.frame_queue, stop_flag)
        reader.start()
        st.session_state.reader_thread = reader
        st.session_state._reader_stop_flag = stop_flag

    if not st.session_state.system_running and st.session_state.reader_thread and st.session_state.reader_thread.is_alive():
        st.session_state._reader_stop_flag.set()
        st.session_state.reader_thread = None

    if st.session_state.system_running:
        try:
            frame = st.session_state.frame_queue.get(timeout=1.0)
        except Exception:
            frame = None

        if frame is None:
            st.info("Waiting for frames...")
        else:
            annotated_frame, violations = process_frame(frame, model, confidence_threshold, per_class_thresholds)

            if violations:
                types_list = [v['type'] for v in violations]
                alert_placeholder.error(f"🚨 VIOLATION DETECTED: {types_list}")
                for v in violations:
                    snapshot_path = save_snapshot_and_crop(frame, v['box'], v['type'])
                    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    row = {
                        "Timestamp": ts_str,
                        "ViolationType": v['type'],
                        "Confidence": f"{v['conf']:.2f}",
                        "SnapshotPath": snapshot_path
                    }
                    append_log_row(row)
                    st.session_state.latest_snapshot = Image.open(snapshot_path)
            else:
                alert_placeholder.success("✅ No violations detected.")

            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    else:
        st.info("System is stopped. Click 'Start System' to begin monitoring.")

# -----------------------------
# 9️⃣ Analytics Tab
# -----------------------------
with tabs[1]:
    st.subheader("📊 Violation Analytics")
    if not st.session_state.violations_df.empty:
        df = st.session_state.violations_df.copy()
        if "ViolationType" in df.columns and not df['ViolationType'].isna().all():
            pie_chart = px.pie(df, names='ViolationType', title="Violation Distribution")
            st.plotly_chart(pie_chart, use_container_width=True)
        st.markdown("### Recent Violations")
        st.dataframe(df.sort_values("Timestamp", ascending=False).head(10), use_container_width=True)

        if st.session_state.latest_snapshot:
            st.image(st.session_state.latest_snapshot, caption="Latest Violation Snapshot", use_container_width=True)
    else:
        st.info("No violations logged yet.")

# -----------------------------
# 10️⃣ Settings / Info Tab
# -----------------------------
with tabs[2]:
    st.subheader("⚙️ System Settings & Information")
    st.markdown(f"""
    ### 🧠 Model Configuration
    - **Model Path:** `{MODEL_PATH}`  
    - **Confidence Threshold:** `{confidence_threshold:.2f}`  
    - **Video Source:** `{video_source}`  
    """)
    st.markdown("---")
    st.markdown(f"""
    ### 🗂️ Log & Snapshot Details
    - **Log File:** `{LOG_FILE}`  
    - **Snapshots Directory:** `{SNAPSHOT_DIR}`  
    - **Total Logs Recorded:** **{len(st.session_state.violations_df)} entries**
    """)
    st.markdown("---")
    st.markdown("""
    ### 🧩 System Controls  
    Use the **sidebar** to:
    - ▶️ Start or stop the real-time monitoring system  
    - 🎥 Change the input source (webcam / video file)  
    - 🎚️ Adjust detection confidence threshold  
    - 📥 Download the latest violation log
    """)

# -----------------------------
# Clean up thread on exit
# -----------------------------
def _cleanup():
    if st.session_state.get('reader_thread') and st.session_state.get('_reader_stop_flag'):
        st.session_state._reader_stop_flag.set()

st.write("")  # trigger Streamlit execution
