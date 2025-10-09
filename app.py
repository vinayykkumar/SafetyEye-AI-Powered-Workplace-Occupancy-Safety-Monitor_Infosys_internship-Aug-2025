from streamlit_autorefresh import st_autorefresh
import streamlit as st
import pandas as pd
import altair as alt
import cv2
import time
import numpy as np
import datetime
import csv
import smtplib
from email.mime.text import MIMEText
from ultralytics import YOLO
import winsound
import os

st.set_page_config(page_title="SafetyEye Dashboard", layout="wide")
st.markdown("<h1 style='text-align:center;'>🦺 SafetyEye Compliance Dashboard</h1>", unsafe_allow_html=True)

MODEL_PATH = 'yolov8s.pt'
LOG_FILE = "SafetyEye_RealTimeDetection/violation_log.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

FROM_EMAIL = 'lokeshdevarapalli92@gmail.com'
FROM_PASSWORD = 'vdhp avot tshq iywh'
TO_EMAIL = 'lokeshdevarapalli92@gmail.com'

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Violation Type", "X1", "Y1", "X2", "Y2"])

st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)
iou_threshold = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.3)
alert_sound_enabled = st.sidebar.checkbox("Enable Alert Sound", True)
email_alert_enabled = st.sidebar.checkbox("Enable Email Alerts", True)
auto_refresh_enabled = st.sidebar.checkbox("Auto Refresh Logs (5 sec)", True)

col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    export_logs = st.button("Export CSV")
with col_sb2:
    clear_logs = st.button("Clear Logs")

if export_logs:
    import shutil
    shutil.copy(LOG_FILE, "exported_violation_log.csv")
    st.sidebar.success("Exported as exported_violation_log.csv")

if clear_logs:
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Violation Type", "X1", "Y1", "X2", "Y2"])
    st.sidebar.success("Logs cleared")

if 'detecting' not in st.session_state:
    st.session_state['detecting'] = False
if 'last_email_time' not in st.session_state:
    st.session_state['last_email_time'] = 0

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("▶️ Start Detection"):
        st.session_state['detecting'] = True
with col_btn2:
    if st.button("⏹️ Stop Detection"):
        st.session_state['detecting'] = False

alert_placeholder = st.empty()
video_placeholder = st.empty()
kpi_placeholder = st.empty()

def send_email_alert(subject, body, to_email, from_email, from_password):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = from_email
        msg['To'] = to_email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, [to_email], msg.as_string())
        st.success("Email alert sent!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea/float(boxAArea+boxBArea-interArea) if (boxAArea+boxBArea-interArea) else 0

def log_violation(vtype, box):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    x1, y1, x2, y2 = map(int, box)
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, vtype, x1, y1, x2, y2])

def detect_and_annotate_frame(frame, model, conf, iou_thresh):
    results = model(frame, conf=conf, iou=iou_thresh)
    det = results[0]
    boxes = det.boxes.xyxy.cpu().numpy() if det.boxes.xyxy is not None else np.empty((0,4))
    classes = det.boxes.cls.cpu().numpy() if det.boxes.cls is not None else np.array([])
    class_names = model.names

    persons = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "person"]
    helmets = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] in ("helmet","hardhat","helmet_ok")]
    vests   = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] in ("vest","safety_vest","vest_ok")]

    violations = []
    alerts = []

    for p_box in persons:
        helmet_found = any(iou(p_box, h_box) > iou_thresh for h_box in helmets)
        vest_found   = any(iou(p_box, v_box) > iou_thresh for v_box in vests)
        if not helmet_found:
            violations.append((p_box, "Helmet Missing"))
            alerts.append("Helmet Missing detected!")
            log_violation("Helmet Missing", p_box)
        if not vest_found:
            violations.append((p_box, "Vest Missing"))
            alerts.append("Vest Missing detected!")
            log_violation("Vest Missing", p_box)

    # Draw violations (red)
    for i, (box, label) in enumerate(violations):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (213, 71, 66), 3)  # deep red
        cv2.putText(frame, label, (x1, y1-10-25*i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (213, 71, 66), 2)

    # Draw other detections (green)
    for i, box in enumerate(boxes):
        if any(np.array_equal(box, v[0]) for v in violations):
            continue
        cls = int(classes[i])
        label = class_names[cls]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 196, 142), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 196, 142), 2)

    return frame, alerts

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(MODEL_PATH)

if st.session_state['detecting']:
    model = load_model()
    cap = cv2.VideoCapture(0)
    fps_t0 = time.time()
    frames = 0
    session_violations = 0

    while st.session_state['detecting']:
        ok, frame = cap.read()
        if not ok:
            alert_placeholder.error("Camera frame not available")
            break

        frames += 1
        frame, alerts = detect_and_annotate_frame(frame, model, confidence_threshold, iou_threshold)
        session_violations += len(alerts)

        now = time.time()
        if alerts:
            if alert_sound_enabled:
                winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
            if email_alert_enabled and (now - st.session_state['last_email_time'] > 30):
                send_email_alert("SafetyEye Alert", "\n".join(alerts), TO_EMAIL, FROM_EMAIL, FROM_PASSWORD)
                st.session_state['last_email_time'] = now

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb)

        msg = "\n".join(alerts) if alerts else "No PPE violations detected"
        alert_placeholder.warning(msg) if alerts else alert_placeholder.info(msg)

        if (now - fps_t0) > 1.0:
            with kpi_placeholder.container():
                c1, c2 = st.columns(2)
                c1.metric("FPS", frames)
                c2.metric("Violations (Session)", session_violations)
            fps_t0 = now
            frames = 0

        time.sleep(0.03)

    cap.release()
else:
    alert_placeholder.info("Detection stopped")

if auto_refresh_enabled:
    st_autorefresh(interval=5000, key="datarefresh")

df = pd.read_csv(LOG_FILE)
violation_col = None
for c in df.columns:
    if "Violation" in c:
        violation_col = c
        break
if not violation_col:
    violation_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

st.subheader("Violation Logs")
recent = df.tail(50).copy()
def highlight_rows(row):
    v = str(row[violation_col])
    if "Helmet" in v:
        return ['background-color: #2a0000; color: #ffb3b3'] * len(row)
    if "Vest" in v:
        return ['background-color: #2a2600; color: #ffe58a'] * len(row)
    return [''] * len(row)

filter_choice = st.selectbox("Filter", ["All"] + sorted(df[violation_col].dropna().unique().tolist()))
if filter_choice != "All":
    recent = recent[recent[violation_col] == filter_choice]
st.dataframe(recent.style.apply(highlight_rows, axis=1), height=360)

# --------- Visual Analytics ---------
st.subheader("Analytics")
palette = {'Helmet Missing': '#E4572E', 'Vest Missing': '#F3A712'}

vc = df[violation_col].value_counts().reset_index()
vc.columns = ['Violation Type', 'Count']
if not set(vc['Violation Type']).issubset(set(palette.keys())):
    extra = [c for c in vc['Violation Type'] if c not in palette]
    dynamic = ['#5BC0EB', '#9B5DE5', '#00C2A8', '#FF66B3', '#2EC4B6']
    for i, t in enumerate(extra):
        palette[t] = dynamic[i % len(dynamic)]

bar = (
    alt.Chart(vc)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        y=alt.Y('Violation Type:N', sort='-x', title='Violation Type'),
        x=alt.X('Count:Q', title='Count'),
        color=alt.Color('Violation Type:N',
                        scale=alt.Scale(domain=list(palette.keys()),
                                        range=list(palette.values())),
                        legend=None),
        tooltip=['Violation Type:N', 'Count:Q']
    )
    .properties(height=max(120, 40 * max(1, len(vc))), title='Violations by Type')
)
labels = bar.mark_text(align='left', dx=6, color='#ffffff', fontWeight='bold').encode(text='Count:Q')
st.altair_chart(bar + labels, use_container_width=True)

total = max(1, vc['Count'].sum())
vc_pct = vc.copy()
vc_pct['Percent'] = (vc_pct['Count'] / total * 100).round(1)
comp = (
    alt.Chart(vc_pct)
    .mark_bar()
    .encode(
        x=alt.X('Percent:Q', stack=None, title='Percent of Total'),
        color=alt.Color('Violation Type:N',
                        scale=alt.Scale(domain=list(palette.keys()),
                                        range=list(palette.values())),
                        legend=alt.Legend(orient='bottom', title=None)),
        tooltip=['Violation Type:N', 'Percent:Q']
    )
    .properties(height=50, title='Violation Composition (%)')
)
st.altair_chart(comp, use_container_width=True)

# --- Robust trendline with flexible timestamp column ---
timestamp_col = None
for col in df.columns:
    if "timestamp" in col.lower():
        timestamp_col = col
        break

if timestamp_col is not None:
    df_ts = df.copy()
    df_ts[timestamp_col] = pd.to_datetime(df_ts[timestamp_col], errors='coerce')
    df_ts = df_ts.dropna(subset=[timestamp_col])
    daily = df_ts.groupby(df_ts[timestamp_col].dt.date).size().reset_index(name='Count')
    trend = (
        alt.Chart(daily)
        .mark_line(point=True, color='#6C63FF')
        .encode(
            x=alt.X(f'{timestamp_col}:T', title='Date'),
            y=alt.Y('Count:Q', title='Daily Violations'),
            tooltip=[f'{timestamp_col}:T', 'Count:Q']
        )
        .properties(height=180, title='7‑Day Trend')
    )
    st.altair_chart(trend, use_container_width=True)
else:
    st.info("No timestamp column detected. Trend chart not available.")

violations_total = len(df)
types_tracked = len(vc)
exposure = max(violations_total, 1)
compliance = max(0.0, 100.0 * (1.0 - violations_total / exposure))
k1, k2, k3 = st.columns(3)
k1.metric('Total Violations (All Time)', f'{violations_total}')
k2.metric('Types Tracked', f'{types_tracked}')
k3.metric('Estimated Compliance', f'{compliance:.1f}%')

st.caption("SafetyEye • Real-time PPE compliance monitoring")
