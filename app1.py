import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import time
import os
import logging
import tempfile
import altair as alt
import threading
import queue
import psutil
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from io import BytesIO
import winsound
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Set up logging
logging.basicConfig(level=logging.INFO, filename='safetyeye.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PPE rules
violation_classes = {
    "NO-Gloves": "Gloves missing",
    "NO-Goggles": "Goggles missing",
    "NO-Hardhat": "Helmet missing",
    "NO-Mask": "Mask missing",
    "NO-Safety Vest": "Safety vest missing"
}
required_ppe = ["Gloves", "Goggles", "Hardhat", "Mask", "Safety Vest"]
severity_levels = {
    "Gloves missing": "Moderate",
    "Goggles missing": "High",
    "Helmet missing": "Critical",
    "Mask missing": "Moderate",
    "Safety vest missing": "High"
}

def check_violations(detected_classes):
    violations = []
    for cls in detected_classes:
        if cls in violation_classes:
            violations.append(violation_classes[cls])
    return violations

# Database functions
DB_FILE = 'safety_dashboard.db'

def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                violation TEXT,
                severity TEXT
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                occupancy INTEGER,
                violation_count INTEGER,
                compliance_score REAL
            )
        ''')
        cur.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                fps REAL,
                cpu_usage REAL
            )
        ''')
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        st.error(f"Database initialization failed: {e}")
    finally:
        if conn is not None:
            conn.close()

def log_violation(violation):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        severity = severity_levels.get(violation, "Unknown")
        cur.execute('INSERT INTO violations (timestamp, violation, severity) VALUES (?, ?, ?)',
                    (datetime.now().isoformat(), violation, severity))
        conn.commit()
        logger.info(f"Logged violation: {violation} (Severity: {severity})")
    except Exception as e:
        logger.error(f"Failed to log violation: {e}")
    finally:
        if conn is not None:
            conn.close()

def log_stats(occupancy, violation_count, compliance_score):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('INSERT INTO stats (timestamp, occupancy, violation_count, compliance_score) VALUES (?, ?, ?, ?)',
                    (datetime.now().isoformat(), occupancy, violation_count, compliance_score))
        conn.commit()
        logger.info(f"Logged stats - Occupancy: {occupancy}, Violations: {violation_count}, Compliance: {compliance_score:.2f}%")
    except Exception as e:
        logger.error(f"Failed to log stats: {e}")
    finally:
        if conn is not None:
            conn.close()

def log_performance(fps, cpu_usage):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        cur = conn.cursor()
        cur.execute('INSERT INTO performance (timestamp, fps, cpu_usage) VALUES (?, ?, ?)',
                    (datetime.now().isoformat(), fps, cpu_usage))
        conn.commit()
        logger.info(f"Logged performance - FPS: {fps:.2f}, CPU Usage: {cpu_usage:.2f}%")
    except Exception as e:
        logger.error(f"Failed to log performance: {e}")
    finally:
        if conn is not None:
            conn.close()

def get_violations_log(limit=50, filter_violation=None):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        query = 'SELECT id, timestamp, violation, severity FROM violations'
        if filter_violation:
            query += ' WHERE violation = ?'
            df = pd.read_sql_query(query, conn, params=(filter_violation,))
        else:
            df = pd.read_sql_query(f'{query} ORDER BY timestamp DESC LIMIT {limit}', conn)
        logger.info(f"Retrieved violation logs with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve violation logs: {e}")
        return pd.DataFrame(columns=['id', 'timestamp', 'violation', 'severity'])
    finally:
        if conn is not None:
            conn.close()

def get_stats_log(limit=100):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f'SELECT * FROM stats ORDER BY timestamp DESC LIMIT {limit}', conn)
        logger.info(f"Retrieved stats logs with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve stats logs: {e}")
        return pd.DataFrame(columns=['id', 'timestamp', 'occupancy', 'violation_count', 'compliance_score'])
    finally:
        if conn is not None:
            conn.close()

def get_performance_log(limit=100):
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query(f'SELECT * FROM performance ORDER BY timestamp DESC LIMIT {limit}', conn)
        logger.info(f"Retrieved performance logs with {len(df)} rows")
        return df
    except Exception as e:
        logger.error(f"Failed to retrieve performance logs: {e}")
        return pd.DataFrame(columns=['id', 'timestamp', 'fps', 'cpu_usage'])
    finally:
        if conn is not None:
            conn.close()

def generate_pdf_report(logs_df):
    pdf_file = BytesIO()
    c = canvas.Canvas(pdf_file, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "SafetyEye PPE Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y = height - 100
    if logs_df.empty or 'severity' not in logs_df.columns:
        c.drawString(50, y, "No violations recorded.")
        logger.info("Generated PDF with no violations message")
    else:
        for index, row in logs_df.iterrows():
            c.drawString(50, y, f"{row['timestamp']} - {row['violation']} (Severity: {row['severity']})")
            y -= 20
            if y < 50:
                c.showPage()
                y = height - 50
    c.save()
    pdf_file.seek(0)
    return pdf_file

def send_email_alert(violation, severity):
    sender_email = st.session_state.get('email_sender', 'your_email@example.com')
    receiver_email = st.session_state.get('email_receiver', 'receiver@example.com')
    password = st.session_state.get('email_password', 'your_password')
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = f"Safety Violation Alert - {severity}"
    body = f"A safety violation has been detected: {violation} (Severity: {severity})"
    message.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        logger.info(f"Email alert sent for {violation}")
    except Exception as e:
        logger.error(f"Failed to send email: {e}")

def update_charts(filter_violation=None):
    logs_df = get_violations_log(filter_violation=filter_violation)
    if not logs_df.empty:
        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        logs_placeholder.dataframe(logs_df[['timestamp', 'violation', 'severity']].style.set_table_styles([
            {'selector': 'th', 'props': [('background-color', '#4CAF50'), ('color', 'white'), ('font-weight', 'bold')]},
            {'selector': 'tr:hover', 'props': [('background-color', '#f0f0f0')]}
        ]))
        violation_counts = logs_df['violation'].value_counts().reset_index()
        violation_counts.columns = ['Violation Type', 'Count']
        bar_chart = alt.Chart(violation_counts).mark_bar().encode(
            x=alt.X('Violation Type', title='Violation Type'),
            y=alt.Y('Count', title='Count'),
            color=alt.Color('Violation Type', scale=alt.Scale(scheme='category10')),
            tooltip=['Violation Type', 'Count']
        ).properties(title='Violation Counts')
        violation_bar_placeholder.altair_chart(bar_chart, use_container_width=True)
        pie_chart = alt.Chart(violation_counts).mark_arc().encode(
            theta=alt.Theta(field='Count', type='quantitative'),
            color=alt.Color(field='Violation Type', type='nominal', scale=alt.Scale(scheme='category10')),
            tooltip=['Violation Type', 'Count']
        ).properties(title='Violation Types Distribution')
        violation_pie_placeholder.altair_chart(pie_chart, use_container_width=True)
        logs_df['date'] = pd.to_datetime(logs_df['timestamp']).dt.date
        trend_data = logs_df.groupby(['date', 'violation']).size().unstack(fill_value=0).reset_index()
        violation_trend_placeholder.bar_chart(trend_data.set_index('date'), color='#4CAF50')
        severity_counts = logs_df['severity'].value_counts().reset_index()
        severity_counts.columns = ['Severity', 'Count']
        severity_bar_chart = alt.Chart(severity_counts).mark_bar().encode(
            x=alt.X('Severity', title='Severity Level'),
            y=alt.Y('Count', title='Count'),
            color=alt.Color('Severity', scale=alt.Scale(scheme='set2')),
            tooltip=['Severity', 'Count']
        ).properties(title='Severity Distribution')
        severity_bar_placeholder.altair_chart(severity_bar_chart, use_container_width=True)
    stats_df = get_stats_log()
    if not stats_df.empty:
        stats_df['timestamp'] = pd.to_datetime(stats_df['timestamp'])
        occupancy_chart = alt.Chart(stats_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('occupancy:Q', title='Occupancy'),
            color=alt.value('#4CAF50'),
            tooltip=['timestamp', 'occupancy']
        ).properties(title='Occupancy Over Time')
        occupancy_line_placeholder.altair_chart(occupancy_chart, use_container_width=True)
        violation_line_chart = alt.Chart(stats_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('violation_count:Q', title='Violation Count'),
            color=alt.value('#FF0000'),
            tooltip=['timestamp', 'violation_count']
        ).properties(title='Violations Over Time')
        violation_line_placeholder.altair_chart(violation_line_chart, use_container_width=True)
        compliance_line_chart = alt.Chart(stats_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('compliance_score:Q', title='Compliance Score (%)'),
            color=alt.value('#2196F3'),
            tooltip=['timestamp', 'compliance_score']
        ).properties(title='Compliance Score Over Time')
        compliance_line_placeholder.altair_chart(compliance_line_chart, use_container_width=True)
        total_violations = stats_df['violation_count'].sum()
        average_occupancy = stats_df['occupancy'].mean()
        average_compliance = stats_df['compliance_score'].mean()
        critical_violations = len(logs_df[logs_df['severity'] == 'Critical']) if 'severity' in logs_df.columns else 0
        stats_col1.metric('Total Violations', total_violations)
        stats_col1.metric('Average Occupancy', round(average_occupancy, 2))
        stats_col2.metric('Compliance Score', f"{round(average_compliance, 2)}%")
        stats_col3.metric('Critical Violations', critical_violations)
    perf_df = get_performance_log()
    if not perf_df.empty:
        perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
        fps_line_chart = alt.Chart(perf_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('fps:Q', title='FPS'),
            color=alt.value('#FFC107'),
            tooltip=['timestamp', 'fps']
        ).properties(title='FPS Over Time')
        perf_fps_placeholder.altair_chart(fps_line_chart, use_container_width=True)
        cpu_line_chart = alt.Chart(perf_df).mark_line().encode(
            x=alt.X('timestamp:T', title='Time'),
            y=alt.Y('cpu_usage:Q', title='CPU Usage (%)'),
            color=alt.value('#9C27B0'),
            tooltip=['timestamp', 'cpu_usage']
        ).properties(title='CPU Usage Over Time')
        perf_cpu_placeholder.altair_chart(cpu_line_chart, use_container_width=True)

# Initialize DB
if not os.path.exists(DB_FILE):
    init_db()

# Streamlit App
st.set_page_config(page_title="SafetyEye Dashboard", layout="wide", initial_sidebar_state="expanded")
st.markdown('<p class="big-font">🏭 SafetyEye Dashboard</p>', unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size: 48px !important;
        font-weight: bold;
        color: #4CAF50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    .medium-font {
        font-size: 28px !important;
        font-weight: bold;
        color: #333;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        background-color: #f9f9f9;
    }
    .stDataFrame th {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    .stDataFrame tr:hover {
        background-color: #e0e0e0 !important;
    }
    .status-active {
        background-color: #e6ffed;
        border: 1px solid #4CAF50;
        padding: 10px;
        border-radius: 5px;
        color: #2e7d32;
    }
    .status-inactive {
        background-color: #ffe6e6;
        border: 1px solid #ff4d4d;
        padding: 10px;
        border-radius: 5px;
        color: #d32f2f;
    }
    .alert-box {
        background-color: #ff4d4d;
        color: white;
        padding: 10px;
        border-radius: 5px;
        animation: flash 0.5s infinite;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    @keyframes flash {
        0% { opacity: 1; }
        50% { opacity: 0.6; }
        100% { opacity: 1; }
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        border-right: 2px solid #ddd;
    }
    .full-screen-btn {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
    }
    @media (max-width: 768px) {
        .stColumn {
            width: 100% !important;
            margin-bottom: 20px;
        }
        .big-font {
            font-size: 36px !important;
        }
        .medium-font {
            font-size: 24px !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for configurations
st.sidebar.header('⚙️ Configuration')
model_type = st.sidebar.selectbox('YOLO Model Type', ['yolov8n', 'yolov8s'], index=0)
model_path = st.sidebar.text_input('YOLO Model Path', r'C:\Users\gaikw\OneDrive\Desktop\SafetyEye\runs\detect\ppe_yolov8s\weights\best.pt')
camera_index = st.sidebar.selectbox('Camera', [0, 1, 2], index=0)
fps = st.sidebar.slider('Target FPS', 1, 30, 5)
frame_skip = st.sidebar.slider('Frame Skip', 1, 10, 6)
resolution = st.sidebar.selectbox('Resolution', ['320x240', '640x480', '1280x720'], index=0)
res_width, res_height = map(int, resolution.split('x'))
save_output = st.sidebar.checkbox('Save Annotated Video Output', value=False)
auto_adjust_fps = st.sidebar.checkbox('Auto-Adjust FPS', value=True)
enable_audio_alerts = st.sidebar.checkbox('Enable Audio Alerts', value=False)
enable_email_alerts = st.sidebar.checkbox('Enable Email Alerts', value=False)
if enable_email_alerts:
    st.session_state['email_sender'] = st.sidebar.text_input('Sender Email', 'your_email@example.com')
    st.session_state['email_receiver'] = st.sidebar.text_input('Receiver Email', 'receiver@example.com')
    st.session_state['email_password'] = st.sidebar.text_input('Email Password', type='password')

# File uploader
st.sidebar.header('📷 Upload Image or Video')
uploaded_file = st.sidebar.file_uploader('Upload an image (JPG/PNG) or video (MP4)', type=['jpg', 'jpeg', 'png', 'mp4'])

# Load model
try:
    if not os.path.exists(model_path):
        st.error(f'Model file not found at: {model_path}')
        logger.error(f"Model file not found at: {model_path}")
        st.stop()
    model = YOLO(model_path)
    logger.info(f"Loaded YOLO model {model_type} from {model_path}")
except Exception as e:
    st.error(f'Error loading model: {e}')
    logger.error(f"Error loading model: {e}")
    st.stop()

# Session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'paused' not in st.session_state:
    st.session_state.paused = False
if 'cap' not in st.session_state:
    st.session_state.cap = None
if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = time.time()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'video_file_cap' not in st.session_state:
    st.session_state.video_file_cap = None
if 'processing_video_file' not in st.session_state:
    st.session_state.processing_video_file = False
if 'processing_image' not in st.session_state:
    st.session_state.processing_image = False
if 'annotated_image' not in st.session_state:
    st.session_state.annotated_image = None
if 'image_caption' not in st.session_state:
    st.session_state.image_caption = None
if 'last_chart_update' not in st.session_state:
    st.session_state.last_chart_update = time.time()
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=3)
if 'results_queue' not in st.session_state:
    st.session_state.results_queue = queue.Queue(maxsize=3)
if 'output_writer' not in st.session_state:
    st.session_state.output_writer = None
if 'start_time' not in st.session_state:
    st.session_state.start_time = time.time()
if 'frame_count_total' not in st.session_state:
    st.session_state.frame_count_total = 0
if 'full_screen' not in st.session_state:
    st.session_state.full_screen = False

# Thread for YOLO processing
def yolo_processor(frame_queue, results_queue, model, res_width, res_height):
    while True:
        try:
            frame = frame_queue.get(timeout=1)
            if frame is None:
                break
            frame = cv2.resize(frame, (res_width, res_height), interpolation=cv2.INTER_AREA)
            results = model(frame, verbose=False, imgsz=res_width, conf=0.01)
            results_queue.put((frame, results))
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"YOLO processing error: {e}")

# Start YOLO thread
if 'yolo_thread' not in st.session_state or not st.session_state.get('yolo_thread', None) or not st.session_state.yolo_thread.is_alive():
    st.session_state.yolo_thread = threading.Thread(target=yolo_processor, args=(st.session_state.frame_queue, st.session_state.results_queue, model, res_width, res_height), daemon=True)
    st.session_state.yolo_thread.start()

# Initialize placeholders
video_col = st.empty()
image_col = st.empty()
stats_col1, stats_col2, stats_col3 = st.columns(3)
occupancy_placeholder = stats_col1.metric('Current Occupancy', 0)
violations_placeholder = stats_col2.empty()
alert_placeholder = stats_col3.empty()
fps_placeholder = stats_col1.metric('Current FPS', 0)
cpu_placeholder = stats_col2.metric('CPU Usage (%)', 0)
compliance_placeholder = stats_col3.metric('Compliance Score', '0%')
filter_violation = None
logs_placeholder = st.empty()
violation_bar_placeholder = st.empty()
violation_pie_placeholder = st.empty()
severity_bar_placeholder = st.empty()
occupancy_line_placeholder = st.empty()
violation_line_placeholder = st.empty()
compliance_line_placeholder = st.empty()
violation_trend_placeholder = st.empty()
perf_fps_placeholder = st.empty()
perf_cpu_placeholder = st.empty()

# Main content
if not st.session_state.full_screen:
    st.markdown('<p class="medium-font">Real-Time Stats</p>', unsafe_allow_html=True)
    st.markdown('<p class="medium-font">Violation Logs</p>', unsafe_allow_html=True)
    filter_violation = st.selectbox('Filter by Violation', ['All'] + list(violation_classes.values()))
    filter_violation = None if filter_violation == 'All' else filter_violation
    st.markdown('<p class="medium-font">Compliance Statistics</p>', unsafe_allow_html=True)
    logs_df_export = get_violations_log(limit=1000)
    if not logs_df_export.empty and 'severity' in logs_df_export.columns:
        col_export1, col_export2 = st.columns(2)
        csv = logs_df_export.to_csv(index=False)
        col_export1.download_button(
            label="Download Violation Logs as CSV",
            data=csv,
            file_name='violation_logs.csv',
            mime='text/csv'
        )
        pdf_file = generate_pdf_report(logs_df_export)
        col_export2.download_button(
            label="Download PDF Report",
            data=pdf_file,
            file_name='violation_report.pdf',
            mime='application/pdf'
        )

# Full-screen toggle
if st.button('🖥️ Toggle Full Screen', key='full_screen_btn'):
    st.session_state.full_screen = not st.session_state.full_screen
    st.rerun()

# Buttons to control webcam
col1, col2, col3 = st.sidebar.columns(3)
if col1.button('▶️ Start Webcam'):
    if st.session_state.processing_video_file or st.session_state.processing_image:
        st.error("Cannot start webcam while processing a video file or image.")
        logger.error("Attempted to start webcam while processing video or image")
    else:
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.processing_image = False
        if st.session_state.cap is None:
            try:
                st.session_state.cap = cv2.VideoCapture(camera_index)
                if not st.session_state.cap.isOpened():
                    st.error('Unable to open webcam.')
                    logger.error(f"Failed to open webcam {camera_index}")
                    st.session_state.running = False
                else:
                    logger.info(f"Webcam {camera_index} opened successfully")
                    if save_output:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
                        st.session_state.output_writer = cv2.VideoWriter('output_webcam.mp4', fourcc, fps, (res_width, res_height))
            except Exception as e:
                st.error(f'Error opening webcam: {e}')
                logger.error(f"Error opening webcam: {e}")
                st.session_state.running = False

if col2.button('⏹️ Stop Webcam'):
    st.session_state.running = False
    st.session_state.paused = False
    if st.session_state.cap is not None:
        st.session_state.cap.release()
        st.session_state.cap = None
        if st.session_state.output_writer is not None:
            st.session_state.output_writer.release()
            st.session_state.output_writer = None
            if os.path.exists('output_webcam.mp4'):
                st.download_button(
                    label="Download Annotated Webcam Video",
                    data=open('output_webcam.mp4', 'rb').read(),
                    file_name='output_webcam.mp4',
                    mime='video/mp4'
                )
        logger.info("Webcam stopped")

if col3.button('⏸️ Pause/Resume'):
    if st.session_state.running:
        st.session_state.paused = not st.session_state.paused
        logger.info(f"Webcam {'paused' if st.session_state.paused else 'resumed'}")

# Status display
status_placeholder = st.sidebar.empty()
if st.session_state.running:
    status_text = "✅ Webcam Active" if not st.session_state.paused else "⏸️ Webcam Paused"
    status_placeholder.markdown(f'<div class="status-box status-active">{status_text}</div>', unsafe_allow_html=True)
elif st.session_state.processing_video_file:
    status_text = "✅ Processing Video File" if not st.session_state.paused else "⏸️ Video Paused"
    status_placeholder.markdown(f'<div class="status-box status-active">{status_text}</div>', unsafe_allow_html=True)
elif st.session_state.processing_image:
    status_placeholder.markdown('<div class="status-box status-active">✅ Processing Image</div>', unsafe_allow_html=True)
else:
    status_placeholder.markdown('<div class="status-box status-inactive">⛔ No Active Input</div>', unsafe_allow_html=True)

# Process uploaded file
if uploaded_file is not None:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension in ['jpg', 'jpeg', 'png']:
        if st.session_state.running or st.session_state.processing_video_file:
            st.error("Cannot process image while webcam or video is active.")
            logger.error("Attempted to process image while webcam or video is active")
        else:
            try:
                st.session_state.processing_image = True
                logger.info(f"Processing uploaded image: {uploaded_file.name}")
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if frame is None:
                    st.error('Failed to decode uploaded image.')
                    logger.error("Failed to decode uploaded image")
                    st.session_state.processing_image = False
                else:
                    frame = cv2.resize(frame, (res_width, res_height), interpolation=cv2.INTER_AREA)
                    results = model(frame, verbose=False, imgsz=res_width, conf=0.01)
                    detected_classes = []
                    persons = 0
                    for res in results:
                        if res.boxes is not None and res.boxes.cls is not None:
                            for cls in res.boxes.cls:
                                class_name = model.names[int(cls)]
                                detected_classes.append(class_name)
                                if class_name.lower() == 'person':
                                    persons += 1
                    violations = check_violations(detected_classes)
                    violation_count = len(violations)
                    try:
                        compliance_score = (1 - violation_count / max(1, persons * len(required_ppe))) * 100 if persons > 0 else 100
                    except Exception as e:
                        logger.error(f"Error calculating compliance_score: {e}")
                        compliance_score = 0
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    st.session_state.annotated_image = annotated_frame_rgb
                    st.session_state.image_caption = f'Annotated Image: {uploaded_file.name}'
                    image_col.image(st.session_state.annotated_image, channels='RGB', caption=st.session_state.image_caption)
                    if not st.session_state.full_screen:
                        occupancy_placeholder.metric('Current Occupancy', persons)
                        violations_text = ", ".join(violations) if violations else "None"
                        violations_placeholder.markdown(f"**Current Violations**: {violations_text}")
                        compliance_placeholder.metric('Compliance Score', f"{round(compliance_score, 2)}%")
                        if violation_count > 0:
                            alert_placeholder.markdown('<div class="alert-box">⚠️ Violation Detected!</div>', unsafe_allow_html=True)
                            if enable_audio_alerts:
                                winsound.Beep(1000, 500)
                            if enable_email_alerts:
                                for v in violations:
                                    send_email_alert(v, severity_levels.get(v, "Unknown"))
                        else:
                            alert_placeholder.empty()
                    for v in violations:
                        log_violation(v)
                    log_stats(persons, violation_count, compliance_score)
                    update_charts(filter_violation)
            except Exception as e:
                st.error(f'Error processing uploaded image: {e}')
                logger.error(f"Error processing uploaded image: {e}")
                st.session_state.processing_image = False
    elif file_extension == 'mp4':
        if st.session_state.running or st.session_state.processing_image:
            st.error("Cannot process video file while webcam or image is active.")
            logger.error("Attempted to process video file while webcam or image is active")
        else:
            st.session_state.processing_video_file = True
            st.session_state.paused = False
            st.session_state.processing_image = False
            if st.session_state.video_file_cap is None:
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    temp_file.write(uploaded_file.read())
                    temp_file.close()
                    st.session_state.video_file_cap = cv2.VideoCapture(temp_file.name)
                    if not st.session_state.video_file_cap.isOpened():
                        st.error('Unable to open video file.')
                        logger.error(f"Failed to open video file: {temp_file.name}")
                        st.session_state.processing_video_file = False
                    else:
                        logger.info(f"Opened video file: {temp_file.name}")
                        st.session_state.temp_video_file = temp_file.name
                        if save_output:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
                            st.session_state.output_writer = cv2.VideoWriter('output_video.mp4', fourcc, fps, (res_width, res_height))
                except Exception as e:
                    st.error(f'Error opening video file: {e}')
                    logger.error(f"Error opening video file: {e}")
                    st.session_state.processing_video_file = False

# Process video file
if st.session_state.processing_video_file and st.session_state.video_file_cap is not None and not st.session_state.paused:
    cap = st.session_state.video_file_cap
    if cap.isOpened():
        success, frame = cap.read()
        if success and frame is not None:
            try:
                st.session_state.frame_count += 1
                if st.session_state.frame_count % frame_skip != 0:
                    time.sleep(1 / max(fps, 1))
                    st.rerun()
                if not st.session_state.frame_queue.full():
                    st.session_state.frame_queue.put(frame)
                if not st.session_state.results_queue.empty():
                    frame, results = st.session_state.results_queue.get()
                    st.session_state.frame_count_total += 1
                    current_time = time.time()
                    elapsed_time = current_time - st.session_state.start_time
                    actual_fps = st.session_state.frame_count_total / elapsed_time if elapsed_time > 0 else 0
                    cpu_usage = psutil.cpu_percent()
                    if auto_adjust_fps:
                        if cpu_usage > 80 and fps > 3:
                            fps = max(3, fps - 1)
                            logger.info(f"Adjusted FPS to {fps} due to high CPU usage: {cpu_usage}%")
                        elif cpu_usage < 50 and fps < 10:
                            fps += 1
                            logger.info(f"Adjusted FPS to {fps} due to low CPU usage: {cpu_usage}%")
                    detected_classes = []
                    persons = 0
                    for res in results:
                        if res.boxes is not None and res.boxes.cls is not None:
                            for cls in res.boxes.cls:
                                class_name = model.names[int(cls)]
                                detected_classes.append(class_name)
                                if class_name.lower() == 'person':
                                    persons += 1
                    violations = check_violations(detected_classes)
                    violation_count = len(violations)
                    try:
                        compliance_score = (1 - violation_count / max(1, persons * len(required_ppe))) * 100 if persons > 0 else 100
                    except Exception as e:
                        logger.error(f"Error calculating compliance_score: {e}")
                        compliance_score = 0
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    if save_output and st.session_state.output_writer is not None:
                        st.session_state.output_writer.write(cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR))
                    video_col.image(annotated_frame_rgb, channels='RGB', caption=f'Processing Video with Overlays (FPS: {actual_fps:.2f})')
                    if not st.session_state.full_screen:
                        occupancy_placeholder.metric('Current Occupancy', persons)
                        violations_text = ", ".join(violations) if violations else "None"
                        violations_placeholder.markdown(f"**Current Violations**: {violations_text}")
                        fps_placeholder.metric('Current FPS', round(actual_fps, 2))
                        cpu_placeholder.metric('CPU Usage (%)', round(cpu_usage, 2))
                        compliance_placeholder.metric('Compliance Score', f"{round(compliance_score, 2)}%")
                        if violation_count > 0:
                            alert_placeholder.markdown('<div class="alert-box">⚠️ Violation Detected!</div>', unsafe_allow_html=True)
                            if enable_audio_alerts:
                                winsound.Beep(1000, 500)
                            if enable_email_alerts:
                                for v in violations:
                                    send_email_alert(v, severity_levels.get(v, "Unknown"))
                        else:
                            alert_placeholder.empty()
                    for v in violations:
                        log_violation(v)
                    if current_time - st.session_state.last_log_time >= 5:
                        log_stats(persons, violation_count, compliance_score)
                        log_performance(actual_fps, cpu_usage)
                        st.session_state.last_log_time = current_time
                    if not st.session_state.full_screen and current_time - st.session_state.last_chart_update >= 5:
                        update_charts(filter_violation)
                        st.session_state.last_chart_update = current_time
                    time.sleep(1 / max(fps, 1))
                    st.rerun()
            except Exception as e:
                st.error(f'Error processing video frame: {e}')
                logger.error(f"Error processing video frame: {e}")
                st.session_state.processing_video_file = False
        else:
            st.session_state.processing_video_file = False
            st.session_state.video_file_cap.release()
            st.session_state.video_file_cap = None
            if 'temp_video_file' in st.session_state:
                os.unlink(st.session_state.temp_video_file)
            if save_output and st.session_state.output_writer is not None:
                st.session_state.output_writer.release()
                st.session_state.output_writer = None
                if os.path.exists('output_video.mp4'):
                    st.download_button(
                        label="Download Annotated Video",
                        data=open('output_video.mp4', 'rb').read(),
                        file_name='output_video.mp4',
                        mime='video/mp4'
                    )
            logger.info("Finished processing video file")
            st.success("Video processing completed!")
            st.rerun()
    else:
        st.error('Video file capture is not open.')
        logger.error("Video file capture is not open")
        st.session_state.processing_video_file = False

# Process webcam
elif st.session_state.running and not st.session_state.paused:
    cap = st.session_state.cap
    if cap is not None and cap.isOpened():
        success, frame = cap.read()
        if success and frame is not None:
            try:
                st.session_state.frame_count += 1
                if st.session_state.frame_count % frame_skip != 0:
                    time.sleep(1 / max(fps, 1))
                    st.rerun()
                if not st.session_state.frame_queue.full():
                    st.session_state.frame_queue.put(frame)
                if not st.session_state.results_queue.empty():
                    frame, results = st.session_state.results_queue.get()
                    st.session_state.frame_count_total += 1
                    current_time = time.time()
                    elapsed_time = current_time - st.session_state.start_time
                    actual_fps = st.session_state.frame_count_total / elapsed_time if elapsed_time > 0 else 0
                    cpu_usage = psutil.cpu_percent()
                    if auto_adjust_fps:
                        if cpu_usage > 80 and fps > 3:
                            fps = max(3, fps - 1)
                            logger.info(f"Adjusted FPS to {fps} due to high CPU usage: {cpu_usage}%")
                        elif cpu_usage < 50 and fps < 10:
                            fps += 1
                            logger.info(f"Adjusted FPS to {fps} due to low CPU usage: {cpu_usage}%")
                    detected_classes = []
                    persons = 0
                    for res in results:
                        if res.boxes is not None and res.boxes.cls is not None:
                            for cls in res.boxes.cls:
                                class_name = model.names[int(cls)]
                                detected_classes.append(class_name)
                                if class_name.lower() == 'person':
                                    persons += 1
                    violations = check_violations(detected_classes)
                    violation_count = len(violations)
                    try:
                        compliance_score = (1 - violation_count / max(1, persons * len(required_ppe))) * 100 if persons > 0 else 100
                    except Exception as e:
                        logger.error(f"Error calculating compliance_score: {e}")
                        compliance_score = 0
                    annotated_frame = results[0].plot()
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    if save_output and st.session_state.output_writer is not None:
                        st.session_state.output_writer.write(cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR))
                    video_col.image(annotated_frame_rgb, channels='RGB', caption=f'Live Webcam Feed with Overlays (FPS: {actual_fps:.2f})')
                    if not st.session_state.full_screen:
                        occupancy_placeholder.metric('Current Occupancy', persons)
                        violations_text = ", ".join(violations) if violations else "None"
                        violations_placeholder.markdown(f"**Current Violations**: {violations_text}")
                        fps_placeholder.metric('Current FPS', round(actual_fps, 2))
                        cpu_placeholder.metric('CPU Usage (%)', round(cpu_usage, 2))
                        compliance_placeholder.metric('Compliance Score', f"{round(compliance_score, 2)}%")
                        if violation_count > 0:
                            alert_placeholder.markdown('<div class="alert-box">⚠️ Violation Detected!</div>', unsafe_allow_html=True)
                            if enable_audio_alerts:
                                winsound.Beep(1000, 500)
                            if enable_email_alerts:
                                for v in violations:
                                    send_email_alert(v, severity_levels.get(v, "Unknown"))
                        else:
                            alert_placeholder.empty()
                    for v in violations:
                        log_violation(v)
                    if current_time - st.session_state.last_log_time >= 5:
                        log_stats(persons, violation_count, compliance_score)
                        log_performance(actual_fps, cpu_usage)
                        st.session_state.last_log_time = current_time
                    if not st.session_state.full_screen and current_time - st.session_state.last_chart_update >= 5:
                        update_charts(filter_violation)
                        st.session_state.last_chart_update = current_time
                    time.sleep(1 / max(fps, 1))
                    st.rerun()
            except Exception as e:
                st.error(f'Error processing webcam frame: {e}')
                logger.error(f"Error processing webcam frame: {e}")
                st.session_state.running = False
        else:
            st.error('Failed to read frame from webcam.')
            logger.error("Failed to read frame from webcam")
            st.session_state.running = False

# Display placeholder
else:
    if not st.session_state.processing_image:
        video_col.image(np.zeros((res_height, res_width, 3), np.uint8), channels='RGB', caption='No Active Video Feed')
    else:
        if 'annotated_image' in st.session_state and st.session_state.annotated_image is not None:
            image_col.image(st.session_state.annotated_image, channels='RGB', caption=st.session_state.image_caption)
            # Ensure charts are updated for image
            update_charts(filter_violation)
        else:
            image_col.image(np.zeros((res_height, res_width, 3), np.uint8), channels='RGB', caption='No Active Image')

# Cleanup
if not st.session_state.running and st.session_state.cap is not None:
    st.session_state.cap.release()
    st.session_state.cap = None
    if st.session_state.output_writer is not None:
        st.session_state.output_writer.release()
        st.session_state.output_writer = None
    logger.info("Webcam released on cleanup")
if not st.session_state.processing_video_file and st.session_state.video_file_cap is not None:
    st.session_state.video_file_cap.release()
    st.session_state.video_file_cap = None
    if 'temp_video_file' in st.session_state:
        os.unlink(st.session_state.temp_video_file)
    if st.session_state.output_writer is not None:
        st.session_state.output_writer.release()
        st.session_state.output_writer = None
    logger.info("Video file capture released on cleanup")