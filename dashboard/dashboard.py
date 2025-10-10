import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from plyer import notification
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import Config
from src.violation_logger import ViolationLogger
from gui_utils import gui_process
from email_utils import EmailNotifier
from mediapipe.python.solutions import pose as mp_pose
import multiprocessing as mp
import queue
import logging
import sqlite3
import struct
import smtplib
from report_utils import generate_violation_report

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Safety EYE Dashboard", layout="wide")

# Initialize session state for email gating and new features
if 'email_validated' not in st.session_state:
    st.session_state.email_validated = False
if 'show_summary_after_stop' not in st.session_state:
    st.session_state.show_summary_after_stop = False
if 'force_tab' not in st.session_state:
    st.session_state.force_tab = None

# Email configuration
st.markdown("### Email Configuration")
sender_email = st.text_input("Sender Email", key="sender_email_input")
sender_password = st.text_input("Sender Password", type="password", key="sender_password_input")
recipient_email = st.text_input("Recipient Email", key="recipient_email_input")
submit_button = st.button("Submit", key="submit_email_config")

if submit_button:
    if not sender_email or not sender_password or not recipient_email:
        st.warning("Please provide sender email, sender password, and recipient email and click Submit to proceed.")
        st.stop()
    else:
        try:
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
            st.success("Email credentials validated successfully!")
            st.session_state.email_validated = True
            st.rerun()
        except smtplib.SMTPAuthenticationError:
            st.error("Invalid email or password. Please check your credentials and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Failed to validate credentials: {e}")
            st.stop()

# Gate dashboard access
if not st.session_state.email_validated:
    st.warning("Please configure and validate email credentials above to access the dashboard and start detection.")
    st.stop()

# Validate paths
Config.validate_paths()
if not os.path.exists(Config.MODEL_PATH):
    st.error(f"Model file not found at: {Config.MODEL_PATH}")
    st.stop()

# Load model
try:
    session = ort.InferenceSession(Config.MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    logger.info("Model loaded successfully")
    logger.info(f"Model input names: {[inp.name for inp in session.get_inputs()]}")
except Exception as e:
    st.error(f"Failed to load ONNX model: {e}")
    st.stop()

def preprocess_frame(frame):
    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            logger.error("Invalid frame dimensions")
            return None
        scale = min(Config.INPUT_SIZE[0] / w, Config.INPUT_SIZE[1] / h)
        nw, nh = int(w * scale), int(h * scale)
        img = np.full((Config.INPUT_SIZE[1], Config.INPUT_SIZE[0], 3), 114, dtype=np.uint8)  # Gray padding
        resized = cv2.resize(frame, (nw, nh))
        y_offset = (Config.INPUT_SIZE[1] - nh) // 2
        x_offset = (Config.INPUT_SIZE[0] - nw) // 2
        img[y_offset:y_offset + nh, x_offset:x_offset + nw] = resized
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logger.error(f"Error preprocessing frame: {e}")
        return None
    
def safe_get_metadata_value(metadata, key, default=0):
    """
    Safely extracts a value from metadata (handles str, dict, None).
    """
    if pd.isna(metadata) or metadata is None:
        return default
    if isinstance(metadata, dict):
        return metadata.get(key, default)
    if isinstance(metadata, str):
        try:
            # Safely parse string like "{'total_persons': 1}" to dict
            parsed = ast.literal_eval(metadata)
            if isinstance(parsed, dict):
                return parsed.get(key, default)
        except (ValueError, SyntaxError, TypeError):
            pass  # Fall back to default if parsing fails
    return default
def postprocess_output(output, frame_shape):
    try:
        predictions = output[0]
        logger.info(f"Predictions shape: {predictions.shape}")

        # Add debug for raw outputs
        logger.info(f"Raw predictions min/max: {np.min(predictions)} / {np.max(predictions)}")
        if predictions.size > 0:
            if len(predictions.shape) == 3:
                sample_box = predictions[0, 0, :4]
                sample_probs = predictions[0, 0, 4:]
            else:
                sample_box = predictions[0, :4]
                sample_probs = predictions[0, 4:]
            logger.info(f"Sample box: {sample_box}, Sample class probs: {sample_probs}")
            max_class_idx = np.argmax(sample_probs)
            logger.info(f"Sample max class: {Config.DETECTION_CLASSES[max_class_idx]} with prob {sample_probs[max_class_idx]}")

        if len(predictions.shape) == 3:
            if predictions.shape[2] == 84 and predictions.shape[1] == 8400:
                boxes = predictions[0, :, :4]
                scores = predictions[0, :, 4:4 + Config.TOTAL_CLASSES]
                scores = np.max(scores, axis=1)
                class_ids = np.argmax(scores, axis=1)
                predictions = predictions[0]
            elif predictions.shape[1] == 6 or predictions.shape[2] == 6:
                predictions = predictions.transpose(0, 2, 1) if predictions.shape[1] == 6 else predictions
                boxes = predictions[:, :, :4]
                scores = predictions[:, :, 4]
                class_ids = predictions[:, :, 5].astype(int)
            else:
                predictions = predictions.transpose(0, 2, 1) if predictions.shape[1] == 4 + Config.TOTAL_CLASSES else predictions
                boxes = predictions[:, :, :4]
                scores = predictions[:, :, 4:4 + Config.TOTAL_CLASSES]
                scores = np.max(scores, axis=2)
                class_ids = np.argmax(predictions[:, :, 4:], axis=2)
        else:
            raise ValueError("Unexpected predictions shape")

        mask = scores > Config.CONFIDENCE_THRESHOLD
        logger.info(f"Detections above threshold: {np.sum(mask)}")
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        # Scale boxes back to original frame size (center form)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * frame_shape[1] / Config.INPUT_SIZE[1]
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * frame_shape[0] / Config.INPUT_SIZE[0]
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # Adjust for letterboxing padding
        scale = min(Config.INPUT_SIZE[0] / frame_shape[1], Config.INPUT_SIZE[1] / frame_shape[0])
        x_offset = (Config.INPUT_SIZE[0] - frame_shape[1] * scale) / 2
        y_offset = (Config.INPUT_SIZE[1] - frame_shape[0] * scale) / 2
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x_offset) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y_offset) / scale

        # Clip to frame bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_shape[0])

        class_ids = np.clip(class_ids, 0, Config.TOTAL_CLASSES - 1)
        return boxes, scores, class_ids
    except Exception as e:
        logger.error(f"Error in post-processing: {e}")
        return np.array([]), np.array([]), np.array([])
def draw_overlays(frame, boxes, scores, class_ids):
    violations = []
    person_violations = {}
    person_ids = {}
    total_persons = 0
    logger.info(f"Processing {len(boxes)} detections")  # Debug

    for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        x1, y1, x2, y2 = map(int, box)
        label = f"{Config.DETECTION_CLASSES[class_id]}: {score:.2f}"
        severity = Config.SEVERITY_LEVELS.get(Config.DETECTION_CLASSES[class_id], 'None')
        color = {'High': (255, 0, 0), 'Medium': (255, 255, 0), 'Low': (0, 0, 255), 'None': (0, 255, 0)}[severity]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({severity})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cls_name = Config.DETECTION_CLASSES[class_id]
        if cls_name == "Person":
            total_persons += 1
            box_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            if box_center not in person_ids:
                person_ids[box_center] = len(person_ids) + 1
            person_id = person_ids[box_center]
            person_violations[person_id] = {"box": box, "violations": [], "confidence": score}
            cv2.putText(frame, f"Person {person_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            logger.info(f"Person {person_id} detected at center {box_center}")  # Debug
        elif cls_name in Config.VIOLATION_CLASSES and score > Config.ALERT_THRESHOLD:
            # Associate with closest person using distance
            closest_person_id = None
            min_distance = float('inf')
            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            for pid, person_data in person_violations.items():
                px1, py1, px2, py2 = map(int, person_data["box"])
                person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                distance = ((box_center[0] - person_center[0]) ** 2 + (box_center[1] - person_center[1]) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_person_id = pid
            if closest_person_id is not None:
                person_violations[closest_person_id]["violations"].append((cls_name, score, severity, box))
                violations.append((cls_name, score, severity, box, closest_person_id))
                logger.info(f"Violation {cls_name} associated with Person {closest_person_id}, distance {min_distance:.2f}")  # Debug

    logger.info(f"Total persons: {total_persons}, Violations: {len(violations)}")  # Debug
    return frame, violations, total_persons, person_violations

class PostureAnalyzer:
    def __init__(self):
        self.mp_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def analyze_posture(self, frame, boxes, class_ids):
        posture_violations = []
        person_id = 0
        for box, class_id in zip(boxes, class_ids):
            if Config.DETECTION_CLASSES[class_id] == "Person":
                person_id += 1
                x1, y1, x2, y2 = map(int, box)
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue
                results = self.mp_pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y > landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 0.2:
                        posture_violations.append(("Unsafe Posture: Bending", 0.9, "Medium", box, person_id))
        return posture_violations
def process_df(df):
    def decode(x):
        if isinstance(x, bytes):
            try:
                return x.decode('utf-8')
            except UnicodeDecodeError:
                return x.decode('latin-1')
        return x

    # Ensure basic columns exist upfront
    required_cols = ['confidence', 'person_id', 'frame']  # Removed 'Person' to avoid merge conflict
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if col == 'confidence' else 0  # 'frame' gets 0 if missing (fallback)

    for col in df.columns:
        if df[col].dtype == 'object' and col != 'confidence':
            df[col] = df[col].apply(decode)

    if 'confidence' in df.columns:
        # More robust confidence conversion with try-except to handle non-numeric cases and keep 0 values
        def to_float(x):
            if isinstance(x, bytes) and len(x) == 4:
                return struct.unpack('<f', x)[0]
            try:
                return float(x)
            except (ValueError, TypeError):
                return np.nan
        df['confidence'] = df['confidence'].apply(to_float)

    # IMPROVED: Robust person count extraction (use metadata or fuzzy match)
    if not df.empty:
        # Ensure 'frame' is numeric for merge
        df['frame'] = pd.to_numeric(df['frame'], errors='coerce').fillna(0).astype(int)
        
        # Find person rows: fuzzy match on violation OR metadata has total_persons > 0
        person_mask = (
            df['violation'].str.contains('Person', case=False, na=False) |
            (df['metadata'].apply(lambda m: safe_get_metadata_value(m, 'total_persons', 0) > 0))
        )
        person_rows = df[person_mask].copy()
        if not person_rows.empty:
            # Ensure 'frame' in person_rows too
            person_rows['frame'] = pd.to_numeric(person_rows['frame'], errors='coerce').fillna(0).astype(int)
            
            # Extract total_persons from metadata, fallback to confidence
            person_counts = person_rows[['frame']].copy()
            person_counts['Person'] = person_rows.apply(
                lambda row: safe_get_metadata_value(row['metadata'], 'total_persons', 
                                                    int(row['confidence']) if not pd.isna(row['confidence']) else 0),
                axis=1
            ).fillna(0).astype(int)
            # If multiple person rows per frame, take max (rare, but safe)
            person_counts = person_counts.groupby('frame')['Person'].agg('max').reset_index()
        else:
            person_counts = pd.DataFrame(columns=['frame', 'Person'])
    else:
        person_counts = pd.DataFrame(columns=['frame', 'Person'])

    # Merge
    df = df.merge(person_counts[['frame', 'Person']], on='frame', how='left')
    
    # Safeguard - If merge failed (no 'Person' added), initialize it
    if 'Person' not in df.columns:
        df['Person'] = 0
    
    df['Person'] = df['Person'].replace([np.inf, -np.inf], 0).fillna(0).astype(int)

    if 'person_id' in df.columns and 'confidence' in df.columns:
        # FIXED: Set 'Person' label robustly in violations
        # Use Python string methods (safe for scalar row['violation'])
        df['violations'] = df.apply(
            lambda row: 'Person' if (
                'person' in str(row['violation'] or '').lower() or  # FIXED: Scalar-safe fuzzy match
                safe_get_metadata_value(row['metadata'], 'total_persons', 0) > 0
            ) else (f"{row['violation']} (Confidence: {row['confidence']:.2f})" if not pd.isna(row['confidence']) else ''),
            axis=1
        )
        df_grouped = df.groupby(['frame', 'person_id']).agg({
            'timestamp': 'first',
            'Person': 'first',
            'violations': lambda x: '; '.join([v for v in x if v]),
            'severity': 'first',
            'location': 'first',
            'metadata': 'first',
            'confidence': 'first'
        }).reset_index()
        df = df_grouped.rename(columns={'violations': 'violation'})

        # Clean 'violation' for graphs (extract base name)
        df['base_violation'] = df['violation'].apply(
            lambda v: v.split(' (Confidence:')[0] if ' (Confidence:' in v else v
        )
        df['violation'] = df['base_violation']  # Use base for filtering
        df.drop('base_violation', axis=1, inplace=True)
        # Safer full_violation (handle empty loc selection)
        df['full_violation'] = df['violation'].apply(
            lambda v: f"{v} (Confidence: {df[df['violation'] == v]['confidence'].iloc[0]:.2f})" 
            if v != 'Person' and not df[df['violation'] == v].empty else v
        )

    return df

def generate_violation_plots(df, frame_count):
    violation_filter = Config.VIOLATION_CLASSES + ["Unsafe Posture: Bending"]
    violation_df = df[df["violation"].isin(violation_filter)].copy()  # Now uses cleaned 'violation' (e.g., "NO-Mask")

    if df.empty or violation_df.empty:
        # Empty pie/bar
        fig_pie = go.Figure().add_annotation(text="No Violations", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_pie.update_layout(title="Violation Distribution (No Violations)")

        fig_bar = go.Figure().add_annotation(text="No Violations", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig_bar.update_layout(title="Violation Counts (No Violations)")

        # Timeline with flat line
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(x=[time.time() - frame_count/Config.FPS, time.time()], y=[0, 0], mode='lines', name='No Violations'))
        fig_timeline.update_layout(title="Violation Timeline (No Violations)", xaxis_title="Time", yaxis_title="Violation Type")

        # Person plot with flat 0
        fig_person = go.Figure()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            person_df = df[df['violation'] == 'Person'].copy()
            if not person_df.empty:
                person_df = person_df.sort_values('timestamp')
                fig_person.add_trace(go.Scatter(x=person_df['timestamp'], y=person_df['Person'], mode='lines+markers', name='Persons'))
            else:
                fig_person.add_trace(go.Scatter(x=[time.time() - 10, time.time()], y=[0, 0], mode='lines', name='No Persons'))
        else:
            fig_person.add_trace(go.Scatter(x=[time.time() - 10, time.time()], y=[0, 0], mode='lines', name='No Data'))
        fig_person.update_layout(title="Persons Detected Over Time", xaxis_title="Time", yaxis_title="Number of Persons")

        return fig_pie, fig_bar, fig_timeline, fig_person

    violation_counts = violation_df["violation"].value_counts()
    total_persons = df['Person'].sum()
    priority_counts = violation_df["severity"].value_counts()

    fig_pie = px.pie(
        names=violation_counts.index,
        values=violation_counts.values,
        title="Violation Distribution",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20))

    fig_bar = px.bar(
        x=violation_counts.index,
        y=violation_counts.values,
        labels={"x": "Violation Type", "y": "Count"},
        title="Violation Counts",
        color=violation_counts.index,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_bar.update_layout(showlegend=False, margin=dict(l=20, r=20, t=50, b=20))

    violation_df["timestamp"] = pd.to_datetime(violation_df["timestamp"], errors='coerce')
    violation_df = violation_df.dropna(subset=['timestamp'])
    fig_timeline = go.Figure()
    if not violation_df.empty:
        for violation in violation_df["violation"].unique():
            df_subset = violation_df[violation_df["violation"] == violation]
            df_subset = df_subset.sort_values('timestamp')
            fig_timeline.add_trace(go.Scatter(
                x=df_subset["timestamp"],
                y=[violation] * len(df_subset),
                mode="markers",
                name=violation,
                marker=dict(size=10, color='red'),
                text=[f"Confidence: {conf:.2f} | Time: {ts}" if not pd.isna(conf) else f"Time: {ts}" for conf, ts in zip(df_subset["confidence"], df_subset["timestamp"])],
                hoverinfo="text"
            ))
    else:
        fig_timeline.add_trace(go.Scatter(x=[time.time() - 10, time.time()], y=[0, 0], mode='lines', name='No Violations'))
    fig_timeline.update_layout(
        title="Violation Timeline (Times of Occurrences)",
        xaxis_title="Time of Violation",
        yaxis_title="Violation Type",
        showlegend=True,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    # ... (rest of function unchanged until here)

    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    person_df = df[df['violation'] == 'Person'].copy()
    person_df = person_df.dropna(subset=['timestamp'])
    if not person_df.empty:
        person_df = person_df.sort_values('timestamp')
        fig_person = px.line(person_df, x='timestamp', y='Person', title="Persons Detected Over Time",
                             labels={"Person": "Number of Persons", "timestamp": "Time"})
    else:
        # FALLBACK: Aggregate from all rows' Person column (per timestamp)
        agg_person_df = df.groupby('timestamp')['Person'].sum().reset_index().dropna(subset=['timestamp'])
        if not agg_person_df.empty:
            agg_person_df = agg_person_df.sort_values('timestamp')
            fig_person = px.line(agg_person_df, x='timestamp', y='Person', title="Persons Detected Over Time (Fallback Aggregate)",
                                 labels={"Person": "Number of Persons", "timestamp": "Time"})
        else:
            fig_person = go.Figure()
            fig_person.add_trace(go.Scatter(x=[time.time() - 10, time.time()], y=[0, 0], mode='lines', name='No Persons'))
            fig_person.update_layout(title="Persons Detected Over Time (No Data)", xaxis_title="Time", yaxis_title="Number of Persons")

    return fig_pie, fig_bar, fig_timeline, fig_person
# NEW: Function to display summary content (used in tab1 post-detection and tab2)
def display_summary_content(df, frame_count):
    if 'summary_id' not in st.session_state:
        st.session_state.summary_id = 0
    st.session_state.summary_id += 1  # Increment for each unique call
    unique_id = st.session_state.summary_id

    df = process_df(df)  # Ensure processed
    st.markdown("### 📊 Full Summary Report (Auto-Generated After Detection)")
    st.markdown(f"**Total Frames Processed**: {frame_count}")
    st.markdown(f"**Final Violation Report**\n\n{generate_violation_report(df)}")
    fig_pie, fig_bar, fig_timeline, fig_person = generate_violation_plots(df, frame_count)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_pie, use_container_width=True, key=f"summary_pie_{unique_id}")  # Unique key with counter
    col2.plotly_chart(fig_bar, use_container_width=True, key=f"summary_bar_{unique_id}")  # Unique key with counter
    st.plotly_chart(fig_timeline, use_container_width=True, key=f"summary_timeline_{unique_id}")  # Unique key with counter
    st.plotly_chart(fig_person, use_container_width=True, key=f"summary_person_{unique_id}")  # Unique key with counter
    columns = ['timestamp', 'person_id', 'full_violation', 'Person', 'severity', 'frame', 'location']  # Use 'full_violation' for display
    if 'metadata' in df.columns:
        columns.append('metadata')
    if 'confidence' in df.columns:
        columns.append('confidence')
    st.dataframe(df[columns], use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button(label="Download Full Report", data=csv, file_name="full_summary.csv", mime="text/csv", key=f"summary_download_{unique_id}")  # Unique key

# NEW: Enhanced function to display alert history in a user-friendly way
def display_alert_history(df):
    df = process_df(df)
    if df.empty:
        st.info("🛡️ No alerts in history yet. Start detection to log violations!")
        return

    st.markdown("### 🚨 Alert History Overview")
    
    # Summary Metrics (User-friendly cards)
    col1, col2, col3, col4 = st.columns(4)
    total_alerts = len(df[df['violation'] != 'Person'])
    total_persons = df['Person'].max()
    high_priority = len(df[df['severity'] == 'High'])
    medium_priority = len(df[df['severity'] == 'Medium'])
    
    with col1:
        st.metric("Total Alerts", total_alerts, delta=None)
    with col2:
        st.metric("Persons Detected", total_persons, delta=None)
    with col3:
        st.metric("High Priority", high_priority, delta=None)
    with col4:
        st.metric("Medium Priority", medium_priority, delta=None)
    
    # Violation Report Summary
    st.markdown("### 📋 Quick Violation Summary")
    st.markdown(generate_violation_report(df))
    
    # Charts for Visual Understanding
    fig_pie, fig_bar, fig_timeline, fig_person = generate_violation_plots(df, 0)
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_pie, use_container_width=True)
    col2.plotly_chart(fig_bar, use_container_width=True)
    st.plotly_chart(fig_timeline, use_container_width=True)
    st.plotly_chart(fig_person, use_container_width=True)
    
    # Searchable, Formatted Table
    st.markdown("### 🔍 Detailed Alert Log")
    search_term = st.text_input("Search alerts (e.g., 'NO-Helmet' or 'Person')", placeholder="Type to filter...")
    filtered_df = df[df["full_violation"].str.contains(search_term, case=False, na=False) | df["severity"].str.contains(search_term, case=False, na=False)] if search_term else df
    
    # Format columns for readability
    display_df = filtered_df[['timestamp', 'person_id', 'full_violation', 'severity', 'confidence', 'Person']].copy()
    display_df.columns = ['🕒 Time', '👤 Person ID', '⚠️ Alert Type', '🔴 Severity', '📊 Confidence', '👥 Persons in Frame']
    display_df['🕒 Time'] = pd.to_datetime(display_df['🕒 Time'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
    display_df['⚠️ Alert Type'] = display_df['⚠️ Alert Type'].str.replace('NO-', 'Missing ').str.replace('Hardhat', 'Hard Hat')
    display_df['🔴 Severity'] = display_df['🔴 Severity'].map({'High': '🚨 High', 'Medium': '⚠️ Medium', 'Low': 'ℹ️ Low', 'None': '✅ Safe'})
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # Download Option
    csv = filtered_df.to_csv(index=False)
    st.download_button(label="📥 Download Alert History", data=csv, file_name="alert_history.csv", mime="text/csv")

st.title("Safety EYE: Real-Time PPE Monitoring")

# Initialize session state
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = False
if 'gui_initialized' not in st.session_state:
    st.session_state.gui_initialized = False
if 'violations_log' not in st.session_state:
    st.session_state.violations_log = []
if 'gui_queue' not in st.session_state:
    st.session_state.gui_queue = mp.Queue(maxsize=50)
if 'seen_violations' not in st.session_state:
    st.session_state.seen_violations = set()
if 'video_writer' not in st.session_state:
    st.session_state.video_writer = None
if 'recording' not in st.session_state:
    st.session_state.recording = False

# Enhanced Sidebar with Expanders for Better Organization
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Detection Controls Expander
    with st.expander("🔍 Detection Settings", expanded=True):
        analysis_mode = st.radio("Analysis Mode", ["Video", "Image"], key="analysis_mode", horizontal=True)
        Config.CONFIDENCE_THRESHOLD = st.slider("Confidence Threshold", 0.1, 0.9, Config.CONFIDENCE_THRESHOLD, 0.05)
        person_confidence_threshold = st.slider("Person Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
        
        if analysis_mode == "Video":
            video_source = st.selectbox("Video Source", ["Real-time Detection", "Upload Video"])
            if video_source == "Upload Video":
                uploaded_video = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"])
                if uploaded_video:
                    video_path = os.path.join(Config.LOG_DIR, uploaded_video.name)
                    with open(video_path, "wb") as f:
                        f.write(uploaded_video.read())
                    if os.path.exists(video_path):
                        cap = cv2.VideoCapture(video_path)
                        if not cap.isOpened():
                            st.error(f"Failed to open uploaded video: {video_path}.")
                            cap.release()
                            Config.VIDEO_SOURCE = 0
                        else:
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            if fps == 0 or frame_count == 0:
                                st.error(f"Invalid video properties: FPS={fps}, Frame count={frame_count}.")
                                cap.release()
                                Config.VIDEO_SOURCE = 0
                            else:
                                Config.VIDEO_SOURCE = video_path
                                st.success(f"Video uploaded: {video_path} (FPS={fps:.2f}, Frames={frame_count})")
                                cap.release()
                    else:
                        st.error(f"Failed to save video: {video_path}")
                        Config.VIDEO_SOURCE = 0
            else:
                enable_webcam = st.checkbox("Enable Webcam (Opens Camera)", value=False, help="Uncheck to avoid camera access; uses dummy loop or uploaded video.")
                if enable_webcam:
                    Config.VIDEO_SOURCE = 0  # Webcam
                    st.warning("Webcam enabled - camera window will open.")
                else:
                    Config.VIDEO_SOURCE = None  # Dummy - no camera
                    st.info("Webcam disabled - using dummy mode (no video feed).")
        else:  # Image mode
            image_source = st.selectbox("Image Source", ["Capture Image", "Upload Image"])
            if image_source == "Upload Image":
                uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            else:
                captured_image = st.camera_input("Capture Image")
            image_file = uploaded_image if uploaded_image else captured_image
    
    # Notification Settings Expander
    with st.expander("📧 Notifications", expanded=False):
        email_enabled = st.checkbox("Enable Email Notifications", value=False)
        notification_type = st.selectbox("Notification Type", ["Real-Time", "Summary"], key="notification_type")
        enable_gui = st.checkbox("Enable Alerts GUI (Extra Window)", value=False, help="Opens a desktop window for alerts. Disable if unwanted.")
    
    # Action Buttons
    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_button = st.button("▶️ Start Detection", key="start_detection", disabled=not st.session_state.email_validated, use_container_width=True)
    with col_btn2:
        stop_button = st.button("⏹️ Stop Detection", key="stop_detection", use_container_width=True)
    
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        record_button = st.button("📹 Start Recording", key="start_recording", use_container_width=True)
    with col_rec2:
        stop_record_button = st.button("⏹️ Stop Recording", key="stop_recording", use_container_width=True)
    
    # Quick Actions
    st.markdown("---")
    if st.button("🧪 Log Test Violation", key="test_violation"):
        st.session_state.logger.log_violation("Test Violation", 0.9, "High", 1, (0, 0, 0, 0), person_id=1)
        st.rerun()
    
    if st.button("🔄 Reset Logger", key="reset_logger"):
        if "logger" in st.session_state:
            del st.session_state.logger
        st.session_state.logger = ViolationLogger()
        st.success("Logger reset.")
    
    # Navigation Buttons
    st.markdown("---")
    if st.button("📊 View Summary Reports", key="view_summary"):
        st.session_state.force_tab = 'summary'
        st.rerun()
    
    if st.button("🚨 View Alert History", key="view_alert_history"):
        st.session_state.force_tab = 'alert'
        st.rerun()
    
    # Admin Section (if enabled)
    is_admin = st.checkbox("🔧 Admin Mode", value=False)
    if is_admin:
        with st.expander("Admin Tools"):
            if st.button("🗑️ Clear Database", key="clear_db"):
                conn = sqlite3.connect(st.session_state.logger.db_path)
                conn.execute("DELETE FROM violations")
                conn.commit()
                conn.close()
                st.success("Database cleared.")
                st.rerun()
            if st.button("🔄 Recreate Database", key="recreate_db"):
                if os.path.exists(st.session_state.logger.db_path):
                    os.remove(st.session_state.logger.db_path)
                st.session_state.logger = ViolationLogger()
                st.success("Database recreated.")
                st.rerun()

if record_button and not st.session_state.recording:
    st.session_state.recording = True
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(Config.LOG_DIR, f"recording_{int(time.time())}.mp4")
    st.session_state.video_writer = cv2.VideoWriter(out_path, fourcc, Config.FPS, Config.INPUT_SIZE)
    st.success(f"Started recording to {out_path}")

if stop_record_button and st.session_state.recording:
    st.session_state.recording = False
    if st.session_state.video_writer:
        st.session_state.video_writer.release()
    st.success("Recording stopped.")

if start_button and not st.session_state.run_detection:
    st.session_state.run_detection = True
    st.session_state.gui_initialized = False
    st.session_state.violations_log = []
    st.session_state.seen_violations = set()
    st.session_state.show_summary_after_stop = False  # Reset flag
    st.session_state.force_tab = None
    st.success("Starting live detection...")

if "logger" not in st.session_state:
    st.session_state.logger = ViolationLogger()
email_notifier = EmailNotifier(sender_email, sender_password)

# Handle force_tab for summary and alert
if st.session_state.force_tab == 'summary':
    df = st.session_state.logger.get_violation_data()
    display_summary_content(df, 0)  # Reuse the new function for consistency
    if st.button("📥 Generate Project Report", key="project_report"):
        total_persons = df['Person'].max()
        violation_df = df[df["violation"].isin(Config.VIOLATION_CLASSES + ["Unsafe Posture: Bending"])]
        priority_counts = violation_df["severity"].value_counts()
        report_content = f"# Safety EYE Project Report\n\n{generate_violation_report(df)}\n\n## Analytics\n- Total Persons: {total_persons}\n- High Priority Violations: {priority_counts.get('High', 0)}\n- Medium Priority Violations: {priority_counts.get('Medium', 0)}\n- Low Priority Violations: {priority_counts.get('Low', 0)}"
        st.download_button(
            label="Download Project Report",
            data=report_content,
            file_name="project_report.md",
            mime="text/markdown"
        )
    if st.button("📹 Record Demo Video", key="demo_video"):
        st.session_state.recording = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        demo_path = os.path.join(Config.LOG_DIR, f"demo_{int(time.time())}.mp4")
        st.session_state.video_writer = cv2.VideoWriter(demo_path, fourcc, Config.FPS, Config.INPUT_SIZE)
        st.success(f"Started demo recording to {demo_path}")
    if st.button("🏠 Back to Dashboard", key="back_to_dashboard"):
        st.session_state.force_tab = None
        st.rerun()
    st.stop()

if st.session_state.force_tab == 'alert':
    df = st.session_state.logger.get_violation_data()
    display_alert_history(df)
    if st.button("🏠 Back to Dashboard", key="back_to_alert_dashboard"):
        st.session_state.force_tab = None
        st.rerun()
    st.stop()

# Main Tabs (removed Alert History tab, now sidebar-driven)
tab1, tab2 = st.tabs(["📹 Live View", "📊 Summary Reports"])

with tab1:
    if st.session_state.run_detection and analysis_mode == "Video":
        col1, col2 = st.columns([3, 1])
        video_placeholder = col1.empty()
        stats_container = col2.empty()
        if Config.VIDEO_SOURCE is None:  # Dummy mode (no webcam)
            cap = None
            st.info("Dummy mode: No video feed. Logs will still be simulated if test button used.")
            st.session_state.run_detection = False  # Exit immediately in dummy
            st.rerun()
        else:
            cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
            if not cap.isOpened():
                st.error(f"Failed to open video source: {Config.VIDEO_SOURCE}")
                st.session_state.run_detection = False
                st.stop()
        if enable_gui and not st.session_state.gui_initialized:
            gui_process_instance = mp.Process(target=gui_process, args=(st.session_state.gui_queue,), daemon=True)
            gui_process_instance.start()
            st.session_state.gui_initialized = True
            with stats_container.container():
                st.write("GUI process started")
        frame_count = 0
        start_time = time.time()
        frame_latencies = []
        process_time_limit = 1.0 / Config.FPS
        posture_analyzer = PostureAnalyzer()
        no_detection_count = 0  # Track no detections
        while st.session_state.run_detection:
            if cap is None:  # Dummy frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Video stream ended.")
                    st.session_state.run_detection = False
                    break
            if stop_button:
                st.session_state.run_detection = False
                break
            frame_count += 1
            process_start = time.time()
            input_img = preprocess_frame(frame)
            if input_img is None:
                continue
            try:
                outputs = session.run(None, {"images": input_img})
            except Exception as e:
                st.error(f"Error running model inference: {e}")
                continue
            boxes, scores, class_ids = postprocess_output(outputs, frame.shape)
            # Person filtering
            person_mask = np.array([Config.DETECTION_CLASSES[class_id] == "Person" for class_id in class_ids]) if len(class_ids) > 0 else np.array([])
            if len(class_ids) > 0 and np.any(person_mask):
                person_scores = scores[person_mask]
                person_boxes = boxes[person_mask]
                person_class_ids = class_ids[person_mask]
                non_person_mask = ~person_mask
                non_person_boxes = boxes[non_person_mask]
                non_person_scores = scores[non_person_mask]
                non_person_class_ids = class_ids[non_person_mask]
                person_mask_filtered = person_scores > person_confidence_threshold
                boxes = np.vstack([person_boxes[person_mask_filtered], non_person_boxes]) if len(non_person_boxes) > 0 else person_boxes[person_mask_filtered]
                scores = np.concatenate([person_scores[person_mask_filtered], non_person_scores]) if len(non_person_scores) > 0 else person_scores[person_mask_filtered]
                class_ids = np.concatenate([person_class_ids[person_mask_filtered], non_person_class_ids]) if len(non_person_class_ids) > 0 else person_class_ids[person_mask_filtered]
            frame, violations, total_persons, person_violations = draw_overlays(frame, boxes, scores, class_ids)
            posture_violations = posture_analyzer.analyze_posture(frame, boxes, class_ids)
            violations.extend(posture_violations)
            occupancy_counts = {cls: 0 for cls in ['Person', 'machinery', 'vehicle']}
            for class_id in class_ids:
                cls_name = Config.DETECTION_CLASSES[class_id]
                if cls_name in occupancy_counts:
                    occupancy_counts[cls_name] += 1
            # Log persons based on direct detection only (removed inference)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.logger.log_violation("Person", total_persons, "None", frame_count, (0, 0, 0, 0), person_id=0, metadata={"total_persons": total_persons})
            st.session_state.violations_log.append((timestamp, "Person", total_persons, "None", frame_count, "(0,0,0,0)", 0, {"total_persons": total_persons}))
            if total_persons == 0:
                no_detection_count += 1
                if no_detection_count > 10:  # Warn after 10 empty frames
                    st.warning("No detections in last 10 frames. Lower thresholds or check input.")
            else:
                no_detection_count = 0
            for violation_type, confidence, severity, box, person_id in violations:
                st.session_state.logger.log_violation(violation_type, confidence, severity, frame_count, box, person_id=person_id)
                st.session_state.violations_log.append((timestamp, violation_type, confidence, severity, frame_count, f"({box[0]},{box[1]},{box[2]},{box[3]})", person_id, None))
                if violation_type not in st.session_state.seen_violations:
                    item = violation_type.replace("NO-", "").replace("Hardhat", "Hard hat")
                    notification.notify(
                        title="Safety EYE Alert",
                        message=f"Person {person_id} detected not wearing {item} (Confidence: {confidence:.2f})",
                        app_name="Safety EYE",
                        timeout=10
                    )
                    st.session_state.seen_violations.add(violation_type)
                if email_enabled and recipient_email and notification_type == "Real-Time":
                    email_notifier.send_violation_email(recipient_email, violation_type, confidence, severity, frame, frame_count)
            if st.session_state.recording and st.session_state.video_writer is not None:
                st.session_state.video_writer.write(frame)
            if enable_gui and st.session_state.gui_initialized:
                try:
                    st.session_state.gui_queue.put_nowait((violations, frame))
                except queue.Full:
                    logger.warning("GUI queue full, skipping frame")
                    with stats_container.container():
                        st.warning("GUI queue full, some frames may be skipped")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, caption="Live Feed", use_container_width=True)
            # Fetch fresh df after logging
            df = st.session_state.logger.get_violation_data()
            df = process_df(df)
            # Update stats
            stats_container.empty()
            with stats_container.container():
                st.write(f"Log file: {st.session_state.logger.log_file}")
                st.write(f"Database path: {st.session_state.logger.db_path}")
                search_term = st.text_input("Search Violations", key=f"video_search_{frame_count}")
                filtered_df = df[df["violation"].str.contains(search_term, case=False, na=False)] if search_term else df
                columns = ['timestamp', 'person_id', 'full_violation', 'Person', 'severity', 'frame', 'location']  # Use 'full_violation' for display
                if 'metadata' in filtered_df.columns:
                    columns.append('metadata')
                if 'confidence' in filtered_df.columns:
                    columns.append('confidence')
                st.dataframe(filtered_df[columns], key=f"video_violation_table_{frame_count}_{int(time.time() * 1000)}")
                st.markdown(f"**Live Violation Report**\n\n{generate_violation_report(df)}")
                st.markdown(f"**Processing Status**: Currently processing Frame {frame_count}")
                st.metric("Occupancy: Persons", occupancy_counts["Person"])
                st.metric("Occupancy: Machinery", occupancy_counts["machinery"])
                st.metric("Occupancy: Vehicles", occupancy_counts["vehicle"])
                elapsed_time = time.time() - process_start
                frame_latencies.append(elapsed_time * 1000)
                fps = frame_count / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                st.metric("FPS", f"{fps:.2f}")
                st.metric("Frame Latency (ms)", f"{frame_latencies[-1]:.2f}")
                queued, sent, failed = email_notifier.get_queue_status()
                st.metric("Emails Queued", queued)
                st.metric("Emails Sent", sent)
                st.metric("Emails Failed", failed)
                fig_pie, fig_bar, fig_timeline, fig_person = generate_violation_plots(df, frame_count)
                col1, col2 = st.columns(2)
                col1.plotly_chart(fig_pie, use_container_width=True, key=f"video_pie_{frame_count}_{int(time.time() * 1000)}")
                col2.plotly_chart(fig_bar, use_container_width=True, key=f"video_bar_{frame_count}_{int(time.time() * 1000)}")
                st.plotly_chart(fig_timeline, use_container_width=True, key=f"video_timeline_{frame_count}_{int(time.time() * 1000)}")
                st.plotly_chart(fig_person, use_container_width=True, key=f"video_person_{frame_count}_{int(time.time() * 1000)}")
            # Rerun every frame for updates
            if elapsed_time > process_time_limit * 2:  # Skip if too slow
                logger.info(f"Skipping frame due to slow processing: {elapsed_time:.2f}s")
                continue
            time.sleep(max(0, process_time_limit - elapsed_time))
            st.rerun()
        if cap is not None:
            cap.release()
        if st.session_state.recording:
            st.session_state.video_writer.release()
            st.session_state.recording = False
        st.session_state.gui_initialized = False
        st.session_state.force_tab = 'summary'
        st.rerun()
        if email_enabled and recipient_email and notification_type == "Summary":  # type: ignore[reportUndefinedVariable]
            email_notifier.send_summary_email(recipient_email, df)
    elif analysis_mode == "Image" and image_file:
        st.session_state.run_detection = False
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        input_img = preprocess_frame(image)
        if input_img is not None:
            outputs = session.run(None, {"images": input_img})
            boxes, scores, class_ids = postprocess_output(outputs, image.shape)
            person_mask = np.array([Config.DETECTION_CLASSES[class_id] == "Person" for class_id in class_ids])
            if len(class_ids) > 0 and np.any(person_mask):
                person_scores = scores[person_mask]
                person_boxes = boxes[person_mask]
                person_class_ids = class_ids[person_mask]
                non_person_mask = ~person_mask
                non_person_boxes = boxes[non_person_mask]
                non_person_scores = scores[non_person_mask]
                non_person_class_ids = class_ids[non_person_mask]
                person_mask_filtered = person_scores > person_confidence_threshold
                boxes = np.vstack([person_boxes[person_mask_filtered], non_person_boxes]) if len(non_person_boxes) > 0 else person_boxes[person_mask_filtered]
                scores = np.concatenate([person_scores[person_mask_filtered], non_person_scores]) if len(non_person_scores) > 0 else person_scores[person_mask_filtered]
                class_ids = np.concatenate([person_class_ids[person_mask_filtered], non_person_class_ids]) if len(non_person_class_ids) > 0 else person_class_ids[person_mask_filtered]
            image, violations, total_persons, person_violations = draw_overlays(image, boxes, scores, class_ids)
            posture_analyzer = PostureAnalyzer()
            posture_violations = posture_analyzer.analyze_posture(image, boxes, class_ids)
            violations.extend(posture_violations)
            # Log persons based on direct detection only (removed inference)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.logger.log_violation("Person", total_persons, "None", 1, (0, 0, 0, 0), person_id=0, metadata={"total_persons": total_persons})
            st.session_state.violations_log.append((timestamp, "Person", total_persons, "None", 1, "(0,0,0,0)", 0, {"total_persons": total_persons}))
            for violation_type, confidence, severity, box, person_id in violations:
                st.session_state.logger.log_violation(violation_type, confidence, severity, 1, box, person_id=person_id)
                st.session_state.violations_log.append((timestamp, violation_type, confidence, severity, 1, f"({box[0]},{box[1]},{box[2]},{box[3]})", person_id, None))
                if violation_type not in st.session_state.seen_violations:
                    item = violation_type.replace("NO-", "").replace("Hardhat", "Hard hat")
                    notification.notify(
                        title="Safety EYE Alert",
                        message=f"Person {person_id} detected not wearing {item} (Confidence: {confidence:.2f})",
                        app_name="Safety EYE",
                        timeout=10
                    )
                    st.session_state.seen_violations.add(violation_type)
                if email_enabled and recipient_email and notification_type == "Real-Time":  # type: ignore[reportUndefinedVariable]
                        email_notifier.send_violation_email(recipient_email, violation_type, confidence, severity, 1, frame_count)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_placeholder = st.empty()
            video_placeholder.image(image_rgb, caption="Image Analysis", use_container_width=True)
            df = st.session_state.logger.get_violation_data()
            df = process_df(df)
            search_term = st.text_input("Search Violations", key="image_search")
            filtered_df = df[df["violation"].str.contains(search_term, case=False, na=False)] if search_term else df
            columns = ['timestamp', 'person_id', 'full_violation', 'Person', 'severity', 'frame', 'location']
            if 'metadata' in filtered_df.columns:
                columns.append('metadata')
            if 'confidence' in filtered_df.columns:
                columns.append('confidence')
            st.dataframe(filtered_df[columns], key="image_violation_table")
            st.markdown(f"**Violation Report**\n\n{generate_violation_report(df)}")
            if email_enabled and recipient_email and notification_type == "Summary":
                email_notifier.send_summary_email(recipient_email, df)
            fig_pie, fig_bar, fig_timeline, fig_person = generate_violation_plots(df, 1)
            col1, col2 = st.columns(2)
            col1.plotly_chart(fig_pie, use_container_width=True, key="image_pie")
            col2.plotly_chart(fig_bar, use_container_width=True, key="image_bar")
            st.plotly_chart(fig_timeline, use_container_width=True, key="image_timeline")
            st.plotly_chart(fig_person, use_container_width=True, key="image_person")

with tab2:
    df = st.session_state.logger.get_violation_data()
    display_summary_content(df, 0)  # Reuse the new function for consistency
    if st.button("📥 Generate Project Report", key="project_report_tab2"):
        total_persons = df['Person'].max()
        violation_df = df[df["violation"].isin(Config.VIOLATION_CLASSES + ["Unsafe Posture: Bending"])]
        priority_counts = violation_df["severity"].value_counts()
        report_content = f"# Safety EYE Project Report\n\n{generate_violation_report(df)}\n\n## Analytics\n- Total Persons: {total_persons}\n- High Priority Violations: {priority_counts.get('High', 0)}\n- Medium Priority Violations: {priority_counts.get('Medium', 0)}\n- Low Priority Violations: {priority_counts.get('Low', 0)}"
        st.download_button(
            label="Download Project Report",
            data=report_content,
            file_name="project_report.md",
            mime="text/markdown"
        )
    if st.button("📹 Record Demo Video", key="demo_video_tab2"):
        st.session_state.recording = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        demo_path = os.path.join(Config.LOG_DIR, f"demo_{int(time.time())}.mp4")
        st.session_state.video_writer = cv2.VideoWriter(demo_path, fourcc, Config.FPS, Config.INPUT_SIZE)
        st.success(f"Started demo recording to {demo_path}")