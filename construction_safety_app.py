import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import time
from PIL import Image
import io
import requests
import json
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import queue
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Email Queue System
class EmailQueue:
    """Thread-safe email queue system for non-blocking email processing"""
    
    def __init__(self):
        self.queue = queue.Queue()
        self.worker_thread = None
        self.is_running = False
        self.sent_count = 0
        self.failed_count = 0
        self.processing = False
        self.current_email_info = None
        self.recent_logs = deque(maxlen=5)  # Keep last 5 email logs
        
    def start_worker(self):
        """Start the email worker thread"""
        if not self.is_running:
            self.is_running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
            
    def stop_worker(self):
        """Stop the email worker thread"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1)
            
    def add_email(self, email_data):
        """Add email to the queue"""
        self.queue.put(email_data)
        
    def get_status(self):
        """Get current queue status"""
        return {
            'queue_size': self.queue.qsize(),
            'sent_count': self.sent_count,
            'failed_count': self.failed_count,
            'is_processing': self.processing,
            'current_email': self.current_email_info,
            'recent_logs': list(self.recent_logs),
            'is_running': self.is_running
        }
        
    def _worker(self):
        """Email worker thread - processes emails from queue"""
        while self.is_running:
            try:
                # Get email from queue with timeout
                email_data = self.queue.get(timeout=1)
                self.processing = True
                self.current_email_info = f"Frame {email_data.get('frame_number', 'N/A')}"
                
                # Process the email
                success = self._send_email(email_data)
                
                if success:
                    self.sent_count += 1
                    self.recent_logs.appendleft(f"✅ Frame {email_data.get('frame_number', 'N/A')} - {datetime.now().strftime('%H:%M:%S')}")
                else:
                    self.failed_count += 1
                    self.recent_logs.appendleft(f"❌ Frame {email_data.get('frame_number', 'N/A')} - {datetime.now().strftime('%H:%M:%S')}")
                
                self.processing = False
                self.current_email_info = None
                self.queue.task_done()
                
            except queue.Empty:
                # No emails in queue, continue
                self.processing = False
                self.current_email_info = None
                continue
            except Exception as e:
                self.failed_count += 1
                self.recent_logs.appendleft(f"❌ Error: {str(e)[:30]}... - {datetime.now().strftime('%H:%M:%S')}")
                self.processing = False
                self.current_email_info = None
                
    def _send_email(self, email_data):
        """Send individual email"""
        try:
            if email_data['type'] == 'realtime_alert':
                success, _ = send_realtime_violation_alert_direct(
                    email_data['frame'], email_data['violations'], 
                    email_data['frame_number'], email_data['recipient_email'], 
                    email_data['sender_name'], email_data.get('timestamp')
                )
                return success
            elif email_data['type'] == 'csv_summary':
                success, _ = send_email_with_csv_smtp(
                    email_data['csv_data'], email_data['filename'],
                    email_data['recipient_email'], email_data['sender_name'],
                    email_data['analysis_type']
                )
                return success
            return False
        except Exception:
            return False

# Global email queue instance
email_queue = EmailQueue()

# Configure Streamlit page
st.set_page_config(
    page_title="Construction Site Safety Monitor",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .alert-box {
        padding: 10px;
        margin: 10px 0;
        border-radius: 5px;
        font-weight: bold;
    }
    .alert-danger {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .alert-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .alert-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Compact dashboard layout */
    .stImage {
        max-height: 55vh !important;
        object-fit: contain;
    }
    
    /* Compact metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
    }
    
    /* Reduce padding in containers */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Make alerts more compact */
    .stAlert {
        padding: 0.5rem 1rem !important;
        margin: 0.25rem 0 !important;
    }
    
    /* Compact headings */
    h3 {
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_class_colors():
    """Define colors for different classes"""
    return {
        0: (0, 255, 0),      # Hardhat - Green
        1: (255, 255, 0),    # Mask - Yellow
        2: (0, 0, 255),      # NO-Hardhat - Red
        3: (255, 0, 255),    # NO-Mask - Magenta
        4: (255, 165, 0),    # NO-Safety Vest - Orange
        5: (0, 255, 255),    # Person - Cyan
        6: (255, 255, 255),  # Safety Cone - White
        7: (0, 128, 255),    # Safety Vest - Blue
        8: (255, 140, 0),    # machinery - Dark Orange (more visible)
        9: (255, 20, 147)    # vehicle - Deep Pink (more visible)
    }


def display_dataframe(df, **kwargs):
    """Compatibility wrapper for st.dataframe across Streamlit versions.

    Tries supported kwarg names ('use_container_width' then 'use_column_width') and
    falls back to calling st.dataframe(df) without width kwargs if neither is accepted.
    """
    # Prefer use_container_width. If caller passed use_column_width (deprecated), map it.
    ucw = None
    if 'use_container_width' in kwargs:
        ucw = kwargs.get('use_container_width')
    elif 'use_column_width' in kwargs:
        # Map older param to the new one. Treat truthy values as True.
        ucw = bool(kwargs.get('use_column_width'))

    if ucw is not None:
        try:
            return st.dataframe(df, use_container_width=ucw)
        except TypeError:
            # In case the installed Streamlit doesn't accept the kwarg, fallback to no-kwarg call
            return st.dataframe(df)

    # No width preference provided — call default
    return st.dataframe(df)

def boxes_overlap(box1, box2, threshold=0.3):
    """Check if two bounding boxes overlap significantly"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection area
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = x_overlap * y_overlap
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU (Intersection over Union)
    if union_area == 0:
        return False
    
    iou = intersection_area / union_area
    return iou > threshold

def expand_box(box, expansion_factor=0.2):
    """Expand bounding box by a factor to create a larger filtering zone"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Expand box by the factor
    expand_w = width * expansion_factor
    expand_h = height * expansion_factor
    
    return (
        max(0, x1 - expand_w),
        max(0, y1 - expand_h), 
        x2 + expand_w,
        y2 + expand_h
    )

def is_box_inside(inner_box, outer_box):
    """Check if inner_box is completely inside outer_box"""
    x1_inner, y1_inner, x2_inner, y2_inner = inner_box
    x1_outer, y1_outer, x2_outer, y2_outer = outer_box
    
    return (x1_inner >= x1_outer and y1_inner >= y1_outer and 
            x2_inner <= x2_outer and y2_inner <= y2_outer)

def is_detection_too_large_for_person(box):
    """Check if detection box is too large to be a person or person-related equipment"""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    # Filter out detections that are too large (likely vehicles/machinery misclassified)
    # Typical person bounding box shouldn't exceed these thresholds
    max_width = 200  # pixels
    max_height = 400  # pixels
    max_area = 50000  # square pixels
    
    return (width > max_width or height > max_height or area > max_area)

def is_violation_near_vehicle_machinery(violation_box, vehicle_boxes, machinery_boxes):
    """Enhanced filtering: Check if violation is near vehicles/machinery using multiple aggressive methods"""
    
    # Method 1: Direct overlap with normal boxes (very low threshold)
    for vehicle_box in vehicle_boxes:
        if boxes_overlap(violation_box, vehicle_box, threshold=0.01):  # Much lower threshold
            return True
        # Method 2: Check overlap with expanded vehicle box (40% larger for more coverage)
        expanded_vehicle = expand_box(vehicle_box, 0.4)  # Increased expansion
        if boxes_overlap(violation_box, expanded_vehicle, threshold=0.01):  # Very low threshold
            return True
        # Method 3: Check if violation is completely inside expanded vehicle area
        super_expanded_vehicle = expand_box(vehicle_box, 0.6)  # Even larger expansion
        if is_box_inside(violation_box, super_expanded_vehicle):
            return True
            
    for machinery_box in machinery_boxes:
        if boxes_overlap(violation_box, machinery_box, threshold=0.01):  # Much lower threshold
            return True
        # Method 2: Check overlap with expanded machinery box (40% larger for more coverage)
        expanded_machinery = expand_box(machinery_box, 0.4)  # Increased expansion
        if boxes_overlap(violation_box, expanded_machinery, threshold=0.01):  # Very low threshold
            return True
        # Method 3: Check if violation is completely inside expanded machinery area
        super_expanded_machinery = expand_box(machinery_box, 0.6)  # Even larger expansion
        if is_box_inside(violation_box, super_expanded_machinery):
            return True
    
    return False

def analyze_safety_violations(detections, class_names, debug_mode=False):
    """Analyze detections for safety violations with enhanced filtering"""
    violations = []
    person_count = 0
    safety_equipped_count = 0
    filtered_count = 0  # Count of filtered detections
    debug_info = []  # Debug information

    # Separate detections by class
    person_boxes = []
    hardhat_boxes = []
    vest_boxes = []
    mask_boxes = []
    no_hardhat_boxes = []
    no_vest_boxes = []
    no_mask_boxes = []
    vehicle_boxes = []
    machinery_boxes = []

    for detection in detections:
        class_id = int(detection[5])
        class_name = class_names.get(class_id, f"Unknown_{class_id}")
        confidence = detection[4]

        # Collect detections above a base threshold for consideration
        if confidence > 0.4:
            x1, y1, x2, y2 = detection[:4]
            box = (x1, y1, x2, y2)

            if class_name == "Person":
                person_count += 1
                person_boxes.append(box)
            elif class_name == "vehicle":
                vehicle_boxes.append(box)
            elif class_name == "machinery":
                machinery_boxes.append(box)
            elif class_name == "Hardhat":
                hardhat_boxes.append(box)
            elif class_name == "Safety Vest":
                vest_boxes.append(box)
            elif class_name == "Mask":
                mask_boxes.append(box)
            elif class_name == "NO-Hardhat":
                no_hardhat_boxes.append((box, confidence))
            elif class_name == "NO-Safety Vest":
                no_vest_boxes.append((box, confidence))
            elif class_name == "NO-Mask":
                no_mask_boxes.append((box, confidence))

    # Helper to check IoU-like overlap; reuse boxes_overlap with a reasonable threshold
    def overlaps(a, b, thr=0.1):
        return boxes_overlap(a, b, threshold=thr)

    def box_center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def point_in_box(pt, box, expand_x=0.0, expand_y=0.0):
        x, y = pt
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        ex = expand_x * w
        ey = expand_y * h
        return (x >= x1 - ex) and (x <= x2 + ex) and (y >= y1 - ey) and (y <= y2 + ey)

    def is_ppe_on_person(person_box, ppe_box, kind):
        # Use PPE box center and spatial heuristics per kind, fallback to IoU overlap
        pcx, pcy = box_center(person_box)
        wc = person_box[2] - person_box[0]
        hc = person_box[3] - person_box[1]
        cx, cy = box_center(ppe_box)

        # Hardhat -> expect at top of person box (upper ~45%) and x inside person x-range
        if kind == 'hardhat':
            in_x = (cx >= person_box[0] - 0.05 * wc) and (cx <= person_box[2] + 0.05 * wc)
            in_y = (cy >= person_box[1] - 0.05 * hc) and (cy <= person_box[1] + 0.45 * hc)
            if in_x and in_y:
                return True

        # Vest -> expect roughly middle-lower torso (y between ~35% and 100%)
        if kind == 'vest':
            in_x = (cx >= person_box[0] - 0.1 * wc) and (cx <= person_box[2] + 0.1 * wc)
            in_y = (cy >= person_box[1] + 0.35 * hc) and (cy <= person_box[3] + 0.05 * hc)
            if in_x and in_y:
                return True

        # Mask -> expect around face region (upper ~50% but below top)
        if kind == 'mask':
            in_x = (cx >= person_box[0] - 0.15 * wc) and (cx <= person_box[2] + 0.15 * wc)
            in_y = (cy >= person_box[1] + 0.12 * hc) and (cy <= person_box[1] + 0.5 * hc)
            if in_x and in_y:
                return True

        # Fallback: IoU overlap
        return overlaps(person_box, ppe_box, thr=0.12)

    # For each detected person, check if they have safety equipment nearby and record missing items
    person_missing = []  # list of dicts per person: {'box': box, 'missing': set(...)}

    for person_box in person_boxes:
        # Skip persons in/near vehicles or machinery
        is_near_vehicle = any(boxes_overlap(person_box, vb, threshold=0.3) for vb in vehicle_boxes + machinery_boxes)
        if is_near_vehicle:
            # Do not analyze PPE for people in vehicles/machinery
            continue

        # Check for PPE on this person using spatial heuristics
        person_has_hardhat = any(is_ppe_on_person(person_box, hb, 'hardhat') for hb in hardhat_boxes)
        person_has_vest = any(is_ppe_on_person(person_box, vb, 'vest') for vb in vest_boxes)
        person_has_mask = any(is_ppe_on_person(person_box, mb, 'mask') for mb in mask_boxes)

        missing = set()
        if not person_has_hardhat:
            missing.add(('hardhat', 'High'))
        if not person_has_vest:
            missing.add(('vest', 'Medium'))
        if not person_has_mask:
            missing.add(('mask', 'Low'))

        person_missing.append({'box': person_box, 'missing': missing})

        # Count as safety equipped if they have at least hardhat and vest
        if person_has_hardhat and person_has_vest:
            safety_equipped_count += 1

    # Create person-based violation entries (one per missing PPE per person)
    for p in person_missing:
        bx = p['box']
        for item, severity in p['missing']:
            vtype = {
                'hardhat': 'Person Missing Hard Hat',
                'vest': 'Person Missing Safety Vest',
                'mask': 'Person Missing Mask'
            }[item]
            violations.append({'type': vtype, 'severity': severity, 'location': f"({int(bx[0])}, {int(bx[1])})", 'box': bx})

    # Now process NO-* detections but avoid duplicates: if a NO-* overlaps a person who already has that missing PPE, skip it.
    def add_no_violation(box, kind, severity, confidence):
        nonlocal filtered_count
        # filter by confidence
        if confidence <= 0.6:
            filtered_count += 1
            return

        # vehicle/machinery proximity and size checks
        is_near_vehicle = is_violation_near_vehicle_machinery(box, vehicle_boxes, machinery_boxes)
        is_too_large = is_detection_too_large_for_person(box)
        if is_near_vehicle or is_too_large:
            filtered_count += 1
            return

        # If overlaps a person who already has a person-based missing violation for this kind, skip
        for p in person_missing:
            if overlaps(p['box'], box, thr=0.12):
                kind_map = {'NO-Hardhat': 'hardhat', 'NO-Safety Vest': 'vest', 'NO-Mask': 'mask'}
                mapped = kind_map.get(kind, None)
                if mapped and any(mapped == miss[0] for miss in p['missing']):
                    # person-based violation already created, skip
                    return

        # Deduplicate against existing NO-* violations by spatial overlap
        for existing in violations:
            if 'box' in existing and boxes_overlap(existing['box'], box, threshold=0.5) and existing['type'].lower().find(kind.split('-')[-1].lower()) != -1:
                # similar existing violation, skip
                return

        # Add NO-* violation
        vlabel = {
            'NO-Hardhat': ('Worker Without Hard Hat Detected', 'High'),
            'NO-Safety Vest': ('Worker Without Safety Vest Detected', 'Medium'),
            'NO-Mask': ('Worker Without Mask Detected', 'Low')
        }.get(kind, (f'Worker Without {kind} Detected', 'Low'))

        violations.append({'type': vlabel[0], 'severity': vlabel[1], 'location': f"({int(box[0])}, {int(box[1])})", 'box': box})

    # Add NO-* boxes
    for box, conf in no_hardhat_boxes:
        add_no_violation(box, 'NO-Hardhat', 'High', conf)
    for box, conf in no_vest_boxes:
        add_no_violation(box, 'NO-Safety Vest', 'Medium', conf)
    for box, conf in no_mask_boxes:
        add_no_violation(box, 'NO-Mask', 'Low', conf)

    # Remove 'box' keys from violations for external use
    for v in violations:
        if 'box' in v:
            v.pop('box')

    # Always return filtered_count for developer mode
    return violations, person_count, safety_equipped_count, filtered_count

def draw_detections(frame, detections, class_names, colors, debug_mode=False):
    """Draw bounding boxes and labels on frame"""
    violations, person_count, safety_equipped, filtered_count = analyze_safety_violations(detections, class_names, debug_mode)
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)
        
        if conf > 0.4:  # Lower confidence threshold to catch more detections
            color = colors.get(class_id, (255, 255, 255))
            label = f"{class_names.get(class_id, f'Unknown_{class_id}')} {conf:.2f}"
            
            # Use different colors for safety violations
            if class_names.get(class_id, "").startswith("NO-"):
                color = (0, 0, 255)  # Red for violations
            elif class_names.get(class_id, "") in ["Person"]:
                color = (0, 255, 255)  # Cyan for persons
            elif class_names.get(class_id, "") in ["Hardhat", "Safety Vest", "Mask"]:
                color = (0, 255, 0)  # Green for safety equipment
            elif class_names.get(class_id, "") == "vehicle":
                color = (255, 20, 147)  # Deep Pink for vehicles
            elif class_names.get(class_id, "") == "machinery":
                color = (255, 140, 0)  # Dark Orange for machinery
            elif class_names.get(class_id, "") in ["Safety Cone"]:
                color = (255, 255, 255)  # White for safety cones
            
            # Determine box thickness - make vehicles and machinery more prominent
            box_thickness = 3 if class_names.get(class_id, "") in ["vehicle", "machinery"] else 2
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, box_thickness)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10),
                         (int(x1) + label_size[0], int(y1)), color, -1)
            
            # Draw label text
            cv2.putText(frame, label, (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Always return filtered_count for developer mode
    return frame, violations, person_count, safety_equipped, filtered_count

def process_video_frame(frame, model, class_names, colors, debug_mode=False):
    """Process a single video frame"""
    results = model(frame)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        detections = results[0].boxes.data.cpu().numpy()
        # Always returns: annotated_frame, violations, person_count, safety_equipped, filtered_count
        annotated_frame, violations, person_count, safety_equipped, filtered_count = draw_detections(
            frame, detections, class_names, colors, debug_mode
        )
    else:
        annotated_frame = frame
        violations = []
        person_count = 0
        safety_equipped = 0
        filtered_count = 0
    
    # Always return filtered_count for developer mode
    return annotated_frame, violations, person_count, safety_equipped, filtered_count

def update_live_alerts(placeholder, violations, current_frame):
    """Update live alerts display using Streamlit components - COMPACT VERSION"""
    if not violations:
        placeholder.success("✅ No violations")
        return
    
    # Get recent violations (last 4 for compact view)
    recent_violations = violations[-4:] if len(violations) > 4 else violations
    
    # Create container for alerts - more compact
    with placeholder.container():
        for violation in recent_violations:
            # Use compact alert format
            severity_icon = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            icon = severity_icon.get(violation['severity'], "⚪")
            
            # Shorter message format
            msg = f"{icon} **{violation['type'][:30]}** - F{violation['frame']}"
            
            if violation['severity'] == 'High':
                st.error(msg, icon="🚨")
            elif violation['severity'] == 'Medium':
                st.warning(msg, icon="⚠️")
            else:
                st.info(msg, icon="ℹ️")

def update_live_stats(placeholder, total_violations, frame_stats, current_frame, show_email_queue=False):
    """Update live statistics display using Streamlit components with optional email queue status - COMPACT VERSION"""
    if not frame_stats:
        return
    
    # Calculate current stats
    total_violations_count = len(total_violations)
    high_violations = len([v for v in total_violations if v['severity'] == 'High'])
    medium_violations = len([v for v in total_violations if v['severity'] == 'Medium'])
    low_violations = len([v for v in total_violations if v['severity'] == 'Low'])
    
    # Recent frame stats
    recent_frames = frame_stats[-10:] if len(frame_stats) >= 10 else frame_stats
    avg_persons = sum(f['persons'] for f in recent_frames) / len(recent_frames) if recent_frames else 0
    total_persons = sum(f['persons'] for f in frame_stats)
    
    # Calculate rates
    violation_rate = (total_violations_count / total_persons * 100) if total_persons > 0 else 0
    
    # Display with Streamlit components - COMPACT LAYOUT
    with placeholder.container():
        # Single row with all key metrics
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            # Alert level indicator
            if high_violations > 5:
                st.metric("Status", "🔴 HIGH")
            elif medium_violations > 3:
                st.metric("Status", "🟡 MED")
            else:
                st.metric("Status", "🟢 LOW")
        
        with col2:
            st.metric("Total", total_violations_count)
        
        with col3:
            st.metric("Frame", current_frame)
        
        with col4:
            st.metric("HIGH", high_violations)
        
        with col5:
            st.metric("MED", medium_violations)
        
        with col6:
            st.metric("LOW", low_violations)
        
        # Second row: Additional metrics (if email queue enabled, show it; otherwise show performance)
        if show_email_queue:
            queue_status = email_queue.get_status()
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📧 Queued", queue_status['queue_size'])
            with col2:
                st.metric("📧 Sent", queue_status['sent_count'])
            with col3:
                st.metric("📧 Failed", queue_status['failed_count'])
            with col4:
                if queue_status['is_processing']:
                    st.metric("📧 Status", "Sending...")
                else:
                    st.metric("📧 Status", "Idle")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Workers", f"{avg_persons:.1f}")
            with col2:
                st.metric("Total Workers", total_persons)
            with col3:
                st.metric("Violation Rate", f"{violation_rate:.1f}%")

def update_detailed_stats(placeholder, total_violations, frame_stats, current_frame):
    """Update detailed statistics with graphs and analytics in an expandable section"""
    if not frame_stats or len(frame_stats) < 2:
        return
    
    with placeholder.expander("📊 **Detailed Analytics & Graphs** (Click to expand)", expanded=False):
        # Calculate additional metrics
        total_violations_count = len(total_violations)
        high_violations = [v for v in total_violations if v['severity'] == 'High']
        medium_violations = [v for v in total_violations if v['severity'] == 'Medium']
        low_violations = [v for v in total_violations if v['severity'] == 'Low']
        
        # Violation type breakdown
        violation_types = {}
        for v in total_violations:
            vtype = v['type']
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        # Create two columns for graphs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Violations Over Time")
            # Prepare data for violations over time
            if len(frame_stats) > 1:
                frames = [f['frame'] for f in frame_stats]
                violations_per_frame = [f['violations'] for f in frame_stats]
                
                # Create simple chart data
                chart_data = pd.DataFrame({
                    'Frame': frames,
                    'Violations': violations_per_frame
                })
                st.line_chart(chart_data.set_index('Frame'), height=200)
            
            st.markdown("#### 🎯 Violation Type Distribution")
            if violation_types:
                type_df = pd.DataFrame(
                    list(violation_types.items()),
                    columns=['Type', 'Count']
                ).sort_values('Count', ascending=False).head(5)
                st.bar_chart(type_df.set_index('Type'), height=200)
        
        with col2:
            st.markdown("#### 👷 Workers Detected Over Time")
            # Prepare data for workers over time
            if len(frame_stats) > 1:
                frames = [f['frame'] for f in frame_stats]
                persons = [f['persons'] for f in frame_stats]
                
                chart_data = pd.DataFrame({
                    'Frame': frames,
                    'Workers': persons
                })
                st.area_chart(chart_data.set_index('Frame'), height=200)
            
            st.markdown("#### 🔢 Severity Distribution")
            severity_data = pd.DataFrame({
                'Severity': ['High', 'Medium', 'Low'],
                'Count': [len(high_violations), len(medium_violations), len(low_violations)]
            })
            st.bar_chart(severity_data.set_index('Severity'), height=200)
        
        # Additional statistics in a row
        st.markdown("---")
        st.markdown("#### 📋 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_violations_per_frame = total_violations_count / len(frame_stats) if frame_stats else 0
            st.metric("Avg Violations/Frame", f"{avg_violations_per_frame:.2f}")
        
        with col2:
            total_persons = sum(f['persons'] for f in frame_stats)
            total_safety_equipped = sum(f['safety_equipped'] for f in frame_stats)
            compliance_rate = (total_safety_equipped / total_persons * 100) if total_persons > 0 else 0
            st.metric("Safety Compliance", f"{compliance_rate:.1f}%")
        
        with col3:
            max_violations_frame = max(frame_stats, key=lambda x: x['violations']) if frame_stats else None
            if max_violations_frame:
                st.metric("Peak Violations Frame", f"#{max_violations_frame['frame']} ({max_violations_frame['violations']})")
        
        with col4:
            max_workers_frame = max(frame_stats, key=lambda x: x['persons']) if frame_stats else None
            if max_workers_frame:
                st.metric("Peak Workers Frame", f"#{max_workers_frame['frame']} ({max_workers_frame['persons']})")
        
        # More detailed metrics
        st.markdown("---")
        st.markdown("#### 🔍 Advanced Analytics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Risk score calculation (weighted by severity)
            risk_score = (len(high_violations) * 3 + len(medium_violations) * 2 + len(low_violations) * 1)
            max_risk = total_violations_count * 3
            risk_percentage = (risk_score / max_risk * 100) if max_risk > 0 else 0
            st.metric("Risk Score", f"{risk_percentage:.1f}%", 
                     delta=f"{risk_score}/{max_risk}" if max_risk > 0 else "0/0")
        
        with col2:
            # Violation density (violations per worker)
            violation_density = (total_violations_count / total_persons) if total_persons > 0 else 0
            st.metric("Violations/Worker", f"{violation_density:.2f}")
        
        with col3:
            # High severity rate
            high_severity_rate = (len(high_violations) / total_violations_count * 100) if total_violations_count > 0 else 0
            st.metric("High Severity Rate", f"{high_severity_rate:.1f}%")
        
        with col4:
            # Frames with violations
            frames_with_violations = len([f for f in frame_stats if f['violations'] > 0])
            violation_frame_rate = (frames_with_violations / len(frame_stats) * 100) if frame_stats else 0
            st.metric("Frames w/ Violations", f"{violation_frame_rate:.1f}%")
        
        with col5:
            # Average workers per frame
            avg_workers = sum(f['persons'] for f in frame_stats) / len(frame_stats) if frame_stats else 0
            st.metric("Avg Workers/Frame", f"{avg_workers:.1f}")
        
        # Time-based analysis
        st.markdown("---")
        st.markdown("#### ⏱️ Temporal Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # First violation frame
            first_violation = next((v for v in total_violations), None)
            if first_violation:
                st.metric("First Violation", f"Frame {first_violation['frame']}")
            else:
                st.metric("First Violation", "N/A")
        
        with col2:
            # Last violation frame
            last_violation = total_violations[-1] if total_violations else None
            if last_violation:
                st.metric("Latest Violation", f"Frame {last_violation['frame']}")
            else:
                st.metric("Latest Violation", "N/A")
        
        with col3:
            # Violation frequency (violations per 100 frames)
            violation_frequency = (total_violations_count / len(frame_stats) * 100) if frame_stats else 0
            st.metric("Violations/100 Frames", f"{violation_frequency:.1f}")
        
        # Top violation types table
        if violation_types:
            st.markdown("---")
            st.markdown("#### 🏆 Top Violation Types")
            top_violations_df = pd.DataFrame(
                list(violation_types.items()),
                columns=['Violation Type', 'Count']
            ).sort_values('Count', ascending=False).head(10)
            
            # Add percentage column
            top_violations_df['Percentage'] = (top_violations_df['Count'] / total_violations_count * 100).round(1)
            top_violations_df['Percentage'] = top_violations_df['Percentage'].astype(str) + '%'
            
            display_dataframe(top_violations_df, use_container_width=True)
        
        # Safety hotspots - frames with most issues
        st.markdown("---")
        st.markdown("#### 🔥 Safety Hotspots (Frames with Most Violations)")
        
        hotspot_frames = sorted(frame_stats, key=lambda x: x['violations'], reverse=True)[:5]
        if hotspot_frames and hotspot_frames[0]['violations'] > 0:
            hotspot_df = pd.DataFrame([
                {
                    'Frame': f['frame'],
                    'Violations': f['violations'],
                    'Workers': f['persons'],
                    'Safety Equipped': f['safety_equipped']
                } for f in hotspot_frames
            ])
            display_dataframe(hotspot_df, use_container_width=True)
        else:
            st.info("No violation hotspots detected")
        
        # Compliance trends
        st.markdown("---")
        st.markdown("#### 📉 Safety Compliance Trend")
        
        if len(frame_stats) > 1:
            # Calculate rolling compliance rate
            compliance_data = []
            for f in frame_stats:
                if f['persons'] > 0:
                    compliance = (f['safety_equipped'] / f['persons'] * 100)
                    compliance_data.append({'Frame': f['frame'], 'Compliance %': compliance})
            
            if compliance_data:
                compliance_df = pd.DataFrame(compliance_data)
                st.line_chart(compliance_df.set_index('Frame'), height=200)

def send_email_with_csv_smtp(csv_data, filename, recipient_email, sender_name, analysis_type="Video Analysis"):
    """Send email with CSV attachment using Gmail SMTP"""
    try:
        # Gmail SMTP configuration
        sender_email = "safetyeyeteam8@gmail.com"
        password = "dtwmwimtbqquqwda"  # App Password for Gmail
        
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"🚧 Construction Safety Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Email body
        email_body = f"""
Construction Safety Analysis Complete!

Analysis Details:
📊 Type: {analysis_type}
⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 File: {filename}
👤 Sent by: {sender_name or 'Safety Monitor'}

This email contains your detailed safety violation report as a CSV attachment.

The CSV includes:
• Frame-by-frame violation data
• Violation types and priorities  
• Worker locations and safety status
• Detailed coordinates and confidence scores

Best regards,
Construction Safety Monitor System 🚧
        """
        
        # Attach body to email
        msg.attach(MIMEText(email_body, "plain"))
        
        # Add CSV attachment
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(csv_data.encode())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {filename}'
        )
        msg.attach(part)
        
        # Connect to Gmail SMTP server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        return True, "✅ Email sent successfully! 📧 (Check spam folder if needed)"
        
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed. Please check Gmail App Password settings."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error occurred: {str(e)}"
    except Exception as e:
        return False, f"❌ Email sending failed: {str(e)}"

def send_email_with_html_smtp(html_content, recipient_email, sender_name, analysis_type="Video Analysis"):
    """Send email with HTML report using Gmail SMTP"""
    try:
        # Gmail SMTP configuration
        sender_email = "safetyeyeteam8@gmail.com"
        password = "dtwmwimtbqquqwda"  # App Password for Gmail
        
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"🚧 Construction Safety HTML Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Email body (text version)
        email_body = f"""
Construction Safety Analysis Complete!

Analysis Details:
📊 Type: {analysis_type}
⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
👤 Sent by: {sender_name or 'Safety Monitor'}

This email contains your interactive HTML report with:
• Visual graphs and charts
• Interactive data table with search and filter
• Violation timeline and statistics
• Safety compliance trends

Please open the HTML attachment in your web browser to view the full report.

Best regards,
Construction Safety Monitor System 🚧
        """
        
        # Attach plain text body
        msg.attach(MIMEText(email_body, "plain"))
        
        # Add HTML attachment
        html_part = MIMEBase('application', 'octet-stream')
        html_part.set_payload(html_content.encode('utf-8'))
        encoders.encode_base64(html_part)
        html_part.add_header(
            'Content-Disposition',
            f'attachment; filename= safety_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        )
        msg.attach(html_part)
        
        # Connect to Gmail SMTP server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        return True, "✅ HTML Report email sent successfully! 📧 (Check spam folder if needed)"
        
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed. Please check Gmail App Password settings."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error occurred: {str(e)}"
    except Exception as e:
        return False, f"❌ Email sending failed: {str(e)}"

def get_master_csv_path():
    """Get the path to the master CSV file"""
    return "violation_history_master.csv"

def initialize_master_csv():
    """Initialize master CSV file if it doesn't exist"""
    csv_path = get_master_csv_path()
    if not os.path.exists(csv_path):
        # Create with headers
        df = pd.DataFrame(columns=[
            'session_id', 'session_date', 'session_time', 'analysis_type',
            'frame', 'timestamp', 'type', 'severity', 'location'
        ])
        df.to_csv(csv_path, index=False)
        return True, "Master CSV file created"
    return True, "Master CSV file exists"

def append_to_master_csv(violations, analysis_type="Video Analysis"):
    """Append violations from current session to master CSV"""
    try:
        csv_path = get_master_csv_path()
        
        # Initialize if doesn't exist
        initialize_master_csv()
        
        # Generate session ID
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_date = datetime.now().strftime('%Y-%m-%d')
        session_time = datetime.now().strftime('%H:%M:%S')
        
        # Prepare data for this session
        session_data = []
        for v in violations:
            session_data.append({
                'session_id': session_id,
                'session_date': session_date,
                'session_time': session_time,
                'analysis_type': analysis_type,
                'frame': v.get('frame', 'N/A'),
                'timestamp': v.get('timestamp', 'N/A'),
                'type': v.get('type', 'Unknown'),
                'severity': v.get('severity', 'Unknown'),
                'location': v.get('location', 'N/A')
            })
        
        # Read existing data
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            existing_df = pd.read_csv(csv_path)
            new_df = pd.DataFrame(session_data)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = pd.DataFrame(session_data)
        
        # Save combined data
        combined_df.to_csv(csv_path, index=False)
        
        return True, f"Added {len(session_data)} violations to master CSV (Session: {session_id})"
    except Exception as e:
        return False, f"Failed to append to master CSV: {str(e)}"

def load_master_csv():
    """Load the master CSV file"""
    try:
        csv_path = get_master_csv_path()
        if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
            df = pd.read_csv(csv_path)
            return df, f"Loaded {len(df)} violations from {df['session_id'].nunique() if len(df) > 0 else 0} sessions"
        else:
            initialize_master_csv()
            return pd.DataFrame(), "No historical data available yet"
    except Exception as e:
        return pd.DataFrame(), f"Error loading master CSV: {str(e)}"

def send_realtime_violation_alert_direct(frame, violations, frame_number, recipient_email, sender_name, timestamp=None):
    """Direct email sending function for queue worker (same as send_realtime_violation_alert but without queue)"""
    if not violations:
        return True, "No violations to report"
    
    try:
        # Gmail SMTP configuration
        sender_email = "safetyeyeteam8@gmail.com"
        password = "dtwmwimtbqquqwda"  # App Password for Gmail
        
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = f"🚨 SAFETY VIOLATION ALERT - Frame {frame_number} - {datetime.now().strftime('%H:%M:%S')}"
        
        # Count violations by severity
        high_violations = [v for v in violations if v['severity'] == 'High']
        medium_violations = [v for v in violations if v['severity'] == 'Medium']
        low_violations = [v for v in violations if v['severity'] == 'Low']
        
        # Create detailed violation list
        violation_details = []
        for violation in violations:
            severity_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
            violation_details.append(
                f"{severity_emoji.get(violation['severity'], '⚪')} {violation['type']} "
                f"at {violation['location']} - {violation['severity']} Priority"
            )
        
        # Email body with violation details
        email_body = f"""
🚨 CONSTRUCTION SITE SAFETY VIOLATION DETECTED!

Alert Details:
⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🎬 Frame: {frame_number}
⌛ Timestamp: {timestamp or 'N/A'}
👤 Reported by: {sender_name or 'Safety Monitor'}

VIOLATION SUMMARY:
📊 Total Violations: {len(violations)}
🔴 High Priority: {len(high_violations)}
🟡 Medium Priority: {len(medium_violations)}  
🟢 Low Priority: {len(low_violations)}

DETAILED VIOLATIONS:
{chr(10).join(violation_details)}

⚠️ IMMEDIATE ACTION REQUIRED ⚠️
Please review the attached frame image and take appropriate safety measures.

The attached image shows the exact frame with bounding boxes highlighting the violations detected by our AI safety monitoring system.

Best regards,
Construction Safety Monitor System 🚧

---
This is an automated safety alert. Please ensure all workers comply with safety regulations.
        """
        
        # Attach body to email
        msg.attach(MIMEText(email_body, "plain"))
        
        # Convert frame to JPG and attach as image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Save image to bytes buffer
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG', quality=95)
        img_data = img_buffer.getvalue()
        
        # Attach image
        img_part = MIMEBase('application', 'octet-stream')
        img_part.set_payload(img_data)
        encoders.encode_base64(img_part)
        img_filename = f"violation_frame_{frame_number}_{datetime.now().strftime('%H%M%S')}.jpg"
        img_part.add_header(
            'Content-Disposition',
            f'attachment; filename= {img_filename}'
        )
        msg.attach(img_part)
        
        # Connect to Gmail SMTP server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        return True, f"✅ Real-time alert sent for {len(violations)} violations"
        
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed for real-time alert"
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error in real-time alert: {str(e)}"
    except Exception as e:
        return False, f"❌ Real-time alert failed: {str(e)}"

def send_realtime_violation_alert(frame, violations, frame_number, recipient_email, sender_name, timestamp=None):
    """Queue-based real-time violation alert (adds email to queue for processing)"""
    if not violations:
        return True, "No violations to report"
    
    try:
        # Add email to queue instead of sending directly
        email_data = {
            'type': 'realtime_alert',
            'frame': frame.copy(),  # Make a copy to avoid memory issues
            'violations': violations.copy(),
            'frame_number': frame_number,
            'recipient_email': recipient_email,
            'sender_name': sender_name,
            'timestamp': timestamp
        }
        
        email_queue.add_email(email_data)
        return True, f"✅ Real-time alert queued for {len(violations)} violations"
        
    except Exception as e:
        return False, f"❌ Failed to queue real-time alert: {str(e)}"

def send_csv_summary_queued(csv_data, filename, recipient_email, sender_name, analysis_type):
    """Queue-based CSV summary email (adds email to queue for processing)"""
    try:
        # Add email to queue instead of sending directly
        email_data = {
            'type': 'csv_summary',
            'csv_data': csv_data,
            'filename': filename,
            'recipient_email': recipient_email,
            'sender_name': sender_name,
            'analysis_type': analysis_type
        }
        
        email_queue.add_email(email_data)
        return True, "✅ CSV summary report queued for sending"
        
    except Exception as e:
        return False, f"❌ Failed to queue CSV summary: {str(e)}"

def send_email_notification_only(recipient_email, sender_name, filename, analysis_type):
    """Send simple email notification without attachment using Gmail SMTP"""
    try:
        # Gmail SMTP configuration
        sender_email = "liki.codes@gmail.com"
        password = "fzxxciqopcbcbgxd"  # App Password for Gmail
        
        # Create email message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = "🚧 Construction Safety Analysis Complete"
        
        # Email body
        email_body = f"""
Hello!

Your construction safety analysis has been completed successfully.

📊 Analysis Type: {analysis_type}
⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📁 Results File: {filename}
👤 Requested by: {sender_name or 'Safety Monitor'}

Note: Please download the CSV file directly from the application.

Best regards,
Construction Safety Monitor 🚧
        """
        
        # Attach body to email
        msg.attach(MIMEText(email_body, "plain"))
        
        # Connect to Gmail SMTP server and send
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
        
        return True, "✅ Notification sent! 📧 (CSV ready for download)"
        
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Email authentication failed. Please check Gmail settings."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error occurred: {str(e)}"
    except Exception as e:
        return False, f"❌ Email notification failed: {str(e)}"

def generate_detailed_analytics_report(total_violations, frame_stats):
    """Generate a comprehensive analytics report as a formatted text/markdown file"""
    
    # Calculate all metrics
    total_violations_count = len(total_violations)
    high_violations = [v for v in total_violations if v['severity'] == 'High']
    medium_violations = [v for v in total_violations if v['severity'] == 'Medium']
    low_violations = [v for v in total_violations if v['severity'] == 'Low']
    
    total_persons = sum(f['persons'] for f in frame_stats)
    total_safety_equipped = sum(f['safety_equipped'] for f in frame_stats)
    compliance_rate = (total_safety_equipped / total_persons * 100) if total_persons > 0 else 0
    
    violation_types = {}
    for v in total_violations:
        vtype = v['type']
        violation_types[vtype] = violation_types.get(vtype, 0) + 1
    
    # Calculate advanced metrics
    avg_violations_per_frame = total_violations_count / len(frame_stats) if frame_stats else 0
    risk_score = (len(high_violations) * 3 + len(medium_violations) * 2 + len(low_violations) * 1)
    max_risk = total_violations_count * 3
    risk_percentage = (risk_score / max_risk * 100) if max_risk > 0 else 0
    violation_density = (total_violations_count / total_persons) if total_persons > 0 else 0
    high_severity_rate = (len(high_violations) / total_violations_count * 100) if total_violations_count > 0 else 0
    frames_with_violations = len([f for f in frame_stats if f['violations'] > 0])
    violation_frame_rate = (frames_with_violations / len(frame_stats) * 100) if frame_stats else 0
    avg_workers = sum(f['persons'] for f in frame_stats) / len(frame_stats) if frame_stats else 0
    violation_frequency = (total_violations_count / len(frame_stats) * 100) if frame_stats else 0
    
    # Generate report
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║        CONSTRUCTION SAFETY MONITORING - DETAILED ANALYTICS REPORT     ║
╚══════════════════════════════════════════════════════════════════════╝

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

═══════════════════════════════════════════════════════════════════════
                        EXECUTIVE SUMMARY
═══════════════════════════════════════════════════════════════════════

Total Frames Analyzed:      {len(frame_stats)}
Total Violations Detected:  {total_violations_count}
Total Workers Processed:    {total_persons}
Overall Safety Compliance:  {compliance_rate:.1f}%

RISK ASSESSMENT:
  Risk Score:               {risk_percentage:.1f}% ({risk_score}/{max_risk})
  High Severity Rate:       {high_severity_rate:.1f}%
  Violation Density:        {violation_density:.2f} violations per worker

═══════════════════════════════════════════════════════════════════════
                     VIOLATION BREAKDOWN BY SEVERITY
═══════════════════════════════════════════════════════════════════════

🔴 HIGH Priority Violations:     {len(high_violations):>6}  ({len(high_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%)
🟡 MEDIUM Priority Violations:   {len(medium_violations):>6}  ({len(medium_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%)
🟢 LOW Priority Violations:      {len(low_violations):>6}  ({len(low_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%)

═══════════════════════════════════════════════════════════════════════
                     KEY PERFORMANCE INDICATORS
═══════════════════════════════════════════════════════════════════════

Average Violations per Frame:    {avg_violations_per_frame:.2f}
Frames with Violations:          {violation_frame_rate:.1f}% ({frames_with_violations}/{len(frame_stats)})
Average Workers per Frame:       {avg_workers:.1f}
Violation Frequency:             {violation_frequency:.1f} per 100 frames

═══════════════════════════════════════════════════════════════════════
                     TOP VIOLATION TYPES
═══════════════════════════════════════════════════════════════════════

"""
    
    # Add top violation types
    sorted_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)
    for i, (vtype, count) in enumerate(sorted_violations[:10], 1):
        percentage = (count / total_violations_count * 100) if total_violations_count > 0 else 0
        report += f"{i:2d}. {vtype:<50} {count:>6} ({percentage:>5.1f}%)\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════════════
                     SAFETY HOTSPOTS (TOP 5 FRAMES)
═══════════════════════════════════════════════════════════════════════

"""
    
    # Add hotspot frames
    hotspot_frames = sorted(frame_stats, key=lambda x: x['violations'], reverse=True)[:5]
    report += f"{'Frame':<10} {'Violations':<15} {'Workers':<15} {'Safety Equipped':<15}\n"
    report += f"{'-'*55}\n"
    for f in hotspot_frames:
        if f['violations'] > 0:
            report += f"#{f['frame']:<9} {f['violations']:<15} {f['persons']:<15} {f['safety_equipped']:<15}\n"
    
    # Temporal analysis
    first_violation = next((v for v in total_violations), None)
    last_violation = total_violations[-1] if total_violations else None
    
    report += f"""
═══════════════════════════════════════════════════════════════════════
                     TEMPORAL ANALYSIS
═══════════════════════════════════════════════════════════════════════

First Violation Detected:   Frame {first_violation['frame'] if first_violation else 'N/A'}
Latest Violation Detected:   Frame {last_violation['frame'] if last_violation else 'N/A'}

═══════════════════════════════════════════════════════════════════════
                     RECOMMENDATIONS
═══════════════════════════════════════════════════════════════════════

"""
    
    # Add recommendations based on analysis
    if high_severity_rate > 50:
        report += "⚠️  CRITICAL: Over 50% violations are HIGH severity. Immediate action required!\n"
    if compliance_rate < 50:
        report += "⚠️  WARNING: Safety compliance below 50%. Enhance safety training programs.\n"
    if violation_density > 1:
        report += "⚠️  ALERT: More than 1 violation per worker on average. Review safety protocols.\n"
    if violation_frame_rate > 75:
        report += "⚠️  CONCERN: Violations detected in over 75% of frames. Site-wide safety audit needed.\n"
    
    if high_severity_rate <= 20 and compliance_rate >= 80:
        report += "✅  GOOD: Low high-severity rate and high compliance. Maintain current standards.\n"
    
    # Add top violation recommendations
    if sorted_violations:
        top_violation_type = sorted_violations[0][0]
        report += f"\n📌 Focus Area: '{top_violation_type}' is the most common violation.\n"
        report += "   Consider targeted training and enhanced monitoring for this specific issue.\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════════════
                     DETAILED VIOLATION LOG
═══════════════════════════════════════════════════════════════════════

"""
    
    # Add detailed violation log (limited to first 50 for readability)
    report += f"{'Frame':<10} {'Severity':<12} {'Type':<45} {'Location':<20}\n"
    report += f"{'-'*87}\n"
    for v in total_violations[:50]:
        report += f"#{v['frame']:<9} {v['severity']:<12} {v['type']:<45} {v.get('location', 'N/A'):<20}\n"
    
    if len(total_violations) > 50:
        report += f"\n... and {len(total_violations) - 50} more violations (see CSV for complete list)\n"
    
    report += f"""
═══════════════════════════════════════════════════════════════════════
                     END OF REPORT
═══════════════════════════════════════════════════════════════════════

Generated by Construction Safety Monitor v1.0
For questions or support, contact: safety@construction-monitor.com
"""
    
    return report

def generate_html_report_with_graphs(total_violations, frame_stats):
    """Generate a comprehensive HTML report with embedded graphs"""
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Calculate all metrics
    total_violations_count = len(total_violations)
    high_violations = [v for v in total_violations if v['severity'] == 'High']
    medium_violations = [v for v in total_violations if v['severity'] == 'Medium']
    low_violations = [v for v in total_violations if v['severity'] == 'Low']
    
    total_persons = sum(f['persons'] for f in frame_stats)
    total_safety_equipped = sum(f['safety_equipped'] for f in frame_stats)
    compliance_rate = (total_safety_equipped / total_persons * 100) if total_persons > 0 else 0
    
    violation_types = {}
    for v in total_violations:
        vtype = v['type']
        violation_types[vtype] = violation_types.get(vtype, 0) + 1
    
    # Calculate advanced metrics
    avg_violations_per_frame = total_violations_count / len(frame_stats) if frame_stats else 0
    risk_score = (len(high_violations) * 3 + len(medium_violations) * 2 + len(low_violations) * 1)
    max_risk = total_violations_count * 3
    risk_percentage = (risk_score / max_risk * 100) if max_risk > 0 else 0
    violation_density = (total_violations_count / total_persons) if total_persons > 0 else 0
    high_severity_rate = (len(high_violations) / total_violations_count * 100) if total_violations_count > 0 else 0
    frames_with_violations = len([f for f in frame_stats if f['violations'] > 0])
    violation_frame_rate = (frames_with_violations / len(frame_stats) * 100) if frame_stats else 0
    avg_workers = sum(f['persons'] for f in frame_stats) / len(frame_stats) if frame_stats else 0
    
    # Generate graphs as base64 encoded images
    graphs = {}
    
    # Graph 1: Violations Over Time
    fig, ax = plt.subplots(figsize=(10, 4))
    frames = [f['frame'] for f in frame_stats]
    violations_per_frame = [f['violations'] for f in frame_stats]
    ax.plot(frames, violations_per_frame, color='#e74c3c', linewidth=2, marker='o', markersize=3)
    ax.fill_between(frames, violations_per_frame, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Frame Number', fontsize=10, fontweight='bold')
    ax.set_ylabel('Violations', fontsize=10, fontweight='bold')
    ax.set_title('Violations Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    graphs['violations_timeline'] = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Graph 2: Severity Distribution Pie Chart
    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [len(high_violations), len(medium_violations), len(low_violations)]
    labels = ['High', 'Medium', 'Low']
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    explode = (0.1, 0.05, 0)
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title('Violation Severity Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    graphs['severity_pie'] = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Graph 3: Workers Detected Over Time
    fig, ax = plt.subplots(figsize=(10, 4))
    persons = [f['persons'] for f in frame_stats]
    ax.fill_between(frames, persons, alpha=0.4, color='#3498db')
    ax.plot(frames, persons, color='#2980b9', linewidth=2)
    ax.set_xlabel('Frame Number', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Workers', fontsize=10, fontweight='bold')
    ax.set_title('Workers Detected Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    graphs['workers_timeline'] = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Graph 4: Top Violation Types Bar Chart
    if violation_types:
        sorted_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:10]
        fig, ax = plt.subplots(figsize=(10, 6))
        types = [v[0][:40] for v in sorted_violations]  # Truncate long names
        counts = [v[1] for v in sorted_violations]
        bars = ax.barh(types, counts, color='#9b59b6')
        ax.set_xlabel('Count', fontsize=10, fontweight='bold')
        ax.set_ylabel('Violation Type', fontsize=10, fontweight='bold')
        ax.set_title('Top 10 Violation Types', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        graphs['violation_types'] = base64.b64encode(buf.read()).decode()
        plt.close()
    
    # Graph 5: Compliance Trend
    fig, ax = plt.subplots(figsize=(10, 4))
    compliance_data = []
    for f in frame_stats:
        if f['persons'] > 0:
            compliance = (f['safety_equipped'] / f['persons'] * 100)
            compliance_data.append(compliance)
        else:
            compliance_data.append(0)
    
    ax.plot(frames, compliance_data, color='#27ae60', linewidth=2, marker='o', markersize=3)
    ax.fill_between(frames, compliance_data, alpha=0.3, color='#27ae60')
    ax.set_xlabel('Frame Number', fontsize=10, fontweight='bold')
    ax.set_ylabel('Compliance %', fontsize=10, fontweight='bold')
    ax.set_title('Safety Compliance Trend', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.axhline(y=80, color='orange', linestyle='--', label='Target: 80%', linewidth=1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    graphs['compliance_trend'] = base64.b64encode(buf.read()).decode()
    plt.close()
    
    # Generate HTML report
    html_report = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Construction Safety Analytics Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            background: white;
            border-radius: 10px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 4px solid #3498db;
            padding-bottom: 20px;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: #34495e;
            border-left: 5px solid #3498db;
            padding-left: 15px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 25px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.3s;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .graph-container {{
            margin: 30px 0;
            text-align: center;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .graph-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: bold;
        }}
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .severity-high {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .severity-medium {{
            color: #f39c12;
            font-weight: bold;
        }}
        .severity-low {{
            color: #2ecc71;
            font-weight: bold;
        }}
        .alert {{
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 5px;
            border-left: 5px solid;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border-color: #e74c3c;
            color: #721c24;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border-color: #f39c12;
            color: #856404;
        }}
        .alert-success {{
            background-color: #d4edda;
            border-color: #2ecc71;
            color: #155724;
        }}
        .footer {{
            text-align: center;
            margin-top: 50px;
            padding: 20px;
            background: #34495e;
            color: white;
            border-radius: 10px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
            text-align: center;
            margin: 20px 0;
        }}
        .filter-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }}
        .filter-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        #searchInput {{
            transition: border-color 0.3s;
        }}
        #searchInput:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 10px rgba(102, 126, 234, 0.3);
        }}
        .data-row {{
            transition: background-color 0.2s;
        }}
        .data-row:hover {{
            background-color: #f0f0f0 !important;
        }}
        th {{
            user-select: none;
            position: relative;
        }}
        th:hover {{
            background: linear-gradient(135deg, #5568d3 0%, #653a8b 100%);
        }}
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .filter-btn, #searchInput {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚧 CONSTRUCTION SAFETY ANALYTICS REPORT</h1>
            <p class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>

        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Total Frames</div>
                <div class="metric-value">{len(frame_stats)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Violations</div>
                <div class="metric-value">{total_violations_count}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Workers</div>
                <div class="metric-value">{total_persons}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Safety Compliance</div>
                <div class="metric-value">{compliance_rate:.1f}%</div>
            </div>
        </div>

        <h2>⚠️ Violation Breakdown</h2>
        <div class="metrics-grid">
            <div class="metric-card" style="background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);">
                <div class="metric-label">High Priority</div>
                <div class="metric-value">{len(high_violations)}</div>
                <div class="metric-label">{len(high_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);">
                <div class="metric-label">Medium Priority</div>
                <div class="metric-value">{len(medium_violations)}</div>
                <div class="metric-label">{len(medium_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);">
                <div class="metric-label">Low Priority</div>
                <div class="metric-value">{len(low_violations)}</div>
                <div class="metric-label">{len(low_violations)/total_violations_count*100 if total_violations_count > 0 else 0:.1f}%</div>
            </div>
        </div>

        <div class="graph-container">
            <h3>Severity Distribution</h3>
            <img src="data:image/png;base64,{graphs.get('severity_pie', '')}" alt="Severity Pie Chart">
        </div>

        <h2>📈 Temporal Analysis</h2>
        
        <div class="graph-container">
            <h3>Violations Over Time</h3>
            <img src="data:image/png;base64,{graphs.get('violations_timeline', '')}" alt="Violations Timeline">
        </div>

        <div class="graph-container">
            <h3>Workers Detected Over Time</h3>
            <img src="data:image/png;base64,{graphs.get('workers_timeline', '')}" alt="Workers Timeline">
        </div>

        <div class="graph-container">
            <h3>Safety Compliance Trend</h3>
            <img src="data:image/png;base64,{graphs.get('compliance_trend', '')}" alt="Compliance Trend">
        </div>

        <h2>🎯 Key Performance Indicators</h2>
        <div class="metrics-grid">
            <div class="metric-card" style="background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);">
                <div class="metric-label">Risk Score</div>
                <div class="metric-value">{risk_percentage:.1f}%</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);">
                <div class="metric-label">Avg Violations/Frame</div>
                <div class="metric-value">{avg_violations_per_frame:.2f}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #1abc9c 0%, #16a085 100%);">
                <div class="metric-label">Violations/Worker</div>
                <div class="metric-value">{violation_density:.2f}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #e67e22 0%, #d35400 100%);">
                <div class="metric-label">Frames w/ Violations</div>
                <div class="metric-value">{violation_frame_rate:.1f}%</div>
            </div>
        </div>

        <h2>🏆 Top Violation Types</h2>
        {'<div class="graph-container"><img src="data:image/png;base64,' + graphs.get('violation_types', '') + '" alt="Top Violations"></div>' if 'violation_types' in graphs else ''}
        
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Violation Type</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add top violations to table
    sorted_violations = sorted(violation_types.items(), key=lambda x: x[1], reverse=True)[:10]
    for i, (vtype, count) in enumerate(sorted_violations, 1):
        percentage = (count / total_violations_count * 100) if total_violations_count > 0 else 0
        html_report += f"""
                <tr>
                    <td>{i}</td>
                    <td>{vtype}</td>
                    <td><strong>{count}</strong></td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""
    
    html_report += """
            </tbody>
        </table>

        <h2>🔥 Safety Hotspots</h2>
        <p>Frames with the highest number of violations:</p>
        <table>
            <thead>
                <tr>
                    <th>Frame</th>
                    <th>Violations</th>
                    <th>Workers</th>
                    <th>Safety Equipped</th>
                </tr>
            </thead>
            <tbody>
"""
    
    # Add hotspot frames
    hotspot_frames = sorted(frame_stats, key=lambda x: x['violations'], reverse=True)[:5]
    for f in hotspot_frames:
        if f['violations'] > 0:
            html_report += f"""
                <tr>
                    <td><strong>#{f['frame']}</strong></td>
                    <td class="severity-high">{f['violations']}</td>
                    <td>{f['persons']}</td>
                    <td>{f['safety_equipped']}</td>
                </tr>
"""
    
    html_report += f"""
        <h2>📊 Interactive Violation Data</h2>
        <p>Search, filter, and sort through all violation records:</p>
        
        <div style="margin: 20px 0;">
            <input type="text" id="searchInput" placeholder="🔍 Search violations..." 
                   style="width: 100%; padding: 12px 20px; font-size: 16px; border: 2px solid #3498db; 
                          border-radius: 5px; box-sizing: border-box;">
        </div>
        
        <div style="margin: 20px 0;">
            <label style="margin-right: 15px; font-weight: bold;">Filter by Severity:</label>
            <button class="filter-btn" onclick="filterTable('all')" style="margin: 5px;">All</button>
            <button class="filter-btn" onclick="filterTable('High')" style="margin: 5px;">🔴 High</button>
            <button class="filter-btn" onclick="filterTable('Medium')" style="margin: 5px;">🟡 Medium</button>
            <button class="filter-btn" onclick="filterTable('Low')" style="margin: 5px;">🟢 Low</button>
        </div>
        
        <div style="overflow-x: auto;">
            <table id="violationTable">
                <thead>
                    <tr>
                        <th onclick="sortTable(0)" style="cursor: pointer;">Frame ↕</th>
                        <th onclick="sortTable(1)" style="cursor: pointer;">Violation Type ↕</th>
                        <th onclick="sortTable(2)" style="cursor: pointer;">Severity ↕</th>
                        <th onclick="sortTable(3)" style="cursor: pointer;">Location ↕</th>
                        <th onclick="sortTable(4)" style="cursor: pointer;">Timestamp ↕</th>
                    </tr>
                </thead>
                <tbody id="tableBody">
"""
    
    # Add all violations to table
    for v in total_violations:
        severity_class = f"severity-{v['severity'].lower()}"
        html_report += f"""
                    <tr class="data-row" data-severity="{v['severity']}">
                        <td><strong>#{v['frame']}</strong></td>
                        <td>{v['type']}</td>
                        <td class="{severity_class}">{v['severity']}</td>
                        <td>{v.get('location', 'N/A')}</td>
                        <td>{v.get('timestamp', 'N/A')}</td>
                    </tr>
"""
    
    html_report += """
                </tbody>
            </table>
        </div>
        
        <p style="margin-top: 20px; color: #7f8c8d; font-style: italic;">
            💡 <strong>Tip:</strong> Click on column headers to sort. Use the search box to find specific violations.
        </p>
        
        <script>
            // Search functionality
            document.getElementById('searchInput').addEventListener('keyup', function() {
                var input = this.value.toLowerCase();
                var rows = document.querySelectorAll('#tableBody tr');
                
                rows.forEach(function(row) {
                    var text = row.textContent.toLowerCase();
                    if (text.includes(input)) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
            });
            
            // Filter by severity
            function filterTable(severity) {
                var rows = document.querySelectorAll('.data-row');
                
                rows.forEach(function(row) {
                    if (severity === 'all' || row.dataset.severity === severity) {
                        row.style.display = '';
                    } else {
                        row.style.display = 'none';
                    }
                });
                
                // Highlight active filter button
                document.querySelectorAll('.filter-btn').forEach(function(btn) {
                    btn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
                    btn.style.color = 'white';
                });
                event.target.style.background = '#2ecc71';
            }
            
            // Sort table
            function sortTable(n) {
                var table = document.getElementById("violationTable");
                var rows = Array.from(table.querySelectorAll('tbody tr'));
                var ascending = true;
                
                rows.sort(function(a, b) {
                    var x = a.cells[n].textContent.toLowerCase();
                    var y = b.cells[n].textContent.toLowerCase();
                    
                    // Try to parse as numbers for frame column
                    if (n === 0) {
                        x = parseInt(x.replace('#', '')) || 0;
                        y = parseInt(y.replace('#', '')) || 0;
                    }
                    
                    if (x < y) return ascending ? -1 : 1;
                    if (x > y) return ascending ? 1 : -1;
                    return 0;
                });
                
                var tbody = table.querySelector('tbody');
                rows.forEach(function(row) {
                    tbody.appendChild(row);
                });
            }
        </script>

        <h2>💡 Recommendations</h2>
"""
    
    # Add recommendations
    recommendations = []
    if high_severity_rate > 50:
        recommendations.append('<div class="alert alert-danger">⚠️ <strong>CRITICAL:</strong> Over 50% of violations are HIGH severity. Immediate corrective action required!</div>')
    if compliance_rate < 50:
        recommendations.append('<div class="alert alert-danger">⚠️ <strong>WARNING:</strong> Safety compliance below 50%. Enhanced safety training programs needed.</div>')
    if violation_density > 1:
        recommendations.append('<div class="alert alert-warning">⚠️ <strong>ALERT:</strong> More than 1 violation per worker on average. Review and strengthen safety protocols.</div>')
    if violation_frame_rate > 75:
        recommendations.append('<div class="alert alert-warning">⚠️ <strong>CONCERN:</strong> Violations detected in over 75% of frames. Site-wide safety audit recommended.</div>')
    
    if high_severity_rate <= 20 and compliance_rate >= 80:
        recommendations.append('<div class="alert alert-success">✅ <strong>GOOD:</strong> Low high-severity rate and high compliance. Maintain current safety standards.</div>')
    
    if sorted_violations:
        top_violation_type = sorted_violations[0][0]
        recommendations.append(f'<div class="alert alert-warning">📌 <strong>Focus Area:</strong> "{top_violation_type}" is the most common violation. Consider targeted training and enhanced monitoring for this specific issue.</div>')
    
    for rec in recommendations:
        html_report += rec + '\n'
    
    html_report += f"""
        <div class="footer">
            <h3>Construction Safety Monitor v1.0</h3>
            <p>This report was automatically generated by AI-powered safety monitoring system</p>
            <p><small>Report ID: {datetime.now().strftime('%Y%m%d%H%M%S')} | © 2025 Construction Safety Monitor</small></p>
        </div>
    </div>
</body>
</html>
"""
    
    return html_report

def process_live_camera(model, class_names, colors, confidence_threshold, debug_mode,
                       recipient_email=None, sender_name=None, 
                       real_time_alerts=False, send_csv_summary=False):
    """Process live camera feed with real-time analysis"""
    
    st.markdown("### 📹 Live Camera Analysis")
    
    # Start email queue worker if real-time alerts are enabled
    if real_time_alerts and recipient_email:
        email_queue.start_worker()
        st.info("📧 Email notification system activated")
    
    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        start_camera = st.button("🎥 Start Camera", use_container_width=True)
    with col_btn2:
        stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
    with col_btn3:
        save_session = st.button("💾 Save Session", use_container_width=True)
    
    # Create session state for camera control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'camera_violations' not in st.session_state:
        st.session_state.camera_violations = []
    if 'camera_frame_stats' not in st.session_state:
        st.session_state.camera_frame_stats = []
    if 'camera_frame_count' not in st.session_state:
        st.session_state.camera_frame_count = 0
    
    if start_camera:
        st.session_state.camera_running = True
        st.session_state.camera_violations = []
        st.session_state.camera_frame_stats = []
        st.session_state.camera_frame_count = 0
    
    if stop_camera:
        st.session_state.camera_running = False
    
    # Layout: Camera feed and alerts side by side
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("#### 📹 Live Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("#### 🚨 Live Alerts")
        alert_placeholder = st.empty()
    
    # Statistics section
    st.markdown("### 📊 Live Statistics")
    stats_placeholder = st.empty()
    
    # Camera processing
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)  # 0 for default camera
        
        if not cap.isOpened():
            st.error("❌ Cannot access camera. Please check camera permissions.")
            st.session_state.camera_running = False
        else:
            st.success("✅ Camera active - Press 'Stop Camera' to end session")
            
            # Process frames
            email_sent_count = 0
            
            while st.session_state.camera_running:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("⚠️ Failed to capture frame")
                    break
                
                st.session_state.camera_frame_count += 1
                
                # Process frame
                annotated_frame, violations, person_count, safety_equipped, filtered_count = process_video_frame(
                    frame, model, class_names, colors, debug_mode
                )
                
                # Update statistics
                st.session_state.camera_frame_stats.append({
                    'frame': st.session_state.camera_frame_count,
                    'violations': len(violations),
                    'persons': person_count,
                    'safety_equipped': safety_equipped,
                    'filtered': filtered_count
                })
                
                # Add violations to session list
                for violation in violations:
                    violation['frame'] = st.session_state.camera_frame_count
                    violation['timestamp'] = datetime.now().strftime('%H:%M:%S')
                    st.session_state.camera_violations.append(violation)
                
                # Send real-time email alerts
                if real_time_alerts and recipient_email and violations:
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    send_realtime_violation_alert(
                        annotated_frame, violations, st.session_state.camera_frame_count,
                        recipient_email, sender_name, timestamp
                    )
                    email_sent_count += 1
                
                # Display frame
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update alerts
                update_live_alerts(alert_placeholder, st.session_state.camera_violations, st.session_state.camera_frame_count)
                
                # Update stats
                update_live_stats(stats_placeholder, st.session_state.camera_violations, 
                                st.session_state.camera_frame_stats, st.session_state.camera_frame_count,
                                show_email_queue=real_time_alerts)
                
                # Small delay to reduce CPU usage
                import time
                time.sleep(0.03)  # ~30 FPS
            
            cap.release()
            
            # Stop email queue worker
            if real_time_alerts and recipient_email:
                email_queue.stop_worker()
                if email_sent_count > 0:
                    st.success(f"✅ Sent {email_sent_count} real-time violation alerts")
    
    # Save session functionality
    if save_session and len(st.session_state.camera_violations) > 0:
        st.markdown("---")
        st.markdown("### 💾 Save Session Data")
        
        # Append to master CSV
        success, message = append_to_master_csv(st.session_state.camera_violations, "Live Camera Analysis")
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
        
        # Generate reports
        col_save1, col_save2 = st.columns(2)
        
        with col_save1:
            # CSV Report
            csv_data = "Frame,Type,Severity,Location,Timestamp\n"
            for v in st.session_state.camera_violations:
                csv_data += f"{v['frame']},{v['type']},{v['severity']},{v.get('location', 'N/A')},{v.get('timestamp', 'N/A')}\n"
            
            filename = f"live_camera_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            st.download_button(
                label="📊 Download CSV Report",
                data=csv_data,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        with col_save2:
            # HTML Report
            html_content = generate_html_report_with_graphs(
                st.session_state.camera_violations,
                st.session_state.camera_frame_stats
            )
            html_filename = f"live_camera_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            st.download_button(
                label="📄 Download HTML Report",
                data=html_content,
                file_name=html_filename,
                mime="text/html",
                use_container_width=True
            )
        
        # Send email summary if configured
        if send_csv_summary and recipient_email:
            send_csv_summary_queued(csv_data, filename, recipient_email, sender_name, "Live Camera Analysis")
            st.info(f"📧 CSV summary queued for email to {recipient_email}")
        
        # Show session summary
        st.markdown("### 📈 Session Summary")
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        
        with sum_col1:
            st.metric("Total Frames", st.session_state.camera_frame_count)
        with sum_col2:
            st.metric("Total Violations", len(st.session_state.camera_violations))
        with sum_col3:
            high_count = len([v for v in st.session_state.camera_violations if v['severity'] == 'High'])
            st.metric("High Severity", high_count)
        with sum_col4:
            total_persons = sum(f['persons'] for f in st.session_state.camera_frame_stats)
            st.metric("Total Workers", total_persons)

def main():
    st.title("🚧 Construction Site Safety Monitor")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "Custom Trained Model (Recommended)": "models/construction_best.pt",
        "YOLOv8n (Base Model)": "models/yolov8n.pt"
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)
    model_path = model_options[selected_model]

    # Analysis mode: Video or Image or History or Live Camera
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Video Analysis", "Image Analysis", "Live Camera", "Violation History"],
        index=0,
        help="Choose whether to analyze a video, image, live camera feed, or view historical violation data"
    )
    
    # Show mode-specific description
    if analysis_mode == "Video Analysis":
        st.markdown("Upload a video to monitor construction site safety and detect PPE violations")
    elif analysis_mode == "Image Analysis":
        st.markdown("Upload an image to detect PPE violations and safety equipment")
    elif analysis_mode == "Live Camera":
        st.markdown("🎥 Use your device camera for real-time construction site safety monitoring")
    else:
        st.markdown("View and analyze historical violation data from all previous sessions")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    
    # Show model information
    if "Custom Trained" in selected_model:
        st.sidebar.success("🎯 Using custom trained model (100 epochs)")
        st.sidebar.info("This model was specifically trained on construction site safety data for optimal performance.")
    else:
        st.sidebar.info(f"Using {selected_model}")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.stop()
    
    # Define class names and colors
    class_names = {
        0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 
        4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone", 
        7: "Safety Vest", 8: "machinery", 9: "vehicle"
    }
    
    colors = get_class_colors()

    # If user selected Image Analysis, show a minimal image annotator UI (reuses model)
    if analysis_mode == "Image Analysis":
        # Image upload control
        uploaded_image = st.file_uploader("Upload Image (JPG/PNG/BMP/TIFF)", type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'])

        # Detection confidence for image mode
        image_conf = st.sidebar.slider("Image Detection Confidence", 0.1, 1.0, 0.25, 0.05)

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image_array = np.array(image)

            # Convert to BGR for OpenCV processing
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)

            # Show original and annotated images side-by-side
            # Use two equal-width columns so users can compare images easily
            col_orig, col_annot = st.columns(2)
            with col_orig:
                st.markdown("### Original Image")
                st.image(image, use_container_width=True)

            with st.spinner("Detecting objects in image..."):
                results = model(image_bgr)

            if len(results) > 0 and len(results[0].boxes) > 0:
                detections = results[0].boxes.data.cpu().numpy()

                # Draw boxes for detections above threshold
                annotated = image_bgr.copy()
                for det in detections:
                    x1, y1, x2, y2, conf, cid = det
                    if conf >= image_conf:
                        cls_id = int(cid)
                        color = colors.get(cls_id, (255,255,255))
                        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        label = f"{class_names.get(cls_id, f'cls_{cls_id}')} {conf:.2f}"
                        cv2.putText(annotated, label, (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                with col_annot:
                    st.markdown("### Annotated Image")
                    st.image(annotated_rgb, use_container_width=True)

                    # Offer download
                    pil_img = Image.fromarray(annotated_rgb)
                    buf = io.BytesIO()
                    pil_img.save(buf, format='PNG')
                    buf.seek(0)
                    st.download_button("Download Annotated Image", buf, file_name="annotated_image.png", mime="image/png")
            else:
                st.warning("No detections found in the image.")

        # Stop here for image analysis
        st.stop()
    
    # If user selected Live Camera, show the live camera interface
    elif analysis_mode == "Live Camera":
        # Camera uses same settings as video analysis
        confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.1)
        
        st.sidebar.markdown("### Detection Settings")
        st.sidebar.info("Lower threshold detects more objects but may include false positives")
        
        # Email configuration for live camera
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📧 Email Notification")
        
        send_email = st.sidebar.checkbox(
            "📬 Enable email notifications",
            value=False,
            help="Choose how you want to receive safety violation alerts"
        )
        
        if send_email:
            recipient_email = st.sidebar.text_input(
                "📮 Recipient Email",
                placeholder="recipient@example.com",
                help="Email address to receive the results"
            )
            
            sender_name = "Construction Safety Monitor - Live Camera"
            
            st.sidebar.markdown("#### 📬 Notification Type")
            email_mode = st.sidebar.radio(
                "Select notification method:",
                ["🚨 Real-time Alerts (Email per violation with frame image)", 
                 "📊 CSV Summary (Email report when you save session)"],
                help="Choose when to receive email notifications"
            )
            
            real_time_alerts = email_mode.startswith("🚨")
            send_csv_summary = email_mode.startswith("📊")
        else:
            recipient_email = None
            sender_name = None
            real_time_alerts = False
            send_csv_summary = False
        
        # Debug mode
        debug_mode = st.sidebar.checkbox("👨‍💻 Developer Mode", help="Show detailed technical information")
        
        # Vehicle filtering info
        with st.sidebar.expander("🚗 Enhanced Vehicle Filtering"):
            st.info("✅ Active: Multi-method filtering to reduce false positives from vehicles and machinery")
        
        # Process live camera
        process_live_camera(
            model, class_names, colors, confidence_threshold, debug_mode,
            recipient_email, sender_name, real_time_alerts, send_csv_summary
        )
        
        # Stop here for live camera
        st.stop()
    
    # If user selected Violation History, show the history dashboard
    elif analysis_mode == "Violation History":

        # Load master CSV
        df, message = load_master_csv()
        
        if len(df) == 0:
            st.info("📝 " + message)
            st.info("💡 Run some video/image analysis sessions to build violation history!")
            st.stop()
        
        st.success(f"✅ {message}")
        
        # Summary statistics
        st.markdown("### 📈 Overall Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Violations", len(df))
        with col2:
            st.metric("Total Sessions", df['session_id'].nunique())
        with col3:
            high_count = len(df[df['severity'] == 'High'])
            st.metric("High Severity", high_count)
        with col4:
            latest_session = df['session_date'].max() if len(df) > 0 else "N/A"
            st.metric("Latest Session", latest_session)
        
        # Filters
        st.markdown("### 🔍 Filter Data")
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Date range filter
            if 'session_date' in df.columns and len(df) > 0:
                date_options = ["All Dates"] + sorted(df['session_date'].unique().tolist(), reverse=True)
                selected_date = st.selectbox("📅 Filter by Date", date_options)
            else:
                selected_date = "All Dates"
        
        with filter_col2:
            # Severity filter
            severity_options = ["All Severities"] + df['severity'].unique().tolist()
            selected_severity = st.selectbox("⚠️ Filter by Severity", severity_options)
        
        with filter_col3:
            # Session filter
            session_options = ["All Sessions"] + sorted(df['session_id'].unique().tolist(), reverse=True)
            selected_session = st.selectbox("📂 Filter by Session", session_options)
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_date != "All Dates":
            filtered_df = filtered_df[filtered_df['session_date'] == selected_date]
        
        if selected_severity != "All Severities":
            filtered_df = filtered_df[filtered_df['severity'] == selected_severity]
        
        if selected_session != "All Sessions":
            filtered_df = filtered_df[filtered_df['session_id'] == selected_session]
        
        # Show filtered results
        st.markdown(f"### 📋 Filtered Results ({len(filtered_df)} violations)")
        
        if len(filtered_df) > 0:
            # Display dataframe
            st.dataframe(filtered_df, use_container_width=True, height=400)
            
            # Download filtered data
            st.markdown("### 📥 Download Reports")
            col_d1, col_d2, col_d3, col_d4 = st.columns(4)
            
            with col_d1:
                csv_data = filtered_df.to_csv(index=False)
                filename = f"filtered_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    label="📊 Filtered CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_d2:
                # Generate HTML report for filtered data
                # Convert filtered_df to violations list format
                filtered_violations = []
                for _, row in filtered_df.iterrows():
                    filtered_violations.append({
                        'frame': row.get('frame', 0),
                        'type': row.get('type', 'Unknown'),
                        'severity': row.get('severity', 'Low'),
                        'location': row.get('location', 'N/A'),
                        'timestamp': row.get('timestamp', 'N/A')
                    })
                
                # Create dummy frame_stats for HTML generation
                frame_count = len(filtered_df['frame'].unique()) if 'frame' in filtered_df.columns else len(filtered_df)
                frame_stats = [{'frame': i, 'violations': 0, 'persons': 0, 'safety_equipped': 0} for i in range(frame_count)]
                
                html_content = generate_html_report_with_graphs(filtered_violations, frame_stats)
                html_filename = f"filtered_violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                st.download_button(
                    label="📄 Filtered HTML",
                    data=html_content,
                    file_name=html_filename,
                    mime="text/html",
                    use_container_width=True
                )
            
            with col_d3:
                # Download full master CSV
                full_csv = df.to_csv(index=False)
                full_filename = f"master_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    label="📁 Full CSV",
                    data=full_csv,
                    file_name=full_filename,
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col_d4:
                # Generate HTML report for full data
                full_violations = []
                for _, row in df.iterrows():
                    full_violations.append({
                        'frame': row.get('frame', 0),
                        'type': row.get('type', 'Unknown'),
                        'severity': row.get('severity', 'Low'),
                        'location': row.get('location', 'N/A'),
                        'timestamp': row.get('timestamp', 'N/A')
                    })
                
                # Create dummy frame_stats for HTML generation
                full_frame_count = len(df['frame'].unique()) if 'frame' in df.columns else len(df)
                full_frame_stats = [{'frame': i, 'violations': 0, 'persons': 0, 'safety_equipped': 0} for i in range(full_frame_count)]
                
                html_full_content = generate_html_report_with_graphs(full_violations, full_frame_stats)
                html_full_filename = f"master_violations_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                st.download_button(
                    label="📄 Full HTML",
                    data=html_full_content,
                    file_name=html_full_filename,
                    mime="text/html",
                    use_container_width=True
                )
            
            # Visualizations
            st.markdown("### 📊 Violation Analysis Charts")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Severity distribution
                st.markdown("#### Severity Distribution")
                severity_counts = filtered_df['severity'].value_counts()
                st.bar_chart(severity_counts)
            
            with viz_col2:
                # Violations by date
                st.markdown("#### Violations by Date")
                if 'session_date' in filtered_df.columns:
                    date_counts = filtered_df['session_date'].value_counts().sort_index()
                    st.line_chart(date_counts)
        else:
            st.warning("No violations match the selected filters")
        
        # Stop here for history view
        st.stop()
    
    # Detection confidence threshold (for video analysis)
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.1)
    
    # Additional sidebar information
    st.sidebar.markdown("### Detection Settings")
    st.sidebar.info("Lower threshold detects more objects but may include false positives")
    
    # Email configuration (moved above model training info for easier access)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Email Notification")
    
    send_email = st.sidebar.checkbox(
        "📬 Enable email notifications",
        value=False,
        help="Choose how you want to receive safety violation alerts"
    )
    
    if send_email:
        recipient_email = st.sidebar.text_input(
            "📮 Recipient Email",
            placeholder="recipient@example.com",
            help="Email address to receive the results"
        )
        
        # Use default sender name
        sender_name = "Construction Safety Monitor System"
        
        # Email notification type selection
        st.sidebar.markdown("#### 📬 Notification Type")
        email_mode = st.sidebar.radio(
            "Select notification method:",
            [
                "� Real-time Violation Alerts",
                "📊 Summary Report (CSV)",
                "📄 Summary Report (HTML)"
            ],
            help="Choose when and how you want to receive notifications"
        )
        
        if email_mode == "� Real-time Violation Alerts":
            # Debug: Show that real-time mode is selected
            st.sidebar.success("✅ REAL-TIME MODE ACTIVATED")
            st.sidebar.write(f"DEBUG: Selected mode = '{email_mode}'")
            st.sidebar.info(
                "📝 **Real-time Alert Mode:**\n"
                "✅ Instant violation notifications\n"
                "✅ Image frame with bounding boxes\n"
                "✅ Detailed violation description\n"
                "✅ Frame number and timestamp\n"
                "🚫 NO summary CSV email"
            )
            st.sidebar.warning(
                "⚠️ **HIGH EMAIL VOLUME WARNING:**\n"
                "You will receive one email for EVERY violation detected!\n"
                "A 5-minute video could generate 50+ emails.\n"
                "Consider using Summary Mode for long videos."
            )
            real_time_alerts = True
            send_csv_summary = False
            # Debug confirmation
            st.sidebar.success("✅ REAL-TIME MODE ACTIVATED")
        elif email_mode == "📊 Summary Report (CSV)":
            # Debug: Show that CSV summary mode is selected
            st.sidebar.success("✅ CSV SUMMARY MODE ACTIVATED")
            st.sidebar.write(f"DEBUG: Selected mode = '{email_mode}'")
            st.sidebar.info(
                "📝 **CSV Summary Report Mode:**\n"
                "✅ Complete CSV data file\n"
                "✅ All violations with details\n"
                "✅ Statistical summary\n"
                "✅ Single email after processing\n"
                "🚫 NO individual violation emails"
            )
            real_time_alerts = False
            send_csv_summary = True
            # Debug confirmation
            st.sidebar.success("✅ CSV SUMMARY MODE ACTIVATED")
        else:  # HTML Summary Report
            # Debug: Show that HTML summary mode is selected
            st.sidebar.success("✅ HTML SUMMARY MODE ACTIVATED")
            st.sidebar.write(f"DEBUG: Selected mode = '{email_mode}'")
            st.sidebar.info(
                "📝 **HTML Summary Report Mode:**\n"
                "✅ Visual HTML report with graphs\n"
                "✅ Interactive data table\n"
                "✅ Embedded violation charts\n"
                "✅ Filter & search capabilities\n"
                "✅ Single email after processing\n"
                "🚫 NO individual violation emails"
            )
            real_time_alerts = False
            send_csv_summary = "html"  # Special flag for HTML report
            # Debug confirmation
            st.sidebar.success("✅ HTML SUMMARY MODE ACTIVATED")
        
        st.sidebar.markdown("---")
        st.sidebar.info(
            "🔧 **Gmail SMTP Service:**\n"
            "✅ Direct Gmail integration\n"
            "✅ Secure SMTP connection\n"
            "✅ Reliable delivery\n"
            "✅ **NEW: Smart Email Queue System**\n"
            "   📧 Non-blocking email processing\n"
            "   🚀 Smooth video processing\n"
            "   📊 Real-time queue status\n"
            "   ⏳ Waits for all emails to complete\n"
            "   🎯 Zero email loss guarantee"
        )
    else:
        recipient_email = sender_name = None
        real_time_alerts = False
        send_csv_summary = False

    if "Custom Trained" in selected_model:
        st.sidebar.markdown("### 🚀 Model Training Info")
        st.sidebar.success("""
        **Custom Model Details:**
        - ✅ Trained for 100 epochs
        - ✅ Optimized for construction site safety
        - ✅ All 10 classes included:
          - Safety equipment detection
          - Violation identification  
          - Person and vehicle recognition
        """)
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("�‍💻 Developer Mode", help="Show detailed technical information for developers (model metrics, detection details, processing stats)")
    
    
    # Vehicle filtering settings - Expandable
    with st.sidebar.expander("🚗 Enhanced Vehicle Filtering"):
        st.markdown("""
        **Multi-Layer False Positive Prevention:**
        - ✅ Ultra-low overlap thresholds (0.01 IoU)
        - ✅ 40% expanded filtering zones around vehicles
        - ✅ 60% super-expanded containment checks
        - ✅ Size-based filtering (max 200×400px)
        - ✅ High confidence thresholds (0.6+)
        - ✅ Complete containment detection
        
        **Reduces:** Vehicle/machinery misclassified as violations
        """)
    
    # Safety requirements - Expandable
    with st.sidebar.expander("Safety Requirements"):
        st.markdown("""
        **Required PPE:**
        - ✅ Hard Hat (High Priority)
        - ✅ Safety Vest (Medium Priority)  
        - ✅ Mask (Low Priority)
        
        **Alert Generation:**
        - Person detected WITHOUT required PPE = Violation
        - Checks for equipment within person's vicinity
        """)
    
    # Color legend - Expandable
    with st.sidebar.expander("🎨 Detection Colors"):
        st.markdown("""
        **Bounding Box Colors:**
        - 🟦 **Cyan**: Workers/People
        - 🟩 **Green**: Safety Equipment  
        - 🟥 **Red**: Safety Violations
        - 🟪 **Deep Pink**: Vehicles (thicker boxes)
        - 🟧 **Dark Orange**: Machinery (thicker boxes)
        - ⬜ **White**: Safety Cones
        
        **Note:** Vehicles & machinery have thicker borders for visibility
        """)
    
    # Video upload
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        # Video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = total_frames / fps if fps > 0 else 0
        
        st.success(f"Video loaded: {total_frames} frames, {fps} FPS, {duration:.1f}s duration")
        
        # Processing options
        process_option = st.radio(
            "Select Processing Mode:",
            ["Live Processing (Frame by Frame)", "Full Video Analysis"],
            help="Live processing shows results in real-time. Full analysis processes entire video first."
        )
        
        if st.button("🚀 Start Processing", type="primary"):
            # Developer Mode: Show technical configuration before processing
            if debug_mode:
                st.info("**👨‍💻 Developer Mode Enabled** - Technical details will be shown during processing")
                with st.expander("� Configuration Details"):
                    st.json({
                        "email_config": {
                            "real_time_alerts": real_time_alerts,
                            "send_csv_summary": send_csv_summary,
                            "recipient_email": recipient_email if recipient_email else "Not configured"
                        },
                        "model_config": {
                            "model_path": model_path,
                            "confidence_threshold": confidence_threshold,
                            "processing_mode": process_option
                        },
                        "video_info": {
                            "total_frames": total_frames,
                            "fps": fps,
                            "duration_seconds": round(duration, 2)
                        }
                    })
            
            if process_option == "Live Processing (Frame by Frame)":
                # Live processing with real-time updates
                process_live_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                                 recipient_email, "Construction Safety Monitor System", real_time_alerts, send_csv_summary)
            else:
                # Full video processing
                process_full_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                                 recipient_email, "Construction Safety Monitor System", real_time_alerts, send_csv_summary)
    
    # Information section
    st.markdown("---")
    st.markdown("### 📋 How Detection Works")
    
    st.markdown("""
    **Detection Process:**
    1. **Detect People**: Identifies all workers in the construction site
    2. **Detect Safety Equipment**: Finds hard hats, safety vests, and masks
    3. **Detect Vehicles/Machinery**: Identifies vehicles and heavy equipment
    4. **Advanced Filtering**: Filters out vehicle false positives using expanded zones
    5. **Analyze Proximity**: Checks if detected equipment is near each person
    6. **Generate Alerts**: Creates violations for missing required PPE
    
    **Violation Priorities:**
    - 🔴 **High Priority**: Person without Hard Hat
    - 🟡 **Medium Priority**: Person without Safety Vest
    - 🟢 **Low Priority**: Person without Mask
    
    **Visual Elements:**
    - **Workers** (Cyan boxes): Detected persons requiring PPE
    - **Safety Equipment** (Green boxes): Hard hats, vests, masks
    - **Safety Violations** (Red boxes): Workers missing PPE
    - **Vehicles** (Deep Pink boxes): Cars, trucks, construction vehicles
    - **Machinery** (Orange boxes): Heavy equipment, excavators, bulldozers
    """)

def process_live_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                      recipient_email=None, sender_name=None, 
                      real_time_alerts=False, send_csv_summary=False):
    """Process video with live frame-by-frame updates"""
    
    # Start email queue worker if real-time alerts are enabled
    if real_time_alerts and recipient_email:
        email_queue.start_worker()
        st.info("📧 Email queue worker started for real-time alerts")
    
    # Confirm which email mode is active
    if real_time_alerts and recipient_email:
        st.info(f"🚨 **REAL-TIME ALERTS ACTIVE**: Will send individual emails to {recipient_email} for each violation during processing")
    elif send_csv_summary and recipient_email:
        st.info(f"📊 **SUMMARY MODE ACTIVE**: Will send CSV report to {recipient_email} after processing completes")
    
    # Create compact dashboard layout - everything in single view
    # Top row: Video (60%) + Alerts (40%)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### 📹 Live Video")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🚨 Live Alerts")
        alerts_placeholder = st.empty()
    
    # Bottom row: Compact statistics (full width)
    st.markdown("### 📊 Live Statistics")
    stats_placeholder = st.empty()
    
    # Expandable section for detailed analytics
    detailed_stats_placeholder = st.empty()
    
    # Initialize tracking variables
    total_violations = []
    frame_stats = []
    current_frame = 0
    email_sent_count = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Email status tracking and mode confirmation
    if real_time_alerts and recipient_email:
        email_status_placeholder = st.empty()
        email_status_placeholder.info("� Real-time email alerts enabled - sending individual violation emails during processing")
        st.success("✅ EMAIL MODE: Real-time Violation Alerts (Individual emails per violation)")
    elif send_csv_summary and recipient_email:
        st.success("✅ EMAIL MODE: Summary Report (Single CSV email after processing)")
    elif recipient_email:
        st.info("📧 Email configured but no notification mode selected")
    else:
        email_status_placeholder = None
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Process frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        timestamp = f"{current_frame / fps:.1f}s" if fps > 0 else None
        
        # Process frame (always returns filtered_count for developer mode)
        annotated_frame, violations, person_count, safety_equipped, filtered_count = process_video_frame(
            frame, model, class_names, colors, debug_mode
        )
        
        # Send real-time email alerts if enabled and violations detected
        if real_time_alerts and violations and recipient_email:
            try:
                success, message = send_realtime_violation_alert(
                    annotated_frame, violations, current_frame, 
                    recipient_email, sender_name, timestamp
                )
                if success:
                    email_sent_count += 1
                    if email_status_placeholder:
                        email_status_placeholder.success(f"📧 {email_sent_count} real-time alerts sent | Last: Frame {current_frame}")
                else:
                    if email_status_placeholder:
                        email_status_placeholder.warning(f"⚠️ Email alert failed: {message}")
            except Exception as e:
                if email_status_placeholder:
                    email_status_placeholder.error(f"❌ Email error: {str(e)}")
        
        # Add frame numbers to violations for tracking
        for violation in violations:
            violation['frame'] = current_frame
            violation['timestamp'] = timestamp
            total_violations.append(violation)
        
        # Track frame statistics
        frame_stats.append({
            'frame': current_frame,
            'persons': person_count,
            'violations': len(violations),
            'safety_equipped': safety_equipped
        })
        
        # Update live displays
        update_live_alerts(alerts_placeholder, total_violations, current_frame)
        update_live_stats(stats_placeholder, total_violations, frame_stats, current_frame, 
                          show_email_queue=real_time_alerts and recipient_email)
        update_detailed_stats(detailed_stats_placeholder, total_violations, frame_stats, current_frame)
        
        # Show developer information if enabled
        if debug_mode:
            with st.expander(f"👨‍💻 Developer Info - Frame {current_frame}", expanded=False):
                dev_col1, dev_col2, dev_col3 = st.columns(3)
                
                with dev_col1:
                    st.markdown("**🎯 Detection Stats**")
                    st.metric("Persons Detected", person_count)
                    st.metric("Violations Found", len(violations))
                    st.metric("Safety Equipped", safety_equipped)
                    st.metric("Filtered Objects", filtered_count)
                
                with dev_col2:
                    st.markdown("**⚙️ Processing Info**")
                    st.metric("Current Frame", current_frame)
                    st.metric("Total Frames", total_frames)
                    st.metric("Progress", f"{(current_frame/total_frames*100):.1f}%")
                    st.metric("Timestamp", f"{timestamp}" if timestamp else "N/A")
                
                with dev_col3:
                    st.markdown("**📊 Model Metrics**")
                    st.metric("FPS", f"{fps:.1f}")
                    st.metric("Confidence Threshold", confidence_threshold)
                    total_detections = person_count + len(violations) + safety_equipped
                    st.metric("Total Detections", total_detections)
                    st.metric("Email Alerts Sent", email_sent_count)
                
                # Violation details if any
                if violations:
                    st.markdown("**🚨 Current Frame Violations:**")
                    for i, v in enumerate(violations, 1):
                        st.text(f"{i}. {v['type']} | Severity: {v['severity']} | Location: {v.get('location', 'N/A')}")
        
        # Convert frame for display
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
        
        # Update progress
        progress = current_frame / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {current_frame}/{total_frames} | Violations: {len(total_violations)}")
        
        # Small delay for smoother updates
        time.sleep(0.1)
    
    cap.release()
    
    # Stop email queue worker - wait for all emails to be processed
    if real_time_alerts and recipient_email:
        queue_status = email_queue.get_status()
        if queue_status['queue_size'] > 0 or queue_status['is_processing']:
            # Create progress indicator for email queue completion
            queue_progress_placeholder = st.empty()
            queue_info_placeholder = st.empty()
            
            with queue_info_placeholder:
                st.info(f"📧 Processing {queue_status['queue_size']} remaining emails in queue...")
            
            # Wait for queue to be empty and not processing
            while True:
                queue_status = email_queue.get_status()
                
                with queue_progress_placeholder:
                    if queue_status['is_processing']:
                        st.info(f"📬 Currently sending: {queue_status['current_email']} | Queue: {queue_status['queue_size']} remaining")
                    else:
                        st.info(f"⏳ Queue: {queue_status['queue_size']} emails remaining")
                
                if queue_status['queue_size'] == 0 and not queue_status['is_processing']:
                    break
                time.sleep(0.5)  # Check every 0.5 seconds
            
            # Clear progress indicators
            queue_progress_placeholder.empty()
            queue_info_placeholder.empty()
                
        email_queue.stop_worker()
        final_status = email_queue.get_status()
        st.success(f"✅ All emails processed! Final stats: {final_status['sent_count']} sent, {final_status['failed_count']} failed")
    
    # Final summary
    st.success(f"✅ Processing complete! Total violations detected: {len(total_violations)}")
    
    if real_time_alerts and email_sent_count > 0:
        st.info(f"📧 Sent {email_sent_count} real-time violation alerts during processing")
    
    # Show final results
    if total_violations:
        st.markdown("### 📋 Final Violation Summary")
        df = pd.DataFrame(total_violations)
        display_dataframe(df, use_container_width=True)
        
        # Download results - CSV and Detailed Analytics
        st.markdown("### 📥 Download Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            filename = f"live_safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="� Download Violations Data (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        
        with col3:
            # Generate HTML report with graphs
            with st.spinner("Generating visual report..."):
                html_report = generate_html_report_with_graphs(total_violations, frame_stats)
                html_filename = f"safety_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            st.download_button(
                label="🎨 Download Visual Report (HTML)",
                data=html_report,
                file_name=html_filename,
                mime="text/html",
                use_container_width=True,
                help="Interactive HTML report with embedded graphs"
            )
        
        with col2:
            # Generate detailed analytics report
            detailed_report = generate_detailed_analytics_report(total_violations, frame_stats)
            report_filename = f"detailed_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                label="📈 Download Detailed Analytics Report (TXT)",
                data=detailed_report,
                file_name=report_filename,
                mime="text/plain",
                use_container_width=True
            )
        
        # Append violations to master CSV
        st.markdown("### 💾 Saving to History")
        with st.spinner("Saving violations to master history..."):
            success, master_message = append_to_master_csv(total_violations, "Live Video Analysis")
            if success:
                st.success(f"✅ {master_message}")
            else:
                st.warning(f"⚠️ {master_message}")
        
        # Send summary email ONLY if in summary mode (not real-time mode)
        if send_csv_summary and recipient_email and not real_time_alerts:
            if not sender_name:
                sender_name = "Safety Monitor"
            
            if send_csv_summary == "html":
                # Send HTML report
                with st.spinner("📧 Generating and sending HTML report..."):
                    html_report = generate_html_report_with_graphs(total_violations, frame_stats)
                    success, message = send_email_with_html_smtp(
                        html_report, recipient_email, sender_name, "Live Processing"
                    )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {message}")
            else:
                # Send CSV report
                with st.spinner("📧 Sending CSV summary report..."):
                    success, message = send_email_with_csv_smtp(
                        csv, filename, recipient_email, sender_name, "Live Processing"
                    )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {message}")
        elif send_csv_summary and not recipient_email:
            st.warning("⚠️ Please enter recipient email address to send summary report.")
        elif real_time_alerts and recipient_email:
            st.info("📧 Real-time violation alerts were sent during processing. No summary email needed.")

def process_full_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                      recipient_email=None, sender_name=None,
                      real_time_alerts=False, send_csv_summary=False):
    """Process entire video and show results"""
    st.markdown("### 🎬 Full Video Analysis")
    
    # Start email queue worker if real-time alerts are enabled
    if real_time_alerts and recipient_email:
        email_queue.start_worker()
        st.info("📧 Email queue worker started for real-time alerts")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize tracking
    all_violations = []
    frame_stats = []
    current_frame = 0
    email_sent_count = 0
    
    # Email status tracking and mode confirmation
    if real_time_alerts and recipient_email:
        email_status_placeholder = st.empty()
        email_status_placeholder.info("� Real-time email alerts enabled - sending individual violation emails during processing")
        st.success("✅ EMAIL MODE: Real-time Violation Alerts (Individual emails per violation)")
    elif send_csv_summary and recipient_email:
        if send_csv_summary == "html":
            st.success("✅ EMAIL MODE: HTML Summary Report (Single HTML email after processing)")
        else:
            st.success("✅ EMAIL MODE: CSV Summary Report (Single CSV email after processing)")
    elif recipient_email:
        st.info("📧 Email configured but no notification mode selected")
    else:
        email_status_placeholder = None
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    status_text.text("🔄 Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        timestamp = f"{current_frame / fps:.1f}s" if fps > 0 else None
        
        # Process frame (always returns filtered_count for developer mode)
        annotated_frame, violations, person_count, safety_equipped, filtered_count = process_video_frame(
            frame, model, class_names, colors, debug_mode
        )
        
        # Send real-time email alerts if enabled and violations detected
        if real_time_alerts and violations and recipient_email:
            try:
                success, message = send_realtime_violation_alert(
                    annotated_frame, violations, current_frame, 
                    recipient_email, sender_name, timestamp
                )
                if success:
                    email_sent_count += 1
                    if email_status_placeholder:
                        email_status_placeholder.success(f"📧 {email_sent_count} real-time alerts sent | Last: Frame {current_frame}")
                else:
                    if email_status_placeholder:
                        email_status_placeholder.warning(f"⚠️ Email alert failed: {message}")
            except Exception as e:
                if email_status_placeholder:
                    email_status_placeholder.error(f"❌ Email error: {str(e)}")
        
        # Add frame info to violations
        for violation in violations:
            violation['frame'] = current_frame
            violation['timestamp'] = timestamp
            all_violations.append(violation)
        
        # Track stats
        frame_stats.append({
            'frame': current_frame,
            'persons': person_count,
            'violations': len(violations),
            'safety_equipped': safety_equipped
        })
        
        # Update progress
        if current_frame % 10 == 0:  # Update every 10 frames
            progress = current_frame / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {current_frame}/{total_frames}")
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text("✅ Processing complete!")
    
    # Stop email queue worker - wait for all emails to be processed
    if real_time_alerts and recipient_email:
        queue_status = email_queue.get_status()
        if queue_status['queue_size'] > 0 or queue_status['is_processing']:
            # Create progress indicator for email queue completion
            queue_progress_placeholder = st.empty()
            queue_info_placeholder = st.empty()
            
            with queue_info_placeholder:
                st.info(f"📧 Processing {queue_status['queue_size']} remaining emails in queue...")
            
            # Wait for queue to be empty and not processing
            while True:
                queue_status = email_queue.get_status()
                
                with queue_progress_placeholder:
                    if queue_status['is_processing']:
                        st.info(f"📬 Currently sending: {queue_status['current_email']} | Queue: {queue_status['queue_size']} remaining")
                    else:
                        st.info(f"⏳ Queue: {queue_status['queue_size']} emails remaining")
                
                if queue_status['queue_size'] == 0 and not queue_status['is_processing']:
                    break
                time.sleep(0.5)  # Check every 0.5 seconds
            
            # Clear progress indicators
            queue_progress_placeholder.empty()
            queue_info_placeholder.empty()
                
        email_queue.stop_worker()
        final_status = email_queue.get_status()
        st.success(f"✅ All emails processed! Final stats: {final_status['sent_count']} sent, {final_status['failed_count']} failed")
    
    if real_time_alerts and email_sent_count > 0:
        st.info(f"📧 Sent {email_sent_count} real-time violation alerts during processing")
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Violations", len(all_violations))
        st.metric("Total Persons Detected", sum(f['persons'] for f in frame_stats))
    
    with col2:
        high_violations = len([v for v in all_violations if v['severity'] == 'High'])
        medium_violations = len([v for v in all_violations if v['severity'] == 'Medium'])
        low_violations = len([v for v in all_violations if v['severity'] == 'Low'])
        
        st.metric("High Priority Violations", high_violations)
        st.metric("Medium Priority Violations", medium_violations)
        st.metric("Low Priority Violations", low_violations)
    
    # Detailed results
    if all_violations:
        st.markdown("### 📊 Violation Details")
        df = pd.DataFrame(all_violations)
        display_dataframe(df, use_container_width=True)
        
        # Download results - CSV and Detailed Analytics
        st.markdown("### 📥 Download Reports")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv = df.to_csv(index=False)
            filename = f"safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            st.download_button(
                label="� Download Violations Data (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
        
        
        with col3:
            # Generate HTML report with graphs
            with st.spinner("Generating visual report..."):
                html_report = generate_html_report_with_graphs(all_violations, frame_stats)
                html_filename = f"safety_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            st.download_button(
                label="🎨 Download Visual Report (HTML)",
                data=html_report,
                file_name=html_filename,
                mime="text/html",
                use_container_width=True,
                help="Interactive HTML report with embedded graphs"
            )
        
        with col2:
            # Generate detailed analytics report
            detailed_report = generate_detailed_analytics_report(all_violations, frame_stats)
            report_filename = f"detailed_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            st.download_button(
                label="📈 Download Detailed Analytics Report (TXT)",
                data=detailed_report,
                file_name=report_filename,
                mime="text/plain",
                use_container_width=True
            )
        
        # Append violations to master CSV
        st.markdown("### 💾 Saving to History")
        with st.spinner("Saving violations to master history..."):
            success, master_message = append_to_master_csv(all_violations, "Full Video Analysis")
            if success:
                st.success(f"✅ {master_message}")
            else:
                st.warning(f"⚠️ {master_message}")
        
        # Send summary email ONLY if in summary mode (not real-time mode)
        if send_csv_summary and recipient_email and not real_time_alerts:
            if not sender_name:
                sender_name = "Safety Monitor"
            
            if send_csv_summary == "html":
                # Send HTML report
                with st.spinner("📧 Generating and sending HTML report..."):
                    success, message = send_email_with_html_smtp(
                        html_report, recipient_email, sender_name, "Full Video Analysis"
                    )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {message}")
            else:
                # Send CSV report
                with st.spinner("📧 Sending CSV summary report..."):
                    success, message = send_email_with_csv_smtp(
                        csv, filename, recipient_email, sender_name, "Full Video Analysis"
                    )
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.warning(f"⚠️ {message}")
        elif send_csv_summary and not recipient_email:
            st.warning("⚠️ Please enter recipient email address to send summary report.")
        elif real_time_alerts and recipient_email:
            st.info("📧 Real-time violation alerts were sent during processing. No summary email needed.")
    else:
        st.success("🎉 No safety violations detected in the video!")

if __name__ == "__main__":
    main()