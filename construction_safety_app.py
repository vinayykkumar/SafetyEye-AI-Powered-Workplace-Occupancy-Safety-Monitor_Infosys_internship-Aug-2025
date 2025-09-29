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

    if debug_mode:
        return violations, person_count, safety_equipped_count, filtered_count, debug_info
    else:
        return violations, person_count, safety_equipped_count

def draw_detections(frame, detections, class_names, colors, debug_mode=False):
    """Draw bounding boxes and labels on frame"""
    result = analyze_safety_violations(detections, class_names, debug_mode)
    
    if debug_mode:
        violations, person_count, safety_equipped, filtered_count, debug_info = result
    else:
        violations, person_count, safety_equipped = result
        filtered_count, debug_info = 0, []
    
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
    
    if debug_mode:
        return frame, violations, person_count, safety_equipped, filtered_count, debug_info
    else:
        return frame, violations, person_count, safety_equipped

def process_video_frame(frame, model, class_names, colors, debug_mode=False):
    """Process a single video frame"""
    results = model(frame)
    
    if len(results) > 0 and len(results[0].boxes) > 0:
        detections = results[0].boxes.data.cpu().numpy()
        result = draw_detections(frame, detections, class_names, colors, debug_mode)
        
        if debug_mode:
            annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info = result
        else:
            annotated_frame, violations, person_count, safety_equipped = result
            filtered_count, debug_info = 0, []
    else:
        annotated_frame = frame
        violations = []
        person_count = 0
        safety_equipped = 0
        filtered_count, debug_info = 0, []
    
    if debug_mode:
        return annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info
    else:
        return annotated_frame, violations, person_count, safety_equipped

def update_live_alerts(placeholder, violations, current_frame):
    """Update live alerts display using Streamlit components"""
    if not violations:
        placeholder.success("✅ No recent safety violations detected")
        return
    
    # Get recent violations (last 5)
    recent_violations = violations[-5:] if len(violations) > 5 else violations
    
    # Create container for alerts
    with placeholder.container():
        st.markdown("### 🚨 Live Safety Alerts")
        
        for violation in recent_violations:
            # Use Streamlit's native alert components
            if violation['severity'] == 'High':
                st.error(f"🚨 **{violation['type']} Violation** - Frame {violation['frame']} | **HIGH SEVERITY**")
            elif violation['severity'] == 'Medium':
                st.warning(f"⚠️ **{violation['type']} Violation** - Frame {violation['frame']} | **MEDIUM SEVERITY**")
            else:  # Low
                st.info(f"ℹ️ **{violation['type']} Violation** - Frame {violation['frame']} | **LOW SEVERITY**")

def update_live_stats(placeholder, total_violations, frame_stats, current_frame, show_email_queue=False):
    """Update live statistics display using Streamlit components with optional email queue status"""
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
    
    # Display with Streamlit components
    with placeholder.container():
        # Alert level
        if high_violations > 5:
            st.error("🔴 HIGH RISK")
        elif medium_violations > 3:
            st.warning("🟡 MEDIUM RISK")
        else:
            st.success("🟢 LOW RISK")
        
        st.markdown("### 📊 Live Statistics")
        
        # Main metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Violations", total_violations_count)
        with col2:
            st.metric("Current Frame", current_frame)
        
        # Email queue status (if enabled)
        if show_email_queue:
            st.markdown("#### 📧 Email Queue Status")
            queue_status = email_queue.get_status()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Queued", queue_status['queue_size'])
            with col2:
                st.metric("Sent", queue_status['sent_count'], delta=None)
            with col3:
                st.metric("Failed", queue_status['failed_count'], delta=None)
            
            # Processing status
            if queue_status['is_processing']:
                st.info(f"📤 Currently processing: {queue_status['current_email']}")
            elif queue_status['queue_size'] > 0:
                st.warning(f"⏳ {queue_status['queue_size']} emails waiting in queue")
            else:
                st.success("✅ Email queue is empty")
            
            # Recent email logs
            if queue_status['recent_logs']:
                st.markdown("**Recent Email Activity:**")
                for log in queue_status['recent_logs']:
                    st.text(log)
        
        # Violation breakdown
        st.markdown("#### Violation Breakdown")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("HIGH", high_violations, delta=None)
        with col2:
            st.metric("MEDIUM", medium_violations, delta=None)
        with col3:
            st.metric("LOW", low_violations, delta=None)
        
        # Additional stats
        st.markdown("#### Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Workers/Frame", f"{avg_persons:.1f}")
        with col2:
            st.metric("Violation Rate", f"{violation_rate:.1f}%")
        
        st.info(f"📈 {total_persons} total workers processed")

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

def main():
    st.title("🚧 Construction Site Safety Monitor")
    st.markdown("Upload a video to monitor construction site safety and detect PPE violations")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Model selection
    model_options = {
        "Custom Trained Model (Recommended)": "models/construction_best.pt",
        "YOLOv8n (Base Model)": "models/yolov8n.pt"
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)
    model_path = model_options[selected_model]
    
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
    
    # Detection confidence threshold
    confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.4, 0.1)
    
    # Additional sidebar information
    st.sidebar.markdown("### Detection Settings")
    st.sidebar.info("Lower threshold detects more objects but may include false positives")
    
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
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show filtered detections and debug information")
    
    # Email configuration
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
                "📊 Summary Report (CSV)"
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
        else:
            # Debug: Show that summary mode is selected
            st.sidebar.success("✅ SUMMARY MODE ACTIVATED")
            st.sidebar.write(f"DEBUG: Selected mode = '{email_mode}'")
            st.sidebar.info(
                "📝 **Summary Report Mode:**\n"
                "✅ Complete analysis report\n"
                "✅ CSV file with all violations\n"
                "✅ Statistical summary\n"
                "✅ Single email after processing\n"
                "🚫 NO individual violation emails"
            )
            real_time_alerts = False
            send_csv_summary = True
            # Debug confirmation
            st.sidebar.success("✅ SUMMARY MODE ACTIVATED")
        
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
    
    # Vehicle filtering settings
    st.sidebar.markdown("### 🚗 Enhanced Vehicle Filtering")
    st.sidebar.markdown("""
    **Multi-Layer False Positive Prevention:**
    - ✅ Ultra-low overlap thresholds (0.01 IoU)
    - ✅ 40% expanded filtering zones around vehicles
    - ✅ 60% super-expanded containment checks
    - ✅ Size-based filtering (max 200×400px)
    - ✅ High confidence thresholds (0.6+)
    - ✅ Complete containment detection
    
    **Reduces:** Vehicle/machinery misclassified as violations
    """)
    
    # Safety requirements
    st.sidebar.markdown("### Safety Requirements")
    st.sidebar.markdown("""
    **Required PPE:**
    - ✅ Hard Hat (High Priority)
    - ✅ Safety Vest (Medium Priority)  
    - ✅ Mask (Low Priority)
    
    **Alert Generation:**
    - Person detected WITHOUT required PPE = Violation
    - Checks for equipment within person's vicinity
    """)
    
    # Color legend
    st.sidebar.markdown("### 🎨 Detection Colors")
    st.sidebar.markdown("""
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
            # Debug: Show final email configuration before processing
            st.write(f"🔍 **DEBUG INFO:**")
            st.write(f"- real_time_alerts = {real_time_alerts}")
            st.write(f"- send_csv_summary = {send_csv_summary}")
            st.write(f"- recipient_email = {recipient_email}")
            
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
    
    # Create layout columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Processing")
        video_placeholder = st.empty()
    
    with col2:
        alerts_placeholder = st.empty()
        stats_placeholder = st.empty()
    
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
        
        # Process frame with debug mode
        result = process_video_frame(frame, model, class_names, colors, debug_mode)
        
        if debug_mode:
            annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info = result
        else:
            annotated_frame, violations, person_count, safety_equipped = result
            filtered_count, debug_info = 0, []
        
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
        
        # Show debug information if enabled
        if debug_mode and debug_info:
            with st.expander(f"🐛 Debug Info - Frame {current_frame} ({filtered_count} detections filtered)"):
                for info in debug_info:
                    st.text(info)
        
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
        
        # Download results
        csv = df.to_csv(index=False)
        filename = f"live_safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        # Send CSV summary email ONLY if in summary mode (not real-time mode)
        if send_csv_summary and recipient_email and not real_time_alerts:
            if not sender_name:
                sender_name = "Safety Monitor"
                
            with st.spinner("📧 Sending summary report..."):
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
        st.success("✅ EMAIL MODE: Summary Report (Single CSV email after processing)")
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
        
        # Process frame with debug mode
        result = process_video_frame(frame, model, class_names, colors, debug_mode)
        
        if debug_mode:
            annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info = result
        else:
            annotated_frame, violations, person_count, safety_equipped = result
            filtered_count, debug_info = 0, []
        
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
        
        # Download results
        csv = df.to_csv(index=False)
        filename = f"safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        # Send CSV summary email ONLY if in summary mode (not real-time mode)
        if send_csv_summary and recipient_email and not real_time_alerts:
            if not sender_name:
                sender_name = "Safety Monitor"
                
            with st.spinner("📧 Sending summary report..."):
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