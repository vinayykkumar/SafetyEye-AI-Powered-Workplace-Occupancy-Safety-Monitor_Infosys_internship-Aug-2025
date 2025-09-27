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
        
        if confidence > 0.4:  # Lower confidence threshold to catch more detections
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
                no_hardhat_boxes.append(box)
            elif class_name == "NO-Safety Vest":
                no_vest_boxes.append(box)
            elif class_name == "NO-Mask":
                no_mask_boxes.append(box)
    
    # For each detected person, check if they have safety equipment nearby
    # Only check safety equipment for PERSONS, not vehicles or machinery
    for person_box in person_boxes:
        # Make sure this person is not overlapping with vehicles or machinery
        is_near_vehicle = False
        
        # Check if person is near/inside a vehicle or machinery
        for vehicle_box in vehicle_boxes:
            if boxes_overlap(person_box, vehicle_box, threshold=0.3):
                is_near_vehicle = True
                break
                
        for machinery_box in machinery_boxes:
            if boxes_overlap(person_box, machinery_box, threshold=0.3):
                is_near_vehicle = True
                break
        
        # Skip safety equipment checks for people in/near vehicles or machinery
        if is_near_vehicle:
            continue
            
        person_has_hardhat = False
        person_has_vest = False
        person_has_mask = False
        
        # Check if person has hardhat nearby
        for hardhat_box in hardhat_boxes:
            if boxes_overlap(person_box, hardhat_box, threshold=0.1):
                person_has_hardhat = True
                break
                
        # Check if person has safety vest nearby  
        for vest_box in vest_boxes:
            if boxes_overlap(person_box, vest_box, threshold=0.1):
                person_has_vest = True
                break
                
        # Check if person has mask nearby
        for mask_box in mask_boxes:
            if boxes_overlap(person_box, mask_box, threshold=0.1):
                person_has_mask = True
                break
        
        # Generate violations for missing equipment
        if not person_has_hardhat:
            violations.append({
                'type': 'Person Missing Hard Hat',
                'severity': 'High',
                'location': f"({int(person_box[0])}, {int(person_box[1])})"
            })
        
        if not person_has_vest:
            violations.append({
                'type': 'Person Missing Safety Vest', 
                'severity': 'Medium',
                'location': f"({int(person_box[0])}, {int(person_box[1])})"
            })
            
        if not person_has_mask:
            violations.append({
                'type': 'Person Missing Mask',
                'severity': 'Low', 
                'location': f"({int(person_box[0])}, {int(person_box[1])})"
            })
        
        # Count as safety equipped if they have at least hardhat and vest
        if person_has_hardhat and person_has_vest:
            safety_equipped_count += 1
    
    # Check direct violations (NO-Hardhat, NO-Safety Vest, NO-Mask) with enhanced vehicle/machinery filtering
    for detection in detections:
        class_id = int(detection[5])
        class_name = class_names.get(class_id, f"Unknown_{class_id}")
        confidence = detection[4]
        x1, y1, x2, y2 = detection[:4]
        box = (x1, y1, x2, y2)
        
        # Only process NO-equipment detections with high confidence to reduce false positives
        if confidence > 0.6 and class_name.startswith("NO-"):  # Higher confidence threshold for violation detections
            is_near_vehicle = is_violation_near_vehicle_machinery(box, vehicle_boxes, machinery_boxes)
            is_too_large = is_detection_too_large_for_person(box)
            
            # Debug information
            if debug_mode:
                width, height = x2 - x1, y2 - y1
                debug_info.append(f"{class_name} (conf: {confidence:.2f}, size: {int(width)}x{int(height)}) - "
                                f"Near vehicle: {is_near_vehicle}, Too large: {is_too_large}")
            
            if class_name == "NO-Hardhat":
                # Enhanced filtering: check vehicle/machinery proximity AND box size
                if not is_near_vehicle and not is_too_large:
                    violations.append({
                        'type': 'Worker Without Hard Hat Detected',
                        'severity': 'High',
                        'location': f"({int(x1)}, {int(y1)})"
                    })
                else:
                    filtered_count += 1
                    
            elif class_name == "NO-Safety Vest":
                # Enhanced filtering: check vehicle/machinery proximity AND box size
                if not is_near_vehicle and not is_too_large:
                    violations.append({
                        'type': 'Worker Without Safety Vest Detected',
                        'severity': 'Medium',
                        'location': f"({int(x1)}, {int(y1)})"
                    })
                else:
                    filtered_count += 1
                    
            elif class_name == "NO-Mask":
                # Enhanced filtering: check vehicle/machinery proximity AND box size
                if not is_near_vehicle and not is_too_large:
                    violations.append({
                        'type': 'Worker Without Mask Detected',
                        'severity': 'Low',
                        'location': f"({int(x1)}, {int(y1)})"
                    })
                else:
                    filtered_count += 1
    
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

def update_live_stats(placeholder, total_violations, frame_stats, current_frame):
    """Update live statistics display using Streamlit components"""
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
        "YOLOv8n": "models/yolov8n.pt",
        "YOLO11n": "models/yolo11n.pt"
    }
    
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    
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
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", help="Show filtered detections and debug information")
    
    # Email configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📧 Email Notification")
    
    send_email = st.sidebar.checkbox(
        "📬 Send results via email",
        value=False,
        help="Automatically send CSV results to specified email address (no login required)"
    )
    
    if send_email:
        recipient_email = st.sidebar.text_input(
            "📮 Recipient Email",
            placeholder="recipient@example.com",
            help="Email address to receive the results"
        )
        
        sender_name = st.sidebar.text_input(
            "👤 Your Name",
            placeholder="Your Name or Company",
            help="Name to appear as sender"
        )
        
        st.sidebar.info(
            "📝 **Gmail SMTP Email Service:**\n"
            "✅ Direct Gmail integration\n"
            "✅ Secure SMTP connection\n"
            "✅ Direct CSV attachment\n"
            "✅ Professional email format\n"
            "✅ Reliable delivery"
        )
    else:
        recipient_email = sender_name = None
    
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
            if process_option == "Live Processing (Frame by Frame)":
                # Live processing with real-time updates
                process_live_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                                 send_email, recipient_email, sender_name)
            else:
                # Full video processing
                process_full_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                                 send_email, recipient_email, sender_name)
    
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
                      send_email=False, recipient_email=None, sender_name=None):
    """Process video with live frame-by-frame updates"""
    
    # Create layout columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📹 Live Video Processing")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📊 Live Dashboard")
        alerts_placeholder = st.empty()
        stats_placeholder = st.empty()
    
    # Initialize tracking variables
    total_violations = []
    frame_stats = []
    current_frame = 0
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Process frame by frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        
        # Process frame with debug mode
        result = process_video_frame(frame, model, class_names, colors, debug_mode)
        
        if debug_mode:
            annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info = result
        else:
            annotated_frame, violations, person_count, safety_equipped = result
            filtered_count, debug_info = 0, []
        
        # Add frame numbers to violations for tracking
        for violation in violations:
            violation['frame'] = current_frame
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
        update_live_stats(stats_placeholder, total_violations, frame_stats, current_frame)
        
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
    
    # Final summary
    st.success(f"✅ Processing complete! Total violations detected: {len(total_violations)}")
    
    # Show final results
    if total_violations:
        st.markdown("### 📋 Final Violation Summary")
        df = pd.DataFrame(total_violations)
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        filename = f"live_safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        # Send email if configured
        if send_email and recipient_email:
            if not sender_name:
                sender_name = "Safety Monitor"
                
            with st.spinner("📧 Sending email..."):
                success, message = send_email_with_csv_smtp(
                    csv, filename, recipient_email, sender_name, "Live Processing"
                )
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
        elif send_email:
            st.warning("⚠️ Please enter recipient email address to send results.")

def process_full_video(video_path, model, class_names, colors, confidence_threshold, debug_mode,
                      send_email=False, recipient_email=None, sender_name=None):
    """Process entire video and show results"""
    st.markdown("### 🎬 Full Video Analysis")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize tracking
    all_violations = []
    frame_stats = []
    current_frame = 0
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    status_text.text("🔄 Processing video frames...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        
        # Process frame with debug mode
        result = process_video_frame(frame, model, class_names, colors, debug_mode)
        
        if debug_mode:
            annotated_frame, violations, person_count, safety_equipped, filtered_count, debug_info = result
        else:
            annotated_frame, violations, person_count, safety_equipped = result
            filtered_count, debug_info = 0, []
        
        # Add frame info to violations
        for violation in violations:
            violation['frame'] = current_frame
            violation['timestamp'] = f"{current_frame / cap.get(cv2.CAP_PROP_FPS):.1f}s"
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
        st.dataframe(df, use_container_width=True)
        
        # Download results
        csv = df.to_csv(index=False)
        filename = f"safety_violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        # Send email if configured
        if send_email and recipient_email:
            if not sender_name:
                sender_name = "Safety Monitor"
                
            with st.spinner("📧 Sending email..."):
                success, message = send_email_with_csv_smtp(
                    csv, filename, recipient_email, sender_name, "Full Video Analysis"
                )
                if success:
                    st.success(f"✅ {message}")
                else:
                    st.warning(f"⚠️ {message}")
        elif send_email:
            st.warning("⚠️ Please enter recipient email address to send results.")
    else:
        st.success("🎉 No safety violations detected in the video!")

if __name__ == "__main__":
    main()