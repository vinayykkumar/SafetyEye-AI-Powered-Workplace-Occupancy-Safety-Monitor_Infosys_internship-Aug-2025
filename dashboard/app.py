# app.py - Refactored SafetyEye Dashboard (full-featured)
# Key: single camera manager + capture thread, detection in capture loop,
# streaming served from latest processed frame, upload uses "video" key.
import os
import sys
import time
import cv2
import csv
import json
import threading
import tempfile
import smtplib
import numpy as np

from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders

from flask import (
    Flask, render_template, Response, jsonify, request,
    send_file, abort, redirect, url_for
)
from werkzeug.utils import secure_filename

# -----------------------
# Basic Flask Setup
# -----------------------
camera = None
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
total_detections_today = 0
current_occupancy_estimate = 0


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = os.path.join(current_dir, 'temp_uploads')
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit for uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Violations folder
violations_dir = os.path.join(parent_dir, 'violations')
os.makedirs(violations_dir, exist_ok=True)

APP_START_TIME = datetime.now()  # 🕒 track when app started
LAST_REPORT_FILE = "last_report.json"  # 💾 store last report timestamp persistently
LAST_REPORT_SENT = None  # 🧭 will be loaded from file on startup
processing_flag = {"active": False}

# -----------------------
# Email Configuration (update these)
# -----------------------
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'your-email@gmail.com',
    'sender_password': 'your-app-password',
    'recipient_email': 'admin@company.com'
}


# =========================
# Helper Functions for Last Report Time
# =========================
def save_last_report_time():
    """Save current time as last report timestamp to file."""
    global LAST_REPORT_SENT
    LAST_REPORT_SENT = datetime.now()
    try:
        with open(LAST_REPORT_FILE, "w") as f:
            json.dump({"last_sent": LAST_REPORT_SENT.strftime("%Y-%m-%d %H:%M:%S")}, f)
        print(f"💾 Saved LAST_REPORT_SENT = {LAST_REPORT_SENT}")
    except Exception as e:
        print(f"❌ Failed to save last report time: {e}")

def load_last_report_time():
    """Load the last report timestamp from file if available."""
    global LAST_REPORT_SENT
    if os.path.exists(LAST_REPORT_FILE):
        try:
            with open(LAST_REPORT_FILE, "r") as f:
                data = json.load(f)
                LAST_REPORT_SENT = datetime.strptime(data["last_sent"], "%Y-%m-%d %H:%M:%S")
                print(f"📂 Loaded LAST_REPORT_SENT = {LAST_REPORT_SENT}")
        except Exception as e:
            print(f"⚠️ Could not load last report time: {e}")
            LAST_REPORT_SENT = None
    else:
        LAST_REPORT_SENT = None

# ✅ Load saved report time immediately on startup
load_last_report_time()


# -----------------------
# Import detection utilities (attempt)
# -----------------------
try:
    sys.path.append(os.path.join(parent_dir, 'predictions'))
    from real_time_ppe_monitor_strict_alerts import (
        iou, center, point_in_box, configure_logger, CentroidTracker
    )
    from ultralytics import YOLO
    DETECTION_UTILS_OK = True
    print("✅ Detection utilities imported")
except Exception as e:
    print(f"❌ Detection utilities import failed: {e}")
    DETECTION_UTILS_OK = False

# -----------------------
# Safety Monitor (Real or Mock)
# -----------------------
class RealSafetyMonitor:
    def __init__(self, model_path=None):
        # Attempt to load model if available
        self.model_loaded = False
        try:
            if DETECTION_UTILS_OK and model_path:
                self.model = YOLO(model_path)
                self.tracker = CentroidTracker(max_disappeared=15, max_distance=100)
                self.model_loaded = True
            else:
                self.model_loaded = False
        except Exception as e:
            print(f"❌ YOLO load failed: {e}")
            self.model_loaded = False

        # counters & config (same defaults as your previous code)
        self.no_helmet_counter = {}
        self.no_vest_counter = {}
        self.last_violation_time = {}
        self.ppe_conf_min = 0.45
        self.person_conf_min = 0.4
        self.head_expand = 0.40
        self.iou_threshold = 0.25
        self.no_ppe_frames_n = 10
        self.alert_cooldown = 8.0

    def process_frame(self, frame):
        """Process a single BGR frame and return (processed_frame, violations_list)."""
        if not self.model_loaded:
            return self._mock_detection(frame), []
        try:
            results = self.model(frame, conf=0.25, verbose=False)
            res = results[0]

            boxes = []; classes = []; confs = []
            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    try:
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cls = int(b.cls[0]); conf = float(b.conf[0])
                    except Exception:
                        arr = b.xyxy
                        x1, y1, x2, y2 = map(int, arr[0])
                        cls = int(b.cls); conf = float(b.conf)
                    boxes.append([x1, y1, x2, y2])
                    classes.append(cls)
                    confs.append(conf)

            # class indices
            worker_idx = [i for i, c in enumerate(classes)
                        if self.model.names[c].lower() in ("worker", "person", "people")]
            helmet_idx = [i for i, c in enumerate(classes)
                        if "hardhat" in self.model.names[c].lower() or "helmet" in self.model.names[c].lower()]
            vest_idx = [i for i, c in enumerate(classes)
                        if "vest" in self.model.names[c].lower()]

            # prepare centroids & person boxes
            centroids = []; person_boxes = []; person_confs = []
            for idx in worker_idx:
                box = boxes[idx]
                centroids.append(center(box))
                person_boxes.append(box)
                person_confs.append(confs[idx])

            # update tracker
            tracked = self.tracker.update(centroids)
            id_map = {}
            for tid, centroid in tracked:
                if len(centroids) == 0:
                    continue
                dists = [np.linalg.norm(np.array(centroid) - np.array(c)) for c in centroids]
                idx_near = int(np.argmin(dists))
                id_map[tid] = (person_boxes[idx_near], person_confs[idx_near])

            violations = []
            current_time = time.time()
            frame_area = frame.shape[0] * frame.shape[1]

            for tid, (pbox, pconf) in id_map.items():
                if pconf < self.person_conf_min:
                    self.no_helmet_counter[tid] = 0
                    self.no_vest_counter[tid] = 0
                    continue

                x1, y1, x2, y2 = pbox
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area < 0.003 * frame_area:
                    self.no_helmet_counter[tid] = 0
                    self.no_vest_counter[tid] = 0
                    continue

                ph = max(1, y2 - y1)
                expanded_pbox = [x1, max(0, int(y1 - self.head_expand * ph)), x2, y2]

                # --- PPE confidence calculation ---
                best_h_conf = 0.0
                for hi in helmet_idx:
                    hbox = boxes[hi]
                    hconf = confs[hi]
                    hcenter = center(hbox)
                    if point_in_box(hcenter, expanded_pbox) and hconf > best_h_conf:
                        best_h_conf = hconf

                best_v_conf = 0.0
                for vi in vest_idx:
                    vbox = boxes[vi]
                    vconf = confs[vi]
                    label = self.model.names[classes[vi]].lower()

                    # Allow both "vest" and "safety" keywords
                    if "vest" in label or "safety" in label:
                        overlap = iou(pbox, vbox)
                        if overlap > 0.05:  # Lowered threshold to 0.05 for testing
                            if vconf > best_v_conf:
                                best_v_conf = vconf



                helmet_present = best_h_conf >= self.ppe_conf_min
                vest_present = best_v_conf >= self.ppe_conf_min

                if not helmet_present:
                    self.no_helmet_counter[tid] = self.no_helmet_counter.get(tid, 0) + 1
                else:
                    self.no_helmet_counter[tid] = 0

                if not vest_present:
                    self.no_vest_counter[tid] = self.no_vest_counter.get(tid, 0) + 1
                else:
                    self.no_vest_counter[tid] = 0

                violation_reasons = []
                if self.no_helmet_counter.get(tid, 0) >= self.no_ppe_frames_n:
                    violation_reasons.append("no_helmet")
                if self.no_vest_counter.get(tid, 0) >= self.no_ppe_frames_n:
                    violation_reasons.append("no_vest")

                if violation_reasons:
                    last_violation = self.last_violation_time.get(tid, 0)
                    if (current_time - last_violation) > self.alert_cooldown:
                        violations.append({
                            "person_id": tid,
                            "frame_conf": pconf,
                            "helmet_conf": best_h_conf,
                            "vest_conf": best_v_conf,
                            "types": violation_reasons
                        })
                        self.last_violation_time[tid] = current_time
                        if "no_helmet" in violation_reasons:
                            self.no_helmet_counter[tid] = 0
                        if "no_vest" in violation_reasons:
                            self.no_vest_counter[tid] = 0


            # Visualization
            processed_frame = frame.copy()
            for i, box in enumerate(boxes):
                cls = classes[i]; label = self.model.names[cls]; conf = confs[i]
                color = (0, 255, 0)
                if i in worker_idx:
                    this_box = box
                    overlap_h = False
                    overlap_v = False
                    for hi in helmet_idx:
                        hcenter = center(boxes[hi])
                        x1, y1, x2, y2 = this_box
                        ph = max(1, y2 - y1)
                        expanded_head = [x1, max(0, int(y1 - self.head_expand * ph)), x2, y2]
                        if point_in_box(hcenter, expanded_head) and confs[hi] >= self.ppe_conf_min:
                            overlap_h = True
                    for vi in vest_idx:
                        if iou(this_box, boxes[vi]) > self.iou_threshold and confs[vi] >= self.ppe_conf_min:
                            overlap_v = True
                    if overlap_h and overlap_v:
                        color = (0, 255, 0)
                    elif overlap_h or overlap_v:
                        color = (0, 165, 255)
                    else:
                        color = (0, 0, 255)
                x1, y1, x2, y2 = box
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(processed_frame, f"{label} {conf:.2f}",
                           (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            return processed_frame, violations
        except Exception as e:
            print(f"❌ Error in real detection: {e}")
            return self._mock_detection(frame), []

    def _mock_detection(self, frame):
        # Draw mock overlays for fallback
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        cv2.putText(frame, "Person: Helmet ✅ Vest ✅", (50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        status = "MOCK DETECTION - Real YOLO system unavailable"
        cv2.putText(frame, status, (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        cv2.putText(frame, f"SafetyEye - {datetime.now().strftime('%H:%M:%S')}",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

# Try to initialize safety_monitor using your trained weights path, if available
MODEL_PATH = os.path.join(parent_dir, 'runs', 'detect', 'yolov8n_ppe_gpu_run_2', 'weights', 'best.pt')
if DETECTION_UTILS_OK and os.path.exists(MODEL_PATH):
    safety_monitor = RealSafetyMonitor(model_path=MODEL_PATH)
else:
    safety_monitor = RealSafetyMonitor(model_path=None)

# -----------------------
# CSV / Violations Management
# -----------------------
def init_violations_system():
    try:
        csv_path = os.path.join(violations_dir, 'violations_log.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "frame", "person_id", "violation_type", "image_path", "helmet_conf", "vest_conf"])

        print("✅ Violations system initialized")
        return True
    except Exception as e:
        print(f"❌ Violations system initialization failed: {e}")
        return False

# -----------------------
# CSV / Violations Management (Improved)
# -----------------------
VIOLATIONS_CSV_PATH = os.path.join(violations_dir, 'violations_log.csv')

def _ensure_violations_csv():
    if not os.path.exists(VIOLATIONS_CSV_PATH):
        with open(VIOLATIONS_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "frame",
                "person_id",
                "violation_type",
                "image_path",
                "helmet_conf",
                "vest_conf"
            ])

def log_violation_to_csv(violation_type, confidence=0.95, image_path="", person_id="0", frame_id="0",
                         helmet_conf=None, vest_conf=None):
    """Append a violation row to CSV with clean formatting and handle multi-type safely."""
    try:
        _ensure_violations_csv()

        # Normalize multi-type violations
        if isinstance(violation_type, (list, tuple)):
            # Join into comma-separated string for display
            violation_type = ", ".join(str(v) for v in violation_type)
        elif isinstance(violation_type, dict):
            # Handle weird YOLO dict input (bug fallback)
            violation_type = violation_type.get('type', 'unknown')

        # Determine confidences properly
        if helmet_conf is None:
            helmet_conf = confidence if "helmet" in str(violation_type).lower() else 0.0
        if vest_conf is None:
            vest_conf = confidence if "vest" in str(violation_type).lower() else 0.0

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(VIOLATIONS_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                frame_id,
                person_id,
                violation_type,
                image_path,
                f"{helmet_conf:.2f}",
                f"{vest_conf:.2f}"
            ])

        print(f"🟢 Logged {violation_type} | Helmet={helmet_conf:.2f} | Vest={vest_conf:.2f}")
        return True

    except Exception as e:
        print(f"❌ Failed to log violation to CSV: {e}")
        return False



def get_violations_from_csv(limit=1000):
    """Read violations safely and return a list of dicts (newest first)."""
    try:
        _ensure_violations_csv()
        violations = []

        with open(VIOLATIONS_CSV_PATH, 'r', newline='') as f:
            reader = list(csv.DictReader(f))
            idx = 0

            for row in reversed(reader):
                if idx >= limit:
                    break

                # 🛑 Skip malformed rows (like when entire dict got written as one string)
                if len(row.keys()) == 1 and list(row.keys())[0].startswith("{"):
                    print(f"⚠️ Skipping malformed CSV row: {row}")
                    continue

                try:
                    ts = (row.get('timestamp') or '').strip()
                    if not ts:
                        continue

                    # Ensure numeric conversions
                    helmet_conf = float(row.get('helmet_conf', 0) or 0)
                    vest_conf = float(row.get('vest_conf', 0) or 0)

                    violations.append({
                        'id': idx + 1,
                        'timestamp': ts,
                        'frame': row.get('frame', ''),
                        'person_id': row.get('person_id', ''),
                        'violation_type': (
                            row.get('violation_type', '') or row.get('reason', '')
                        ),
                        'image_path': row.get('image_path', ''),
                        'helmet_conf': helmet_conf,
                        'vest_conf': vest_conf
                    })
                    idx += 1

                except Exception as inner_err:
                    print(f"⚠️ Skipped malformed row: {inner_err} — {row}")
                    continue

        return violations

    except Exception as e:
        print(f"❌ Error reading violations: {e}")
        return []



def get_violations_by_date_range(start_date, end_date):
    """Return violations within the given date range (inclusive)."""
    try:
        violations = get_violations_from_csv(limit=10000)
        filtered = []
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()

        for violation in violations:
            try:
                v_time = violation.get('timestamp', '')[:19]  # e.g. "2025-10-06 15:45:03"
                v_date = datetime.strptime(v_time, '%Y-%m-%d %H:%M:%S').date()
                if start <= v_date <= end:
                    filtered.append(violation)
            except Exception as inner_e:
                print(f"⚠️ Skipping malformed row: {inner_e}")
                continue

        print(f"✅ Filtered {len(filtered)} violations between {start_date} and {end_date}")
        return filtered
    except Exception as e:
        print(f"❌ Error filtering by date: {e}")
        return []

# -----------------------
# Email Configuration (Persistent)
# -----------------------
import json

EMAIL_CONFIG_FILE = os.path.join(current_dir, 'email_settings.json')

# Default email configuration
email_config = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': '',
    'sender_password': '',
    'recipient_email': ''
}

# ✅ Load saved configuration if available
if os.path.exists(EMAIL_CONFIG_FILE):
    try:
        with open(EMAIL_CONFIG_FILE, 'r') as f:
            saved_config = json.load(f)
            email_config.update(saved_config)
            print(f"📩 Loaded saved email configuration for {email_config.get('sender_email', 'N/A')}")
    except Exception as e:
        print(f"⚠️ Failed to load saved email config: {e}")


# -----------------------
# Email Utility Functions
# -----------------------
# -----------------------
# Email Functions (with image attachment)
# -----------------------
def send_violation_email(violation_data):
    """
    Sends an email alert with violation details and attaches the snapshot image.
    """
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = f"🚨 SafetyEye Alert: {violation_data.get('violation_type', '').upper()}"

        # --- Email Body ---
        body = f"""
        ⚠️ SafetyEye Violation Alert ⚠️

        Violation Type: {violation_data.get('violation_type', '')}
        Timestamp: {violation_data.get('timestamp', '')}
        Person ID: {violation_data.get('person_id', '')}
        Frame: {violation_data.get('frame', '')}
        Helmet Confidence: {float(violation_data.get('helmet_conf', 0.0)):.2f}
        Vest Confidence: {float(violation_data.get('vest_conf', 0.0)):.2f}

        📷 Attached below is the snapshot of the violation for review.
        """

        msg.attach(MIMEText(body, 'plain'))

        # --- Attach Snapshot Image ---
        image_path = violation_data.get('image_path', '')
        if image_path:
            # Handle relative or absolute paths
            abs_path = image_path
            if image_path.startswith("/violations/"):
                abs_path = os.path.join(os.getcwd(), image_path.lstrip("/"))

            if os.path.exists(abs_path):
                with open(abs_path, 'rb') as f:
                    img = MIMEImage(f.read(), name=os.path.basename(abs_path))
                    img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(abs_path))
                    msg.attach(img)
                print(f"📎 Attached image: {abs_path}")
            else:
                print(f"⚠️ Image not found for attachment: {abs_path}")

        # --- Send Email ---
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        server.send_message(msg)
        server.quit()

        print(f"✅ Email sent successfully to {email_config['recipient_email']}")
        return True

    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def send_daily_report():
    """Send a daily CSV summary of all violations."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        violations = get_violations_by_date_range(today, today)
        if not violations:
            print("ℹ️ No violations found for daily report.")
            return False

        csv_filename = f"violations_report_{today}.csv"
        csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Violation Type", "Person ID", "Frame", "Helmet Conf", "Vest Conf"])
            for v in violations:
                writer.writerow([
                    v['timestamp'],
                    v['violation_type'],
                    v['person_id'],
                    v['frame'],
                    v['helmet_conf'],
                    v['vest_conf']
                ])

        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = f"SafetyEye Daily Report - {today}"

        body = f"""
SafetyEye Daily Violation Report

Date: {today}
Total Violations: {len(violations)}

Please find the detailed CSV report attached.
"""
        msg.attach(MIMEText(body, 'plain'))

        with open(csv_path, 'rb') as f:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{csv_filename}"')
            msg.attach(part)

        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        server.send_message(msg)
        server.quit()

        os.remove(csv_path)
        LAST_REPORT_SENT = datetime.now()  # ✅ record the time
        # ✅ Load saved report time immediately on startup
        save_last_report_time()
        print(f"✅ Daily report sent successfully to {email_config['recipient_email']}")
        return True
    except Exception as e:
        print(f"❌ Failed to send daily report: {e}")
        return False


# -----------------------
# Email Settings API (Dashboard Integration)
# -----------------------
@app.route('/api/settings/email', methods=['POST'])
def update_email_settings():
    """Update email configuration dynamically via dashboard and persist to disk."""
    global email_config
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'message': 'Missing JSON body'}), 400

        email_config.update({
            'smtp_server': data.get('smtp_server', 'smtp.gmail.com'),
            'smtp_port': int(data.get('smtp_port', 587)),
            'sender_email': data.get('sender_email', ''),
            'sender_password': data.get('sender_password', ''),
            'recipient_email': data.get('recipient_email', '')
        })

        # ✅ Save to JSON
        with open(EMAIL_CONFIG_FILE, 'w') as f:
            json.dump(email_config, f, indent=4)

        print(f"✅ Email configuration updated for {email_config['sender_email']} → {email_config['recipient_email']}")
        return jsonify({'success': True, 'message': 'Email settings saved successfully'})

    except Exception as e:
        print(f"❌ Error saving email settings: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/test-email', methods=['POST'])
def send_test_email():
    """Send a test email using the current saved configuration."""
    try:
        data = request.json or {}
        test_recipient = data.get('recipient') or email_config.get('recipient_email')

        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = test_recipient
        msg['Subject'] = "✅ SafetyEye Test Email"
        msg.attach(MIMEText(
            "This is a test email from SafetyEye. Your email settings are working correctly!",
            'plain'
        ))

        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['sender_password'])
        server.send_message(msg)
        server.quit()

        print(f"✅ Test email sent successfully to {test_recipient}")
        return jsonify({'success': True, 'message': f'Test email sent to {test_recipient}'})
    except Exception as e:
        print(f"❌ Failed to send test email: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500



# -----------------------
# Dashboard Data Helper
# -----------------------
class DashboardData:
    def __init__(self):
        self.violation_count_today = 0
        self.compliance_rate = 100.0
        self.current_occupancy = 0
        self.violations_data = []

    def update_stats(self):
        global total_detections_today, current_occupancy_estimate

        try:
            violations = get_violations_from_csv(limit=1000)
            self.violations_data = violations[:10]
            today = datetime.now().strftime("%Y-%m-%d")
            today_violations = [v for v in violations if v['timestamp'].startswith(today)]
            self.violation_count_today = len(today_violations)

            # ✅ Compute live compliance rate
            if total_detections_today > 0:
                self.compliance_rate = max(0, min(100, ((total_detections_today - self.violation_count_today) / total_detections_today) * 100))
            else:
                self.compliance_rate = 100.0

            # ✅ Live occupancy from tracker
            self.current_occupancy = current_occupancy_estimate

        except Exception as e:
            print(f"❌ Error updating stats: {e}")
            self.violation_count_today = 0
            self.compliance_rate = 100.0
            self.current_occupancy = 0
            self.violations_data = []


# -----------------------
# Camera / Capture Manager
# -----------------------
# Single capture thread handles reading frames AND running detection
camera_lock = threading.Lock()
camera = None
capture_thread = None
capture_thread_stop = threading.Event()
# source can be integer (webcam) or filepath (uploaded video)
active_source = 0
# latest frames (raw and processed) stored for streaming & UI endpoints
latest_raw_frame = None
latest_processed_frame = None
# simple de-dup for reported violations in current session
detected_violations_session = []

def open_capture(src):
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None
        if isinstance(src, str) and not os.path.exists(src):
            return False
        # cv2 will accept str path or integer index
        camera = cv2.VideoCapture(src)
        # try some common camera settings (optional)
        try:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        except:
            pass
        return camera.isOpened()

def close_capture():
    global camera
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
            except:
                pass
            camera = None

def capture_loop(src):
    global latest_raw_frame, latest_processed_frame, detected_violations_session
    print(f"🎬 Capture loop started for source: {src}")
    processing_flag["active"] = True
    cap_ok = open_capture(src)
    if not cap_ok:
        processing_flag["active"] = False
        print(f"❌ open_capture failed for {src}")
        return

    frame_count = 0

    # 🕒 Get FPS-based frame delay
    fps = camera.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:  # handle zero or NaN
        fps = 25.0
    frame_delay = 1.0 / fps

    while not capture_thread_stop.is_set():
        with camera_lock:
            if camera is None:
                print("❌ camera became None inside capture loop")
                break
            ret, frame = camera.read()
        if not ret:
            print("📹 End of stream or failed to read frame")
            break

        frame_count += 1
        latest_raw_frame = frame.copy()

        # ✅ Run YOLO detection once per frame
        try:
            processed_frame, violations = safety_monitor.process_frame(frame.copy())

            # ✅ Update live statistics
            global total_detections_today, current_occupancy_estimate
            if hasattr(safety_monitor, "tracker") and hasattr(safety_monitor.tracker, "objects"):
                current_occupancy_estimate = len(safety_monitor.tracker.objects)
            else:
                current_occupancy_estimate = max(1, len(violations)) if violations else 0

            if current_occupancy_estimate > 0:
                total_detections_today += current_occupancy_estimate

        except Exception as e:
            print(f"❌ Exception during detection: {e}")
            processed_frame = frame.copy()
            violations = []

        latest_processed_frame = processed_frame

        # ✅ Log each violation properly
        try:
            for violation in violations:
                if not isinstance(violation, dict):
                    print(f"⚠️ Skipping invalid violation: {violation}")
                    continue

                person_id = violation.get('person_id', 'N/A')
                helmet_conf = violation.get('helmet_conf', 0.0)
                vest_conf = violation.get('vest_conf', 0.0)
                types = violation.get('types', [])

                violation_type_str = ', '.join(types) if isinstance(types, list) else str(types)
                key = (person_id, violation_type_str)

                if key not in detected_violations_session:
                    detected_violations_session.append(key)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"violation_{timestamp}.jpg"
                    image_path = os.path.join(violations_dir, image_filename)
                    cv2.imwrite(image_path, processed_frame)

                    public_image_path = f"/violations/{image_filename}"

                    log_violation_to_csv(
                        violation_type=violation_type_str,
                        confidence=0.95,
                        image_path=public_image_path,
                        person_id=person_id,
                        frame_id=frame_count,
                        helmet_conf=helmet_conf,
                        vest_conf=vest_conf
                    )

                    print(f"⚠️ New violation logged: {violation} for person {person_id}")

        except Exception as e:
            print(f"❌ Error handling violations: {e}")

        # 🕐 Sleep according to video’s real FPS
        time.sleep(frame_delay)

    close_capture()
    processing_flag["active"] = False
    print("✅ Capture loop exiting — processing_flag set to False")



def start_capture_thread(src):
    global capture_thread, capture_thread_stop, active_source, detected_violations_session
    stop_capture_thread()
    time.sleep(0.3)
    capture_thread_stop.clear()
    active_source = src
    detected_violations_session = []  # reset session de-dupe on new source
    capture_thread = threading.Thread(target=capture_loop, args=(src,), daemon=True)
    capture_thread.start()
    return True

def stop_capture_thread():
    global capture_thread, capture_thread_stop
    capture_thread_stop.set()
    # allow some time for thread to exit
    if capture_thread and capture_thread.is_alive():
        capture_thread.join(timeout=2)
    capture_thread = None

# -----------------------
# Streaming generator
# -----------------------
def generate_frames_stream():
    """Continuously yields the latest processed frame as MJPEG stream."""
    global latest_processed_frame
    print("🔁 Video stream generator started")
    retry_blank = 0

    while True:
        if latest_processed_frame is None:
            # show waiting screen until detection starts
            retry_blank += 1
            h, w = 480, 640
            placeholder = np.zeros((h, w, 3), dtype=np.uint8)
            msg = "Waiting for video feed..." if retry_blank < 50 else "No Active Source"
            cv2.putText(placeholder, msg, (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            ret, buffer = cv2.imencode('.jpg', placeholder)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)
            continue

        retry_blank = 0  # reset once frames arrive
        try:
            ret, buffer = cv2.imencode('.jpg', latest_processed_frame)
            if not ret:
                time.sleep(0.03)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except GeneratorExit:
            break
        except Exception as e:
            print(f"❌ Stream error: {e}")
            time.sleep(0.05)
            continue
        time.sleep(0.03)


# -----------------------
# Background Report Scheduler
# -----------------------
def daily_report_scheduler():
    """Sends daily/periodic reports. Runs in background thread started once."""
    while True:
        now = datetime.now()
        # Example: send a small summary every 15 minutes (you had this in original)
        if now.minute % 15 == 0:
            try:
                send_daily_report()
            except Exception as e:
                print(f"❌ daily_report_scheduler error: {e}")
            time.sleep(60)  # avoid double send within the same minute
        time.sleep(30)

# -----------------------
# Flask Route Handlers
# -----------------------
startup_done = False

@app.before_request
def initialize_on_first_request():
    global startup_done
    if not startup_done:
        init_violations_system()
        # start background daily report scheduler
        threading.Thread(target=daily_report_scheduler, daemon=True).start()
        startup_done = True

@app.route('/')
def index():
    # Make sure index.html exists in templates
    return render_template('index.html')

@app.route('/logs')
def logs():
    return render_template('logs.html')

@app.route('/settings')
def settings():
    return render_template('settings.html', email_config=email_config)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_webcam', methods=['GET', 'POST'])
def api_start_webcam():
    """Start webcam as active source (use device index 0)."""
    try:
        src = 0
        ok = start_capture_thread(src)
        if not ok:
            return jsonify({'status': 'failed', 'message': 'Failed to start webcam'}), 500
        return jsonify({'status': 'started', 'source': 'webcam'})
    except Exception as e:
        print(f"❌ start_webcam error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/stop_processing', methods=['POST', 'GET'])
def api_stop_processing():
    """Stop any active capture thread and clear camera."""
    try:
        stop_capture_thread()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        print(f"❌ stop_processing error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

ALLOWED_EXT = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

@app.route('/api/upload_video', methods=['POST'])
def api_upload_video():
    """
    Expects a multipart/form-data POST with the key 'video'.
    Frontend must use: formData.append('video', file)
    """
    if 'video' not in request.files:
        return jsonify({'error': 'No video file in request. Use key "video"'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate extension
    if '.' in file.filename:
        ext = file.filename.rsplit('.', 1)[1].lower()
        if ext not in ALLOWED_EXT:
            return jsonify({'error': f'Invalid file type. Allowed: {", ".join(sorted(ALLOWED_EXT))}'}), 400

    try:
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        # verify the video can be opened by OpenCV
        tcap = cv2.VideoCapture(save_path)
        if not tcap.isOpened():
            tcap.release()
            # remove corrupted file
            try:
                os.remove(save_path)
            except:
                pass
            return jsonify({'error': 'Uploaded video is invalid or corrupted'}), 400
        tcap.release()
        processing_flag["active"] = True
        # start capture thread on uploaded file
        # reset frame buffer and start capture
        global latest_processed_frame
        latest_processed_frame = None
        start_capture_thread(save_path)
        return jsonify({'status': 'started', 'source': 'uploaded_video', 'filename': filename})
    except Exception as e:
        processing_flag["active"] = False
        print(f"❌ upload_video error: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

# ✅ ADD THIS DIRECTLY BELOW
@app.route("/api/is_processing", methods=["GET"])
def is_processing():
    """Return whether the system is currently processing a video."""
    try:
        return jsonify({"processing": processing_flag["active"]})
    except Exception as e:
        print(f"❌ Error in is_processing: {e}")
        return jsonify({"processing": False, "error": str(e)}), 500

@app.route('/api/violations/all', methods=['GET'])
def get_violations():
    """Return all logged violations as proper JSON list."""
    try:
        violations = []

        # ✅ Read CSV file and convert rows into dicts
        if os.path.exists(VIOLATIONS_CSV_PATH):
            with open(VIOLATIONS_CSV_PATH, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Ensure numeric types are converted back
                    violations.append({
                        'timestamp': row.get('timestamp', ''),
                        'violation_type': row.get('violation_type', ''),
                        'person_id': row.get('person_id', ''),
                        'frame': int(row.get('frame', 0)),
                        'helmet_conf': float(row.get('helmet_conf', 0.0)),
                        'vest_conf': float(row.get('vest_conf', 0.0)),
                        'image_path': row.get('image_path', '')
                    })

        return jsonify(violations)

    except Exception as e:
        print(f"❌ Error reading violations: {e}")
        return jsonify([])

@app.route('/api/violations', methods=['GET'])
    
def get_recent_violations():
    """Return only the 10 most recent valid violations for dashboard display."""
    try:
        violations = []

        if os.path.exists(VIOLATIONS_CSV_PATH):
            with open(VIOLATIONS_CSV_PATH, 'r', newline='') as f:
                reader = list(csv.DictReader(f))

                for row in reversed(reader):  # newest first
                    # Skip empty or invalid rows
                    vtype = (row.get('violation_type') or '').strip()
                    if not vtype or vtype.lower() == 'unknown':
                        continue

                    try:
                        violations.append({
                            'timestamp': row.get('timestamp', ''),
                            'violation_type': vtype,
                            'person_id': row.get('person_id', ''),
                            'frame': int(float(row.get('frame', 0))),
                            'helmet_conf': float(row.get('helmet_conf', 0.0)),
                            'vest_conf': float(row.get('vest_conf', 0.0)),
                            'image_path': row.get('image_path', '')
                        })
                    except Exception as parse_err:
                        print(f"⚠️ Skipping malformed row: {parse_err} — {row}")
                        continue

                    if len(violations) >= 10:
                        break

        print(f"📡 Returning {len(violations)} recent violations to dashboard")
        return jsonify(violations)

    except Exception as e:
        print(f"❌ Error reading recent violations: {e}")
        return jsonify([])



@app.route('/api/violations/date-range')
def get_violations_by_date():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    if not start_date or not end_date:
        return jsonify({'error': 'Start and end dates required'}), 400
    violations = get_violations_by_date_range(start_date, end_date)
    return jsonify(violations)

@app.route('/api/violations/send-email', methods=['POST'])
def send_violation_email_api():
    violation_data = request.json
    if not violation_data:
        return jsonify({'success': False, 'message': 'Missing JSON body'}), 400
    success = send_violation_email(violation_data)
    return jsonify({'success': success})

@app.route('/api/violations/download-csv')
def download_csv():
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    if not start_date or not end_date:
        return jsonify({'error': 'Start and end dates required'}), 400
    violations = get_violations_by_date_range(start_date, end_date)
    csv_filename = f"violations_{start_date}_to_{end_date}.csv"
    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Timestamp", "Violation Type", "Person ID", "Frame", "Helmet Conf", "Vest Conf", "Image Path"])
        for v in violations:
            writer.writerow([
                v.get('id',''), v.get('timestamp',''), v.get('violation_type',''),
                v.get('person_id',''), v.get('frame',''), v.get('helmet_conf',''),
                v.get('vest_conf',''), v.get('image_path','')
            ])
    return send_file(csv_path, as_attachment=True, download_name=csv_filename)

@app.route('/api/violations/clear', methods=['DELETE'])
def clear_all_violations():
    """Delete all entries in the violations CSV file."""
    try:
        _ensure_violations_csv()
        with open(VIOLATIONS_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame", "person_id", "violation_type", "image_path", "helmet_conf", "vest_conf"])
        print("🗑️ All violations cleared from CSV.")
        return jsonify({'success': True, 'message': 'All violations cleared successfully.'})
    except Exception as e:
        print(f"❌ Error clearing violations: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500




# --------------------------------------------
# 📊 Dashboard Stats API — with compliance trend
# --------------------------------------------
from collections import deque

# Store last 10 compliance readings (global in-memory cache)
compliance_history = deque(maxlen=10)
compliance_timestamps = deque(maxlen=10)

@app.route('/api/stats')
def get_stats():
    dashboard_data = DashboardData()
    dashboard_data.update_stats()

    # Record compliance rate trend (every call adds a new point)
    now_label = datetime.now().strftime("%H:%M")
    compliance_history.append(dashboard_data.compliance_rate)
    compliance_timestamps.append(now_label)

    stats = {
        'violation_count_today': dashboard_data.violation_count_today,
        'compliance_rate': round(dashboard_data.compliance_rate, 1),
        'current_occupancy': dashboard_data.current_occupancy,
        'detection_mode': 'REAL' if safety_monitor.model_loaded else 'MOCK',
        'processing_status': 'active' if capture_thread and capture_thread.is_alive() else 'inactive',
        'violations_by_type': {
            'no_helmet': sum(1 for v in dashboard_data.violations_data if 'no_helmet' in v['violation_type'].lower()),
            'no_vest': sum(1 for v in dashboard_data.violations_data if 'no_vest' in v['violation_type'].lower()),
            'other': 0
        },
        # ✅ New section for chart
        'compliance_trend': {
            'timestamps': list(compliance_timestamps),
            'values': list(compliance_history)
        }
    }

    return jsonify(stats)


@app.route('/violations/<path:filename>')
def serve_violation_image(filename):
    try:
        # Handle both absolute and relative Windows paths
        filename = filename.replace("\\", "/")  # normalize slashes

        if os.path.isabs(filename):
            abs_path = filename
        else:
            # Allow nested subpaths like "violations/violation_123.jpg"
            abs_path = os.path.join(violations_dir, os.path.basename(filename))

        if not os.path.exists(abs_path):
            print(f"⚠️ Snapshot not found: {abs_path}")
            abort(404)

        return send_file(abs_path)

    except Exception as e:
        print(f"❌ Could not serve image: {e}")
        abort(404)

# ----------------------------
# 📡 System Information API
# ----------------------------
APP_START_TIME = datetime.now()
LAST_REPORT_SENT = None  # will update whenever a report is sent

@app.route('/api/system/info', methods=['GET'])
def get_system_info():
    """Return real-time system info for the settings page."""
    try:
        # ✅ Calculate uptime
        uptime_seconds = (datetime.now() - APP_START_TIME).total_seconds()
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        uptime_str = f"{days}d {hours}h {minutes}m"

        # ✅ Count total violations from CSV
        total_violations = 0
        if os.path.exists(VIOLATIONS_CSV_PATH):
            with open(VIOLATIONS_CSV_PATH, 'r', newline='') as f:
                reader = list(csv.DictReader(f))
                total_violations = len(reader)

        # ✅ Detection mode
        detection_mode = "REAL YOLO" if safety_monitor.model_loaded else "MOCK"

        # ✅ Model name
        model_name = "N/A"
        if safety_monitor.model_loaded and hasattr(safety_monitor, "model") and hasattr(safety_monitor.model, "names"):
            model_name = os.path.basename(MODEL_PATH)

        # ✅ Last report time
        last_report = LAST_REPORT_SENT.strftime("%Y-%m-%d %H:%M:%S") if LAST_REPORT_SENT else "Never"

        return jsonify({
            "detection_mode": detection_mode,
            "model_version": model_name,
            "total_violations": total_violations,
            "system_uptime": uptime_str,
            "last_report_sent": last_report
        })
    except Exception as e:
        print(f"❌ Error in get_system_info: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------
# Run App
# -----------------------
if __name__ == '__main__':
    print("🎯 Enhanced SafetyEye Dashboard Started")
    print("📊 Main Dashboard: http://localhost:5000")
    print("📋 Violation Logs: http://localhost:5000/logs")
    print("⚙️ Settings: http://localhost:5000/settings")
    print("🔧 Detection Mode:", "REAL YOLO" if safety_monitor.model_loaded else "MOCK")
    # Start with no capture active; user triggers start via frontend
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
