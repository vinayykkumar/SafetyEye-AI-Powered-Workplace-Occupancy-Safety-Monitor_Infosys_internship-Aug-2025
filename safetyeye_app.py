import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import threading
import time
import csv
import datetime
import smtplib
from email.mime.text import MIMEText
import shutil


def send_email_alert(subject, body, to_email, from_email, from_password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(from_email, from_password)
            server.sendmail(from_email, [to_email], msg.as_string())
        print("Email alert sent!")
    except Exception as e:
        print(f"Failed to send email: {e}")


class SafetyEyeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SafetyEye AI Monitoring System")
        self.root.geometry("1000x700")

        self.model = YOLO('yolov8s.pt')
        self.cap = None
        self.running = False

        self.last_alert_time = 0
        self.alert_cooldown = 3
        self.IOU_THRESHOLD = 0.1

        self.last_logged_violations = {}
        self.log_cooldown = 2.0

        self.email_cooldown = 60
        self.last_email_time = 0

        self.from_email = 'lokeshdevarapalli92@gmail.com'
        self.from_password = 'gvlbilfbxizswual'
        self.to_email = 'lokeshdevarapalli92@gmail.com'

        self.create_widgets()
        self.setup_logging()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.tab_camera = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_camera, text='Camera')

        self.tab_logs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_logs, text='Violation Logs')

        self.tab_reports = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_reports, text='Reports')
        ttk.Label(self.tab_reports, text="Reports coming soon...").pack(pady=20)

        self.tab_settings = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_settings, text='Settings')

        ttk.Label(self.tab_settings, text="Confidence Threshold").pack(pady=5)
        self.conf_slider = ttk.Scale(self.tab_settings, from_=0.1, to=1.0, value=0.5, orient=tk.HORIZONTAL)
        self.conf_slider.pack(fill='x', padx=20)

        ttk.Label(self.tab_settings, text="IOU Threshold").pack(pady=5)
        self.iou_slider = ttk.Scale(self.tab_settings, from_=0.1, to=1.0, value=0.3, orient=tk.HORIZONTAL)
        self.iou_slider.pack(fill='x', padx=20)

        # Camera index input
        self.camera_var = tk.IntVar(value=0)
        ttk.Label(self.tab_camera, text="Camera Index:").pack()
        camera_entry = ttk.Entry(self.tab_camera, textvariable=self.camera_var)
        camera_entry.pack()

        # Status label for violations
        self.status_label = ttk.Label(self.tab_camera, text="Status: Waiting...")
        self.status_label.pack(pady=5)

        self.canvas = tk.Canvas(self.tab_camera, width=640, height=480)
        self.canvas.pack(pady=10)

        btn_frame = ttk.Frame(self.tab_camera)
        btn_frame.pack()
        self.btn_start = ttk.Button(btn_frame, text="Start Detection", command=self.start_detection)
        self.btn_start.pack(side=tk.LEFT, padx=10)
        self.btn_stop = ttk.Button(btn_frame, text="Stop Detection", command=self.stop_detection, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        self.log_text = scrolledtext.ScrolledText(self.tab_logs, state='disabled', height=25)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Export logs button in Logs tab
        export_btn = ttk.Button(self.tab_logs, text="Export Logs", command=self.export_log)
        export_btn.pack(pady=10)

    def export_log(self):
        try:
            shutil.copy('violation_log.csv', 'exported_violation_log.csv')
            messagebox.showinfo("Export", "Violation logs saved to exported_violation_log.csv")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs: {e}")

    def setup_logging(self):
        self.logfile = 'violation_log.csv'
        try:
            with open(self.logfile, 'x', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Timestamp', 'ViolationType', 'X1', 'Y1', 'X2', 'Y2'])
        except FileExistsError:
            pass

    def log_violation(self, violation_type, box):
        box_tuple = tuple(map(int, box))
        key = (violation_type, box_tuple)
        now = time.time()
        if key in self.last_logged_violations and now - self.last_logged_violations[key] < self.log_cooldown:
            return
        self.last_logged_violations[key] = now

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        x1, y1, x2, y2 = box_tuple
        with open(self.logfile, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, violation_type, x1, y1, x2, y2])
        self.append_log(f"{timestamp} - {violation_type} at ({x1},{y1},{x2},{y2})")

        self.status_label.config(text=f"Last violation: {violation_type} at {datetime.datetime.now().strftime('%H:%M:%S')}")

        if now - self.last_email_time > self.email_cooldown:
            subject = f"SafetyEye Alert: {violation_type}"
            body = f"Violation detected: {violation_type}\nLocation: {box}\nTimestamp: {timestamp}"
            send_email_alert(subject, body, self.to_email, self.from_email, self.from_password)
            self.last_email_time = now

    def append_log(self, text):
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if boxAArea + boxBArea - interArea == 0:
            return 0
        return interArea / float(boxAArea + boxBArea - interArea)

    def start_detection(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.camera_var.get())
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Failed to open webcam")
            return
        # Performance tweak: set capture resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self.detect_loop, daemon=True).start()

    def stop_detection(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.canvas.delete('all')

    def detect_loop(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                continue
            conf_val = float(self.conf_slider.get())
            iou_val = float(self.iou_slider.get())
            results = self.model(frame, conf=conf_val, iou=iou_val)
            detections = results[0]
            boxes = detections.boxes.xyxy.cpu().numpy()
            classes = detections.boxes.cls.cpu().numpy()
            class_names = self.model.names
            person_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "person"]
            helmet_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "helmet"]
            vest_boxes = [boxes[i] for i, c in enumerate(classes) if class_names[int(c)] == "vest"]
            violations = []
            for p_box in person_boxes:
                helmet_found = any(self.iou(p_box, h_box) > iou_val for h_box in helmet_boxes)
                vest_found = any(self.iou(p_box, v_box) > iou_val for v_box in vest_boxes)
                if not helmet_found:
                    self.log_violation("Helmet Missing", p_box)
                    violations.append((p_box, "Helmet Missing"))
                if not vest_found:
                    self.log_violation("Vest Missing", p_box)
                    violations.append((p_box, "Vest Missing"))
            for i, (box, label) in enumerate(violations):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, y1 - 10 - 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.imgtk = imgtk
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            elapsed = time.time() - start_time
            print(f"Loop time: {elapsed:.3f} seconds")

            time.sleep(0.03)

        if self.cap:
            self.cap.release()


if __name__ == '__main__':
    root = tk.Tk()
    app = SafetyEyeGUI(root)
    root.mainloop()
