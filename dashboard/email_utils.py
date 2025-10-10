import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import cv2
from datetime import datetime
from src.config import Config
import logging
from report_utils import generate_violation_report  # Fixed: Removed 'dashboard.' prefix

# Setup logging
logger = logging.getLogger(__name__)

class EmailNotifier:
    def __init__(self, sender_email, sender_password):
        Config.validate_paths()
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 465
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.email_queue = []
        self.sent_count = 0
        self.failed_count = 0
        self.last_email_time = 0
        self.email_cooldown = 60  # Seconds

    def send_violation_email(self, recipient, violation, confidence, severity, frame, frame_number):
        current_time = time.time()
        if current_time - self.last_email_time < self.email_cooldown:
            logger.warning("High email volume detected. Waiting to prevent spam.")
            return
        self.last_email_time = current_time
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = f"Safety Violation Alert: {violation}"
            
            body = f"Violation: {violation}\nConfidence: {confidence:.2f}\nSeverity: {severity}\nFrame: {frame_number}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            msg.attach(MIMEText(body, 'plain'))
            
            frame_path = os.path.join(Config.LOG_DIR, f"violation_{frame_number}.jpg")
            cv2.imwrite(frame_path, frame)
            with open(frame_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f"attachment; filename=violation_{frame_number}.jpg")
            msg.attach(part)
            
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient, msg.as_string())
            self.sent_count += 1
            logger.info(f"Sent email for {violation} to {recipient}")
        except Exception as e:
            self.failed_count += 1
            logger.error(f"Failed to send email: {e}")

    def send_summary_email(self, recipient, df):
        try:
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient
            msg['Subject'] = "Safety Violation Summary Report"
            
            body = generate_violation_report(df)
            msg.attach(MIMEText(body, 'plain'))
            
            csv_path = os.path.join(Config.LOG_DIR, "summary_report.csv")
            df.to_csv(csv_path, index=False)
            with open(csv_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', "attachment; filename=summary_report.csv")
            msg.attach(part)
            
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
                server.login(self.sender_email, self.sender_password)
                server.sendmail(self.sender_email, recipient, msg.as_string())
            self.sent_count += 1
            logger.info(f"Sent summary email to {recipient}")
        except Exception as e:
            self.failed_count += 1
            logger.error(f"Failed to send summary email: {e}")

    def get_queue_status(self):
        return len(self.email_queue), self.sent_count, self.failed_count