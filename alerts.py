import smtplib

VIOLATION_CLASSES = ["no_helmet", "no_vest"]

def check_violation(detected_label):
    return detected_label in VIOLATION_CLASSES

def send_alert(violation):
    print(f"[ALERT] Safety violation detected: {violation}")

