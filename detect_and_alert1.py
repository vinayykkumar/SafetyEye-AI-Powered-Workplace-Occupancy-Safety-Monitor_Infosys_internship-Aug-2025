from ultralytics import YOLO
import yagmail
import os

# -----------------------------
# 1️⃣ Load YOLO model
# -----------------------------
model = YOLO("C:/Users/Shreya/Desktop/infosys_project/best.pt")  # path to your trained model

# -----------------------------
# 2️⃣ Input image
# -----------------------------
source_image = "C:/Users/Shreya/Desktop/infosys_project/safety1.jpeg"  # your test image

# -----------------------------
# 3️⃣ Run detection
# -----------------------------
results = model.predict(
    source=source_image,
    show=False,  # avoids cv2.imshow pop-up
    save=True    # saves output in runs/detect/
)

# -----------------------------
# 4️⃣ Path to saved detection image
# -----------------------------
# YOLO saves images in runs/detect/predict by default
output_dir = "runs/detect/predict"
# Dynamically get the first saved image
detected_image_path = os.path.join(output_dir, os.listdir(output_dir)[0])

# -----------------------------
# 5️⃣ Check if any objects detected
# -----------------------------
if len(results[0].boxes) > 0:
    print("Violation detected! Sending email...")

    # -----------------------------
    # 6️⃣ Email setup
    # -----------------------------
    receiver_email = "sejalkothali2003@gmail.com"  # who will receive the alert
    sender_email = "shreyarajoba@gmail.com"    # your Gmail address
    sender_password = "Advik@2421"    # Gmail App Password

    # Initialize yagmail
    yag = yagmail.SMTP(user=sender_email, password=sender_password)

    # Send email with attached detected image
    yag.send(
        to=receiver_email,
        subject="🚨 Safety Alert Detected!",
        contents="A violation was detected by SafetyEye AI. Check the attached image.",
        attachments=detected_image_path
    )

    print("Email sent successfully!")
else:
    print("No violations detected.")
