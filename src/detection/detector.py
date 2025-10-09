# from ultralytics import YOLO
# import cv2
# import os
# import argparse
# import tkinter as tk
# from tkinter import filedialog

# # ---------------------------
# # 1. Load YOLOv8 model
# # ---------------------------
# model = YOLO(r"C:\SafetyEye\models\best.pt")
#   # replace with your trained weights

# # ---------------------------
# # 2. Process frame (common for photo, video, webcam)
# # ---------------------------
# def process_frame(frame):
#     results = model(frame, conf=0.5)
#     annotated_frame = results[0].plot()
#     return annotated_frame

# # ---------------------------
# # 3. Main function with mode selection
# # ---------------------------
# def main(mode, source=None):
#     # Initialize tkinter root for file dialogs
#     root = tk.Tk()
#     root.withdraw()  # Hide main tkinter window

#     if mode == "photo":
#         # If no source given, open file dialog
#         if not source:
#             source = filedialog.askopenfilename(
#                 title="Select Photo",
#                 filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
#             )
#         if not source:
#             print("❌ No photo selected!")
#             return

#         frame = cv2.imread(source)
#         annotated_frame = process_frame(frame)
#         cv2.imshow("YOLOv8 Detection - Photo", annotated_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     elif mode == "video":
#         if not source:
#             source = filedialog.askopenfilename(
#                 title="Select Video",
#                 filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
#             )
#         if not source:
#             print("❌ No video selected!")
#             return

#         cap = cv2.VideoCapture(source)
#         if not cap.isOpened():
#             print("❌ Could not open video!")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             annotated_frame = process_frame(frame)
#             cv2.imshow("YOLOv8 Detection - Video", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     elif mode == "webcam":
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("❌ Could not open webcam!")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             annotated_frame = process_frame(frame)
#             cv2.imshow("YOLOv8 Detection - Webcam", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     else:
#         print("❌ Invalid mode! Choose 'photo', 'video', or 'webcam'.")

# # ---------------------------
# # 4. CLI argument parsing
# # ---------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, required=True, help="Mode: photo, video, webcam")
#     parser.add_argument("--source", type=str, help="Path to photo or video (optional)")
#     args = parser.parse_args()

#     main(args.mode, args.source)


# from ultralytics import YOLO
# import cv2
# import os
# import argparse
# import tkinter as tk
# from tkinter import filedialog
# import numpy as np

# # ---------------------------
# # 1. Load YOLOv8 model
# # ---------------------------
# model = YOLO(r"C:\SafetyEye\models\best.pt")
# # The path to your trained weights

# # This list holds the violation classes for easy checking
# VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# # ---------------------------
# # 2. Process frame (common for photo, video, webcam)
# # ---------------------------
# def process_frame(frame):
#     # Perform inference
#     results = model(frame, conf=0.5, verbose=False)

#     # Get the raw detection results
#     detections = results[0].boxes.cpu().numpy()
    
#     violation_detected = False
    
#     # Create a copy of the frame to draw on
#     annotated_frame = frame.copy()
    
#     for box in detections:
#         # Get coordinates and class label
#         x1, y1, x2, y2 = box.xyxy[0].astype(int)
#         cls = int(box.cls[0])
#         label = model.names[cls]
        
#         # Default colors
#         box_color = (0, 255, 0) # Green for safe detections
#         text_color = (0, 0, 0) # Black text

#         # Check for violations
#         if label in VIOLATION_CLASSES:
#             # Set colors for violation
#             box_color = (0, 0, 255) # Red for alerts
#             text_color = (255, 255, 255) # White text
#             violation_detected = True

#             # Print a console alert for testing
#             print(f"🚨 VIOLATION DETECTED: {label}")
        
#         # Draw the bounding box and label
#         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
#         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

#     return annotated_frame, violation_detected

# # ---------------------------
# # 3. Main function with mode selection
# # ---------------------------
# def main(mode, source=None):
#     # Initialize tkinter root for file dialogs
#     root = tk.Tk()
#     root.withdraw()  # Hide main tkinter window

#     if mode == "photo":
#         # If no source given, open file dialog
#         if not source:
#             source = filedialog.askopenfilename(
#                 title="Select Photo",
#                 filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
#             )
#         if not source:
#             print("❌ No photo selected!")
#             return

#         frame = cv2.imread(source)
#         annotated_frame, _ = process_frame(frame)
#         cv2.imshow("YOLOv8 Detection - Photo", annotated_frame)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     elif mode == "video":
#         if not source:
#             source = filedialog.askopenfilename(
#                 title="Select Video",
#                 filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
#             )
#         if not source:
#             print("❌ No video selected!")
#             return

#         cap = cv2.VideoCapture(source)
#         if not cap.isOpened():
#             print("❌ Could not open video!")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             annotated_frame, _ = process_frame(frame)
#             cv2.imshow("YOLOv8 Detection - Video", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     elif mode == "webcam":
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             print("❌ Could not open webcam!")
#             return

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             annotated_frame, _ = process_frame(frame)
#             cv2.imshow("YOLOv8 Detection - Webcam", annotated_frame)
            
#             # This loop will continuously check and display violations
#             # which are handled by the process_frame function.
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#     else:
#         print("❌ Invalid mode! Choose 'photo', 'video', or 'webcam'.")

# # ---------------------------
# # 4. CLI argument parsing
# # ---------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", type=str, required=True, help="Mode: photo, video, webcam")
#     parser.add_argument("--source", type=str, help="Path to photo or video (optional)")
#     args = parser.parse_args()

#     main(args.mode, args.source)


from ultralytics import YOLO
import cv2
import os
import argparse
import tkinter as tk
from tkinter import filedialog
import numpy as np
from datetime import datetime

# ---------------------------
# 1. Load YOLOv8 model
# ---------------------------
model = YOLO(r"C:\SafetyEye\models\best.pt")
# The path to your trained weights

# This list holds the violation classes for easy checking
VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# Define the output directory for screenshots
OUTPUT_DIR = "outputs/violation_screenshots"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# Process frame (common for photo, video, webcam)

COOLDOWN_SECONDS = 5
last_screenshot_time = None
def process_frame(frame):
    global last_screenshot_time
    # Perform inference
    results = model(frame, conf=0.5, verbose=False)

    # Get the raw detection results
    detections = results[0].boxes.cpu().numpy()
    
    violation_detected = False
    
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()
    
    for box in detections:
        # Get coordinates and class label
        x1, y1, x2, y2 = box.xyxy[0].astype(int)
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = model.names[cls]
        
        # Default colors
        box_color = (0, 255, 0) # Green for safe detections
        text_color = (0, 0, 0) # Black text

         # Create the full text label string
        display_label = f"{label} {conf:.2f}"

        # Check for violations
        if label in VIOLATION_CLASSES:
            # Set colors for violation
            box_color = (0, 0, 255) # Red for alerts
            text_color = (255, 255, 255) # White text
            violation_detected = True

            # Print a console alert
            print(f"🚨 VIOLATION DETECTED: {label}  (Confidence: {conf:.2f})")
        
        # Draw the bounding box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        # cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        cv2.putText(annotated_frame, display_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)

    # --- Save screenshot if a violation is detected ---
    if violation_detected:
         current_time = datetime.now()
         if last_screenshot_time is None or (current_time - last_screenshot_time).total_seconds() > COOLDOWN_SECONDS:
            timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp}.jpg")
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 Screenshot saved: {filename}")
            last_screenshot_time = current_time
        
    return annotated_frame, violation_detected

# ---------------------------
# 3. Main function with mode selection
# ---------------------------
def main(mode, source=None):
    # Initialize tkinter root for file dialogs
    root = tk.Tk()
    root.withdraw() # Hide main tkinter window

    if mode == "photo":
        if not source:
            source = filedialog.askopenfilename(
                title="Select Photo",
                filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
            )
        if not source:
            print("❌ No photo selected!")
            return

        frame = cv2.imread(source)
        annotated_frame, _ = process_frame(frame)
        cv2.imshow("YOLOv8 Detection - Photo", annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif mode == "video":
        if not source:
            source = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
            )
        if not source:
            print("❌ No video selected!")
            return

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("❌ Could not open video!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, _ = process_frame(frame)
            cv2.imshow("YOLOv8 Detection - Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    elif mode == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam!")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated_frame, _ = process_frame(frame)
            cv2.imshow("YOLOv8 Detection - Webcam", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        print("❌ Invalid mode! Choose 'photo', 'video', or 'webcam'.")

# ---------------------------
# 4. CLI argument parsing
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Mode: photo, video, webcam")
    parser.add_argument("--source", type=str, help="Path to photo or video (optional)")
    args = parser.parse_args()

    main(args.mode, args.source)