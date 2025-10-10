# train_and_plot_yolov8.py
from ultralytics import YOLO
import pandas as pd
import matplotlib.pyplot as plt
import os

# --------------------------
# 1️⃣ Train YOLOv8 model
# --------------------------
model = YOLO("yolov8s.pt")  # or yolov8m.pt for medium

# Training parameters
project_folder = "runsbyrevanth1"
run_name = "runsbyrevanth"

model.train(
    data="content/data/data/data.yaml",        # your dataset YAML
    epochs=25,               # number of training epochs
    batch=32,                # batch size
    imgsz=640,               # image size
    optimizer="SGD",         # optimizer
    lr0=0.001,               # initial learning rate
    lrf=0.05,                 # final learning rate multiplier
    momentum=0.937,          # momentum for SGD
    weight_decay=0.0005,     # regularization
    workers=4,               # CPU threads for data loading
    project=project_folder,  # output folder
    name=run_name,           # subfolder for this run
    exist_ok=True
)

print(f"Training complete! Best model saved in {project_folder}/{run_name}/best.pt")

# --------------------------
# 2️⃣ Load training results
# --------------------------
results_csv = os.path.join(project_folder, run_name, "results.csv")

if not os.path.exists(results_csv):
    raise FileNotFoundError(f"Results file not found at {results_csv}")


























# train_yolov8_adamw_colab.py
from ultralytics import YOLO
import os

# --------------------------
# 1️⃣ Load YOLOv8 model
# --------------------------
model = YOLO("yolov8s.pt")   # use yolov8m.pt if GPU allows, but s is safe for Colab free

# Training parameters
project_folder = "YOLOv8_training"
run_name = "custom_run_adamw_64"

# --------------------------
# 2️⃣ Train with AdamW
# --------------------------
model.train(
    data="/content/data/data/data.yaml",   # dataset YAML path
    epochs=25,               # fixed at 25
    batch=32,                # ✅ better for Colab T4 (16GB VRAM)
    imgsz=640,               # image size
    optimizer="AdamW",       # switched to AdamW
    lr0=0.002,               # good default LR for AdamW
    lrf=0.1,                 # final LR multiplier
    weight_decay=0.01,       # AdamW benefits from stronger decay
    workers=4,               # adjust if dataloader issues
    project=project_folder,  # save path
    name=run_name,           # run name
    exist_ok=True,
    augment=True             # enable built-in augmentations
)

print(f"✅ Training complete! Best model saved in {project_folder}/{run_name}/best.pt")






















# # 🚀 train_yolov8_adamw_colab_v2.py
# from ultralytics import YOLO
# import os

# # --------------------------
# # 1️⃣ Load YOLOv8 model
# # --------------------------
# # Use 'yolov8m.pt' if Colab GPU allows (better accuracy). Otherwise keep 'yolov8s.pt'.
# model = YOLO("yolov8s.pt")

# # Training parameters
# project_folder = "YOLOv8_training"
# run_name = "custom_run_adamw_90plus"

# # --------------------------
# # 2️⃣ Train with AdamW + Augmentations
# # --------------------------
# model.train(
#     data="/content/data/data/data.yaml",  # dataset YAML path
#     epochs=30,                # increased for better convergence
#     patience=5,               # early stopping if no improvement
#     batch=16,                  # Colab T4 safe, adjust if OOM
#     imgsz=640,                 # standard image size
#     optimizer="AdamW",         # optimizer
#     lr0=0.002,                 # initial learning rate
#     lrf=0.01,                  # final LR = 1% of initial (cosine decay)
#     weight_decay=0.01,         # AdamW benefits from stronger decay
#     workers=4,                 # dataloader workers
#     project=project_folder,    # save path
#     name=run_name,             # run name
#     exist_ok=True,

#     # 🔥 Advanced Augmentations
#     hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,   # color jitter
#     degrees=0.0, translate=0.1, scale=0.9, shear=0.0,
#     perspective=0.0, flipud=0.0, fliplr=0.5,
#     mosaic=1.0, mixup=0.2,              # mosaic + mixup
# )

# print(f"✅ Training complete! Best model saved in {project_folder}/{run_name}/weights/best.pt")

# # --------------------------
# # 3️⃣ Validate best model
# # --------------------------
# metrics = model.val(
#     conf=0.25,  # confidence threshold
#     iou=0.6     # NMS IoU threshold
# )
# print("📊 Validation metrics:", metrics)





from ultralytics import YOLO
import cv2

# Restore original OpenCV imshow
cv2.imshow = cv2.__dict__["imshow"]
cv2.namedWindow = cv2.__dict__["namedWindow"]

# Load YOLO model
model = YOLO("runbyrevanth1/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    cv2.namedWindow("YOLOv8 Real-Time Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
