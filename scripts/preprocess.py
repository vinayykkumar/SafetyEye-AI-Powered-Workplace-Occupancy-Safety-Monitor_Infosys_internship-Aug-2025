import os
import cv2
import numpy as np

# Paths — update as needed
image_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_images"
label_dir = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_labels"

# Expected image size after normalization (YOLOv8 default = 640x640)
EXPECTED_SIZE = (640, 640)

def verify_dataset(image_dir, label_dir):
    issues = []

    for img_file in os.listdir(image_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # paths
        img_path = os.path.join(image_dir, img_file)
        lbl_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        # check label exists
        if not os.path.exists(lbl_path):
            issues.append(f"❌ Label missing for {img_file}")
            continue

        # read image
        img = cv2.imread(img_path)
        if img is None:
            issues.append(f"❌ Corrupted image {img_file}")
            continue

        h, w, c = img.shape

        # check normalization size
        if (w, h) != EXPECTED_SIZE:
            issues.append(f"⚠️ {img_file} has size {(w,h)}, expected {EXPECTED_SIZE}")

        # check pixel values
        min_val, max_val = img.min(), img.max()
        if not (0 <= min_val and max_val <= 255):
            issues.append(f"⚠️ {img_file} pixel values out of [0,255] range → min={min_val}, max={max_val}")

        # read labels
        with open(lbl_path, "r") as f:
            labels = [line.strip().split() for line in f if line.strip()]

        for line in labels:
            if len(line) != 5:
                issues.append(f"❌ Wrong label format in {lbl_path}: {line}")
                continue

            cls, x, y, bw, bh = map(float, line)

            # check class is int
            if not cls.is_integer() and not isinstance(cls, int):
                issues.append(f"⚠️ Non-integer class ID in {lbl_path}")

            # check YOLO bbox format (0–1 range)
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= bw <= 1 and 0 <= bh <= 1):
                issues.append(f"❌ Out-of-range bbox in {lbl_path}: {line}")

    if not issues:
        print("✅ All checks passed! Preprocessing looks correct.")
    else:
        print("⚠️ Found some issues:")
        for i in issues:
            print(i)


if __name__ == "__main__":
    verify_dataset(image_dir, label_dir)