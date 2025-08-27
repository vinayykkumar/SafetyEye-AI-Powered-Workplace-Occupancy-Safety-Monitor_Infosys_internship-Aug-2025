import os
import cv2
import random
import numpy as np

# Paths
DATASET_PATH = "data/processed"
SRC_SPLIT = "train"   # augment only training data
AUG_SPLIT = "train_augmented"

IMG_DIR = os.path.join(DATASET_PATH, SRC_SPLIT, "images")
LBL_DIR = os.path.join(DATASET_PATH, SRC_SPLIT, "labels")

AUG_IMG_DIR = os.path.join(DATASET_PATH, AUG_SPLIT, "images")
AUG_LBL_DIR = os.path.join(DATASET_PATH, AUG_SPLIT, "labels")

# Create new folders
os.makedirs(AUG_IMG_DIR, exist_ok=True)
os.makedirs(AUG_LBL_DIR, exist_ok=True)


def adjust_brightness(img, factor):
    """Increase or decrease brightness"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] *= factor  # scale V channel
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def save_augmented(img, label_path, img_name, aug_type):
    """Save augmented image and copy labels as-is"""
    # Labels remain unchanged
    aug_img_name = img_name.replace(".jpg", f"_{aug_type}.jpg").replace(".png", f"_{aug_type}.png")
    aug_lbl_name = img_name.replace(".jpg", f"_{aug_type}.txt").replace(".png", f"_{aug_type}.txt")

    cv2.imwrite(os.path.join(AUG_IMG_DIR, aug_img_name), img)
    with open(label_path, "r") as f:
        labels = f.readlines()
    with open(os.path.join(AUG_LBL_DIR, aug_lbl_name), "w") as f:
        f.writelines(labels)


print("🚀 Starting augmentation...")

for img_file in os.listdir(IMG_DIR):
    if img_file.endswith(".jpg") or img_file.endswith(".png"):
        img_path = os.path.join(IMG_DIR, img_file)
        lbl_path = os.path.join(LBL_DIR, img_file.replace(".jpg", ".txt").replace(".png", ".txt"))

        if not os.path.exists(lbl_path):
            continue  # skip if no label file

        img = cv2.imread(img_path)

        # 1️⃣ Brightness (random between 0.5x and 1.5x)
        bright_factor = random.uniform(0.5, 1.5)
        bright_img = adjust_brightness(img, bright_factor)
        save_augmented(bright_img, lbl_path, img_file, "bright")

        # 2️⃣ Gaussian Blur (kernel size 5)
        blur_img = cv2.GaussianBlur(img, (5, 5), 0)
        save_augmented(blur_img, lbl_path, img_file, "blur")

print("✅ Augmentation complete! Images saved in:", AUG_IMG_DIR)
