import cv2
import numpy as np
import random
from pathlib import Path
import shutil

# -----------------------------
# Config
# -----------------------------
DATA_DIR = Path("processed/safetyeye_v1/train")       # Original train folder
AUG_TMP_DIR = Path("processed/safetyeye_v1_aug_tmp")  # Temporary folder for augmented images

IMG_DIR = DATA_DIR / "images"
LABEL_DIR = DATA_DIR / "labels"

AUG_IMG_DIR = AUG_TMP_DIR / "images"
AUG_LABEL_DIR = AUG_TMP_DIR / "labels"

AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABEL_DIR.mkdir(parents=True, exist_ok=True)

# Minority classes (adjust based on your dataset analysis)
MINORITY_CLASSES = [0, 4, 5, 6, 7, 10, 11, 13]

# -----------------------------
# Helper functions
# -----------------------------
def horizontal_flip(img, labels):
    img = cv2.flip(img, 1)
    flipped_labels = [(cls_id, 1-x, y, w, h) for cls_id, x, y, w, h in labels]
    return img, flipped_labels

def rotate_image(img, labels, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    rotated_labels = []
    for cls_id, x, y, bw, bh in labels:
        cx = x * w
        cy = y * h
        coords = np.array([cx, cy, 1])
        rx, ry = np.dot(M, coords)
        rx /= w
        ry /= h
        rotated_labels.append((cls_id, rx, ry, bw, bh))
    return img_rot, rotated_labels

def adjust_brightness(img):
    factor = random.uniform(0.7, 1.3)
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def apply_cutout(img, labels, num_holes=1, max_size=0.2):
    h, w = img.shape[:2]
    for _ in range(num_holes):
        ch = int(h * random.uniform(0.05, max_size))
        cw = int(w * random.uniform(0.05, max_size))
        cx = random.randint(0, w - cw)
        cy = random.randint(0, h - ch)
        img[cy:cy+ch, cx:cx+cw] = 0
    return img, labels

# -----------------------------
# Main Augmentation Loop
# -----------------------------
aug_count = 0

for img_path in IMG_DIR.glob("*.*"):
    label_file = LABEL_DIR / (img_path.stem + ".txt")
    if not label_file.exists():
        continue

    with open(label_file, "r") as f:
        labels = [tuple(map(float, line.strip().split())) for line in f]

    # Skip if image has no minority classes
    if not any(int(cls_id) in MINORITY_CLASSES for cls_id, *_ in labels):
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    aug_imgs = []
    aug_labels = []

    # 1. Horizontal flip
    hf_img, hf_labels = horizontal_flip(img, labels)
    aug_imgs.append(hf_img)
    aug_labels.append(hf_labels)

    # 2. Random rotation ±15°
    angle = random.uniform(-15, 15)
    rot_img, rot_labels = rotate_image(img, labels, angle)
    aug_imgs.append(rot_img)
    aug_labels.append(rot_labels)

    # 3. Brightness adjustment
    bright_img = adjust_brightness(img)
    aug_imgs.append(bright_img)
    aug_labels.append(labels)

    # 4. Cutout
    cut_img, cut_labels = apply_cutout(img.copy(), labels)
    aug_imgs.append(cut_img)
    aug_labels.append(cut_labels)

    # Save augmented images to temporary folder
    for i, (a_img, a_lbl) in enumerate(zip(aug_imgs, aug_labels)):
        out_img_file = AUG_IMG_DIR / f"{img_path.stem}_aug{i}.jpg"
        out_lbl_file = AUG_LABEL_DIR / f"{img_path.stem}_aug{i}.txt"

        cv2.imwrite(str(out_img_file), a_img)
        with open(out_lbl_file, "w") as f:
            for cls_id, x, y, w_box, h_box in a_lbl:
                f.write(f"{int(cls_id)} {x:.6f} {y:.6f} {w_box:.6f} {h_box:.6f}\n")
        aug_count += 1

print(f"Total augmented images generated: {aug_count}")

# -----------------------------
# Merge augmented images into original train folder
# -----------------------------
for f in AUG_IMG_DIR.glob("*.*"):
    shutil.copy(f, IMG_DIR / f.name)
for f in AUG_LABEL_DIR.glob("*.txt"):
    shutil.copy(f, LABEL_DIR / f.name)

print("All augmented images and labels merged into the main training dataset!")

# Optionally, remove temporary folder
shutil.rmtree(AUG_TMP_DIR)
print("Temporary augmented folder deleted.")
