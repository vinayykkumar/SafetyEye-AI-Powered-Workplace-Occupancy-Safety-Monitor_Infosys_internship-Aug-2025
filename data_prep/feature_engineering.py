import os
import cv2
import numpy as np
from pathlib import Path
import traceback

# --- LAYMAN EXPLANATION ---
# This script augments your dataset by flipping, rotating, and grayscaling images.
# It also updates the YOLO label files so the bounding boxes match the new images.
# This helps the model learn better and prevents bias towards any class.


# --- CONFIG ---
IMG_DIR = 'd:/team8/data/train/images'
LBL_DIR = 'd:/team8/data/train/labels'
AUG_IMG_DIR = 'd:/team8/data/train/images_aug'
AUG_LBL_DIR = 'd:/team8/data/train/labels_aug'
ROTATE_ANGLE = 15  # degrees
LOG_FILE = 'd:/team8/data_prep/feature_engineering_debug.log'

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(str(msg) + '\n')

try:
    os.makedirs(AUG_IMG_DIR, exist_ok=True)
    os.makedirs(AUG_LBL_DIR, exist_ok=True)
except Exception as e:
    log(f"Error creating directories: {e}")
    log(traceback.format_exc())

def flip_image_and_labels(img, labels, img_w, img_h, mode='horizontal'):
    if mode == 'horizontal':
        img_flipped = cv2.flip(img, 1)
        new_labels = []
        for l in labels:
            cls, x, y, w, h = l
            x = 1 - x  # flip x center
            new_labels.append([cls, x, y, w, h])
    elif mode == 'vertical':
        img_flipped = cv2.flip(img, 0)
        new_labels = []
        for l in labels:
            cls, x, y, w, h = l
            y = 1 - y  # flip y center
            new_labels.append([cls, x, y, w, h])
    else:
        raise ValueError('mode must be horizontal or vertical')
    return img_flipped, new_labels

def rotate_image_and_labels(img, labels, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    img_rot = cv2.warpAffine(img, M, (w, h))
    new_labels = []
    for l in labels:
        cls, x, y, bw, bh = l
        # Convert normalized to pixel
        x_pix = x * w
        y_pix = y * h
        # Rotate point
        xy_rot = np.dot(M, np.array([x_pix, y_pix, 1]))
        x_new = xy_rot[0] / w
        y_new = xy_rot[1] / h
        new_labels.append([cls, x_new, y_new, bw, bh])
    return img_rot, new_labels

def grayscale_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def read_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                labels.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return labels

def save_labels(label_path, labels):
    with open(label_path, 'w') as f:
        for l in labels:
            f.write(f"{int(l[0])} {l[1]:.6f} {l[2]:.6f} {l[3]:.6f} {l[4]:.6f}\n")


def augment_all():
    try:
        img_files = list(Path(IMG_DIR).glob('*.jpg'))
        log(f"Found {len(img_files)} images in {IMG_DIR}")
        for img_path in img_files:
            base = img_path.stem
            label_path = Path(LBL_DIR) / f"{base}.txt"
            if not label_path.exists():
                log(f"Label not found for {img_path}, skipping.")
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                log(f"Failed to read image: {img_path}")
                continue
            h, w = img.shape[:2]
            labels = read_labels(label_path)
            # 1. Horizontal Flip
            try:
                img_hf, labels_hf = flip_image_and_labels(img, labels, w, h, 'horizontal')
                cv2.imwrite(str(Path(AUG_IMG_DIR) / f"{base}_hf.jpg"), img_hf)
                save_labels(str(Path(AUG_LBL_DIR) / f"{base}_hf.txt"), labels_hf)
            except Exception as e:
                log(f"Error in horizontal flip for {img_path}: {e}")
                log(traceback.format_exc())
            # 2. Vertical Flip
            try:
                img_vf, labels_vf = flip_image_and_labels(img, labels, w, h, 'vertical')
                cv2.imwrite(str(Path(AUG_IMG_DIR) / f"{base}_vf.jpg"), img_vf)
                save_labels(str(Path(AUG_LBL_DIR) / f"{base}_vf.txt"), labels_vf)
            except Exception as e:
                log(f"Error in vertical flip for {img_path}: {e}")
                log(traceback.format_exc())
            # 3. Rotation
            try:
                img_rot, labels_rot = rotate_image_and_labels(img, labels, ROTATE_ANGLE)
                cv2.imwrite(str(Path(AUG_IMG_DIR) / f"{base}_rot.jpg"), img_rot)
                save_labels(str(Path(AUG_LBL_DIR) / f"{base}_rot.txt"), labels_rot)
            except Exception as e:
                log(f"Error in rotation for {img_path}: {e}")
                log(traceback.format_exc())
            # 4. Grayscale
            try:
                img_gray = grayscale_image(img)
                cv2.imwrite(str(Path(AUG_IMG_DIR) / f"{base}_gray.jpg"), img_gray)
                save_labels(str(Path(AUG_LBL_DIR) / f"{base}_gray.txt"), labels)
            except Exception as e:
                log(f"Error in grayscale for {img_path}: {e}")
                log(traceback.format_exc())
    except Exception as e:
        log(f"General error in augment_all: {e}")
        log(traceback.format_exc())

if __name__ == '__main__':
    with open(LOG_FILE, 'w') as f:
        f.write('Starting feature engineering debug log\n')
    augment_all()
