import os
import cv2
import numpy as np
from collections import Counter
import hashlib

# ✅ Paths
dataset_path = r"C:\safetyeye\preprocessed\test\images"  # change to train/val when needed

def check_images(dataset_path, target_size=(640, 640)):
    image_files = [f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    total = len(image_files)
    normalized = 0
    correct_size = 0
    hashes = []
    
    for file in image_files:
        path = os.path.join(dataset_path, file)
        img = cv2.imread(path)

        if img is None:
            print(f"⚠️ Could not read {file}")
            continue

        # Normalization check
        if img.max() <= 255 and img.min() >= 0:
            normalized += 1

        # Resize check
        if img.shape[0] == target_size[1] and img.shape[1] == target_size[0]:
            correct_size += 1

        # Duplicate check (hashing)
        with open(path, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            hashes.append(file_hash)

    # Count duplicates
    dup_count = sum(count - 1 for count in Counter(hashes).values() if count > 1)

    # Report
    print("✅ Preprocessing Verification Report:")
    print(f"Total images         : {total}")
    print(f"Normalized images    : {normalized}/{total}")
    print(f"Correct size ({target_size[0]}x{target_size[1]}) : {correct_size}/{total}")
    print(f"Duplicate images     : {dup_count}")

    # Augmentation check (file naming or count > original)
    aug_files = [f for f in image_files if "aug" in f.lower()]
    if len(aug_files) > 0:
        print(f"🎉 Augmented images detected: {len(aug_files)}")
    else:
        print("⚠️ No augmented images found. Maybe augmentation not applied yet.")

# Run
check_images(dataset_path)
