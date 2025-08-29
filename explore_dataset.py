# explore_dataset.py

import os
from pathlib import Path
from collections import Counter
from PIL import Image

DATA_DIR = Path("processed/safetyeye_v1")  # your preprocessed folder

def count_images_and_labels(split):
    img_dir = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"

    num_images = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
    num_labels = len(list(label_dir.glob("*.txt")))
    
    print(f"{split.upper()}: {num_images} images, {num_labels} labels")

    # Optional: check class distribution
    class_counter = Counter()
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counter[class_id] += 1

    print(f"Class distribution for {split}: {dict(class_counter)}\n")


def check_image_resolution(split):
    img_dir = DATA_DIR / split / "images"
    resolutions = Counter()
    for img_path in img_dir.glob("*.*"):
        with Image.open(img_path) as im:
            resolutions[im.size] += 1
    print(f"Resolutions for {split}: {dict(resolutions)}\n")


if __name__ == "__main__":
    for split in ["train", "valid", "test"]:
        count_images_and_labels(split)
        check_image_resolution(split)

#Verifies dataset consistency before training.

#Detects missing labels or images.
#Shows class imbalance for augmentation decisions.
#Reveals inconsistent image sizes