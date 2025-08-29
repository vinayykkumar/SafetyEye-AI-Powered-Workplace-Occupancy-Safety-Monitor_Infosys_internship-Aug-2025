import os
import cv2
import shutil
import hashlib
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get DATA_DIR and PROCESSED_NAME from .env
DATA_DIR = os.getenv("DATA_DIR")
if DATA_DIR is None:
    raise ValueError("DATA_DIR is not set in your .env file")
DATA_DIR = Path(DATA_DIR)

PROCESSED_NAME = os.getenv("PROCESSED_NAME", "safetyeye_v1")
OUTPUT_DIR = Path(f"./processed/{PROCESSED_NAME}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
IMG_SIZE = 640  # Resize all images to 640x640


def hash_image(image_path):
    """Return MD5 hash of an image to avoid duplicates"""
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def resize_image(image_path, output_path, size=640):
    """Resize image to square size and save"""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    resized = cv2.resize(img, (size, size))
    cv2.imwrite(str(output_path), resized)
    return True


def process_dataset(split):
    """Preprocess a single split (test/valid/train)"""
    img_dir = DATA_DIR / split / "images"
    label_dir = DATA_DIR / split / "labels"

    out_img_dir = OUTPUT_DIR / split / "images"
    out_label_dir = OUTPUT_DIR / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()

    if not img_dir.exists():
        print(f"Skipping {split} - no images found at {img_dir}")
        return

    for img_path in tqdm(list(img_dir.glob("*.*")), desc=f"Processing {split}"):
        hash_val = hash_image(img_path)
        if hash_val in seen_hashes:
            continue
        seen_hashes.add(hash_val)

        new_img_path = out_img_dir / img_path.name
        new_label_path = out_label_dir / (img_path.stem + ".txt")

        if resize_image(img_path, new_img_path, IMG_SIZE):
            label_file = label_dir / (img_path.stem + ".txt")
            if label_file.exists():
                new_label_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(label_file, new_label_path)

    print(f"{split} done: {len(seen_hashes)} unique images")


def rename_train_long_files():
    """Rename train images and labels to short names to avoid Windows long path errors"""
    train_img_dir = DATA_DIR / "train" / "images"
    train_lbl_dir = DATA_DIR / "train" / "labels"

    if not train_img_dir.exists():
        return

    for i, img_path in enumerate(train_img_dir.glob("*.*")):
        new_name = f"train_{i:05d}{img_path.suffix}"
        old_stem = img_path.stem
        img_path.rename(train_img_dir / new_name)

        label_file = train_lbl_dir / f"{old_stem}.txt"
        if label_file.exists():
            label_file.rename(train_lbl_dir / f"train_{i:05d}.txt")


if __name__ == "__main__":
    print(f"Preprocessing dataset from {DATA_DIR} → {OUTPUT_DIR}")

    # Step 1: Process test and valid first
    for split in ["test", "valid"]:
        if (DATA_DIR / split).exists():
            process_dataset(split)

    # Step 2: Rename train files to short names to avoid long path errors
    rename_train_long_files()

    # Step 3: Process train
    if (DATA_DIR / "train").exists():
        process_dataset("train")

    print("Preprocessing complete!")
