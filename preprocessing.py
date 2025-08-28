import os
import cv2
import numpy as np
import albumentations as A
import shutil
import yaml
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
import hashlib
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
SOURCE_DIR = SCRIPT_DIR / "../dataset/raw"
OUTPUT_DIR = SCRIPT_DIR / "../dataset/processed"

train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

class_names = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]

AUGMENTATIONS_PER_IMAGE = 2


def get_file_hash(filepath):
    """Computes the SHA256 hash of a file for duplicate detection."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def yolo_to_albumentations(yolo_bboxes, image_h, image_w):
    """Converts YOLO format bounding boxes to albumentations format."""
    alb_bboxes = []
    for bbox in yolo_bboxes:
        class_id, x_c, y_c, w, h = bbox
        x_min = (x_c - w / 2) * image_w
        y_min = (y_c - h / 2) * image_h
        x_max = (x_c + w / 2) * image_w
        y_max = (y_c + h / 2) * image_h
        alb_bboxes.append([x_min / image_w, y_min / image_h, x_max / image_w, y_max / image_h, class_id])
    return alb_bboxes

def albumentations_to_yolo(alb_bboxes, image_h, image_w):
    """Converts albumentations format bounding boxes back to YOLO format."""
    yolo_bboxes = []
    for bbox in alb_bboxes:
        x_min, y_min, x_max, y_max, class_id = bbox
        x_c = ((x_min + x_max) / 2)
        y_c = ((y_min + y_max) / 2)
        w = (x_max - x_min)
        h = (y_max - y_min)
        yolo_bboxes.append([class_id, x_c, y_c, w, h])
    return yolo_bboxes

# --- MAIN WORKFLOW FUNCTIONS ---

def clean_dataset():
    """Scans the source directory for corrupted and duplicate images."""
    print("--- Phase 1: Cleaning Dataset ---")
    image_paths = sorted(list(Path(SOURCE_DIR / "images").glob("*.*")))
    hashes = {}
    duplicates = []
    corrupted = []

    for img_path in tqdm(image_paths, desc="Scanning images"):
        # Check for corrupted images
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupted.append(img_path)
            continue
        # Check for duplicates
        file_hash = get_file_hash(img_path)
        if file_hash in hashes:
            duplicates.append(img_path)
        else:
            hashes[file_hash] = img_path
    
    files_to_remove = set(duplicates + corrupted)
    if not files_to_remove:
        print("No duplicate or corrupted images found.")
        return
        
    print(f"\nFound {len(duplicates)} duplicates and {len(corrupted)} corrupted images.")
    user_input = input("Do you want to remove these files? (y/n): ").lower()
    
    if user_input == 'y':
        for img_path in files_to_remove:
            label_path = SOURCE_DIR / "labels" / (img_path.stem + ".txt")
            if img_path.exists(): os.remove(img_path)
            if label_path.exists(): os.remove(label_path)
        print(f"Removed {len(files_to_remove)} images and their corresponding labels.")
    else:
        print("Skipping removal of bad files.")

def augment_training_data(train_paths):
    """Applies augmentations to the training set and saves them."""
    if AUGMENTATIONS_PER_IMAGE == 0:
        print("\nManual augmentation is disabled.")
        return
        
    print("\n--- Phase 3: Applying Manual Augmentations ---")
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=25, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.Blur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='albumentations', label_fields=['class_labels']))

    for img_path in tqdm(train_paths, desc="Augmenting training images"):
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        label_path = OUTPUT_DIR / "labels" / "train" / (img_path.stem + ".txt")
        with open(label_path, 'r') as f:
            bboxes_yolo = [list(map(float, line.strip().split())) for line in f.readlines()]
        
        class_ids = [int(b[0]) for b in bboxes_yolo]
        bboxes_alb = yolo_to_albumentations([b for b in bboxes_yolo], h, w)
        
        for i in range(AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=image, bboxes=bboxes_alb, class_labels=class_ids)
            aug_image = augmented['image']
            aug_bboxes_alb = augmented['bboxes']
            
            if not aug_bboxes_alb: continue # Skip if all bboxes are lost

            new_img_filename = f"{img_path.stem}_aug_{i}{img_path.suffix}"
            new_lbl_filename = f"{img_path.stem}_aug_{i}.txt"
            
            aug_img_path = OUTPUT_DIR / "images" / "train" / new_img_filename
            aug_lbl_path = OUTPUT_DIR / "labels" / "train" / new_lbl_filename
            
            cv2.imwrite(str(aug_img_path), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            
            aug_bboxes_yolo = albumentations_to_yolo(aug_bboxes_alb, h, w)
            with open(aug_lbl_path, 'w') as f:
                for bbox in aug_bboxes_yolo:
                    f.write(f"{int(bbox[0])} {' '.join(map(str, bbox[1:]))}\n")

def demonstrate_normalization(image_path):
    """Loads an image, normalizes it, and prints the results."""
    print("\n--- Phase 4: Demonstrating Manual Normalization ---")
    image = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print(f"Original data type: {image_rgb.dtype}, pixel range: [{image_rgb.min()}, {image_rgb.max()}]")
    
    # Normalize the pixel values from [0, 255] to [0.0, 1.0]
    normalized_image = image_rgb / 255.0
    
    print(f"Normalized data type: {normalized_image.dtype}, pixel range: [{normalized_image.min()}, {normalized_image.max()}]")


def main():
    """Main function to run the entire preprocessing workflow."""
    # Phase 1: Clean the source directory
    clean_dataset()

    # Phase 2: Split data and create the directory structure
    print("\n--- Phase 2: Splitting Dataset ---")
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
        
    all_image_paths = sorted([p for p in (SOURCE_DIR / "images").glob("*.*") if (SOURCE_DIR / "labels" / (p.stem + ".txt")).exists()])
    
    train_val_paths, test_paths = train_test_split(all_image_paths, test_size=test_ratio, random_state=42)
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    train_paths, val_paths = train_test_split(train_val_paths, test_size=val_size_adjusted, random_state=42)

    splits = {"train": train_paths, "val": val_paths, "test": test_paths}

    for split, paths in splits.items():
        for img_path in tqdm(paths, desc=f"Copying original {split} files"):
            img_dir = OUTPUT_DIR / "images" / split
            lbl_dir = OUTPUT_DIR / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, img_dir)
            shutil.copy(SOURCE_DIR / "labels" / (img_path.stem + ".txt"), lbl_dir)
    print("Dataset split and copy complete.")
            
    # Phase 3: Apply augmentations to the training set
    augment_training_data(train_paths)
    
    # Phase 4: Create the final data.yaml file
    yaml_data = {
        'train': str((OUTPUT_DIR / "images" / "train").resolve()),
        'val': str((OUTPUT_DIR / "images" / "val").resolve()),
        'test': str((OUTPUT_DIR / "images" / "test").resolve()),
        'nc': len(class_names),
        'names': class_names
    }
    with open(OUTPUT_DIR / "data.yaml", 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False)
    print(f"\nSuccessfully created 'data.yaml' for the final dataset.")
    
    # Phase 5: Demonstrate normalization on one sample image
    if train_paths:
        demonstrate_normalization(train_paths[0])

    print("\n✅ All preprocessing steps are complete!")

if __name__ == "__main__":
    main()