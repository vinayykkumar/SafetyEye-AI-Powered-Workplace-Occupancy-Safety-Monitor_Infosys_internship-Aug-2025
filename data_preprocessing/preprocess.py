# A script to check, clean, and split our PPE image dataset for YOLO training.

import os
import shutil
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


SOURCE_DIR = "source_data"
OUTPUT_DIR = "data\processed\ppe_dataset"
CLASS_NAMES = ['hardhat', 'vest', 'worker']

# Spliting the data: 70% for training, 20% for validation, 10% for testing.
SPLIT_RATIO = (0.7, 0.2, 0.1)


def process_and_split_data():
    """
    Finds all matching image-label pairs, checks them, and splits them
    into train, validation, and test sets.
    """
    print(f"Starting data processing from '{SOURCE_DIR}'...")
    
    source_images_path = os.path.join(SOURCE_DIR, "images")
    source_labels_path = os.path.join(SOURCE_DIR, "labels")

    # Finding all image files and assume their labels have the same name but with a .txt extension.
    all_images = os.listdir(source_images_path)
    
    # Checking for missing labels and corrupted images
    print(f"Found {len(all_images)} images. Verifying pairs and checking for corruption...")
    
    valid_files = []
    for img_name in tqdm(all_images, desc="Verifying files"):
        basename, _ = os.path.splitext(img_name)
        label_name = basename + ".txt"
        
        img_path = os.path.join(source_images_path, img_name)
        label_path = os.path.join(source_labels_path, label_name)

        # 1. Checking if the matching label file exists
        if not os.path.exists(label_path):
            continue

        # 2. Checking if the label file is empty
        if os.path.getsize(label_path) == 0:
            continue

        # 3. Checking if the image can be opened and is not corrupted
        try:
            with Image.open(img_path) as img:
                img.verify()
            valid_files.append(basename)
        except (IOError, SyntaxError):
            print(f"Skipping corrupted image: {img_name}")

    print(f"Found {len(valid_files)} clean, matching image-label pairs.")
    

    random.shuffle(valid_files)
    
    train_end = int(len(valid_files) * SPLIT_RATIO[0])
    val_end = train_end + int(len(valid_files) * SPLIT_RATIO[1])
    
    train_set = valid_files[:train_end]
    val_set = valid_files[train_end:val_end]
    test_set = valid_files[val_end:]
    
    # Create the new directory structure and copy files
    print(f"Creating new dataset at '{OUTPUT_DIR}'...")
    
    splits = {'train': train_set, 'val': val_set, 'test': test_set}
    
    for split_name, file_list in splits.items():
        # Create image and label folders for the split (e.g., '.../images/train')
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split_name), exist_ok=True)
        
        print(f"Copying {split_name} files...")
        for basename in tqdm(file_list, desc=f"Copying {split_name}"):
            # Find the original image file extension (.jpg, .png, etc.)
            original_img_name = ""
            for img_name in all_images:
                if os.path.splitext(img_name)[0] == basename:
                    original_img_name = img_name
                    break
            
            # Copy image and label
            shutil.copy(os.path.join(source_images_path, original_img_name), os.path.join(OUTPUT_DIR, 'images', split_name))
            shutil.copy(os.path.join(source_labels_path, basename + '.txt'), os.path.join(OUTPUT_DIR, 'labels', split_name))
            
    print("\nDataset split complete!")
    print(f"  - Training set: {len(train_set)} files")
    print(f"  - Validation set: {len(val_set)} files")
    print(f"  - Test set: {len(test_set)} files")



if __name__ == "__main__":
    process_and_split_data()
    print("\nAll done! Your dataset is ready.")