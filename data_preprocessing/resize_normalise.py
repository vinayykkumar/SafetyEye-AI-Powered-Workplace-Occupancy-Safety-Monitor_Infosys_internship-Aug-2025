# A simple script to resize a YOLO dataset while keeping labels correct.

import os
import cv2
import shutil
from tqdm import tqdm
import numpy as np

# --- Settings ---
SOURCE_DATASET = "data/processed/final_dataset"
RESIZED_DATASET = "data/processed/final_dataset_resized"
IMG_SIZE = 640
# --- End of Settings ---


def normalize_pixel_values(image):
    # Convert the image data type to float32 for division
    image = image.astype(np.float32)
    # Divide by 255 to scale pixels to the 0.0-1.0 range
    image /= 255.0
    return image


# Clean up any old resized data first
if os.path.exists(RESIZED_DATASET):
    print(f"Removing old directory: {RESIZED_DATASET}")
    shutil.rmtree(RESIZED_DATASET)

# Loop through the train, validation, and test sets
for split in ['train', 'val', 'test']:
    
    source_img_folder = os.path.join(SOURCE_DATASET, 'images', split)
    source_label_folder = os.path.join(SOURCE_DATASET, 'labels', split)
    
    # Skip if a split folder (like 'test') doesn't exist
    if not os.path.exists(source_img_folder):
        continue
        
    # Create the new folders for our resized data
    resized_img_folder = os.path.join(RESIZED_DATASET, 'images', split)
    resized_label_folder = os.path.join(RESIZED_DATASET, 'labels', split)
    os.makedirs(resized_img_folder, exist_ok=True)
    os.makedirs(resized_label_folder, exist_ok=True)
    
    print(f"\nProcessing {split} set...")
    
    # Go through every image in the folder
    for filename in tqdm(os.listdir(source_img_folder)):
        
        # --- 1. Resize the image with padding ---
        
        # Load the original image
        img_path = os.path.join(source_img_folder, filename)
        img = cv2.imread(img_path)

        # --- DEMONSTRATION of Pixel Normalization ---
        normalized_img = normalize_pixel_values(img)
        
        old_h, old_w, _ = img.shape
        
        # Figure out the new size while keeping the aspect ratio
        scale = min(IMG_SIZE / old_w, IMG_SIZE / old_h)
        new_w, new_h = int(old_w * scale), int(old_h * scale)
        resized_img = cv2.resize(img, (new_w, new_h))
        
        # Add gray padding to make it exactly 640x640
        pad_top = (IMG_SIZE - new_h) // 2
        pad_bottom = IMG_SIZE - new_h - pad_top
        pad_left = (IMG_SIZE - new_w) // 2
        pad_right = IMG_SIZE - new_w - pad_left
        
        padded_img = cv2.copyMakeBorder(resized_img, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=[114, 114, 114])
        
        # Save the new resized image (still in 0-255 format)
        cv2.imwrite(os.path.join(resized_img_folder, filename), padded_img)
        
        # --- 2. Recalculate the labels ---
        
        basename, _ = os.path.splitext(filename)
        label_path = os.path.join(source_label_folder, basename + '.txt')
        
        if os.path.exists(label_path):
            new_labels = []
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.split())
                    
                    # Math to convert old label to new label
                    new_x = (x * old_w * scale + pad_left) / IMG_SIZE
                    new_y = (y * old_h * scale + pad_top) / IMG_SIZE
                    new_w = (w * old_w * scale) / IMG_SIZE
                    new_h = (h * old_h * scale) / IMG_SIZE
                    
                    new_labels.append(f"{int(class_id)} {new_x} {new_y} {new_w} {new_h}\n")
            
            # Save the new resized label
            with open(os.path.join(resized_label_folder, basename + '.txt'), 'w') as f:
                f.writelines(new_labels)

print(f"\nDone! Resized dataset is ready in '{RESIZED_DATASET}'.")
