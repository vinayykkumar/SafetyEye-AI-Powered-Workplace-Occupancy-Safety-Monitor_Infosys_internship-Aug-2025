# File: augment_and_save.py

import os
import cv2
import numpy as np
import albumentations as A

def augment_and_save_dataset(input_dir: str, output_dir: str, num_augmentations: int = 3):
   
    # Define the augmentation pipeline with all the requested transformations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
        A.Perspective(scale=(0.05, 0.1), p=0.3, keep_size=True, pad_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # Set up the input and output directories
    input_image_dir = os.path.join(input_dir, 'images')
    input_label_dir = os.path.join(input_dir, 'labels')

    output_image_dir = os.path.join(output_dir, 'images')
    output_label_dir = os.path.join(output_dir, 'labels')
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    print(f"Starting augmentation on data from '{input_image_dir}'...")

    image_files = os.listdir(input_image_dir)
    for image_file in image_files:
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(input_image_dir, image_file)
            label_path = os.path.join(input_label_dir, f"{base_name}.txt")

            if not os.path.exists(label_path):
                print(f"Warning: Skipping {image_file} as no label file was found.")
                continue

            # Load the original image and labels
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            with open(label_path, 'r') as f:
                bboxes_list = [list(map(float, line.strip().split())) for line in f.readlines()]
            
            class_labels = [int(bbox[0]) for bbox in bboxes_list]
            bboxes = [bbox[1:] for bbox in bboxes_list]

            # Apply and save augmented versions
            for i in range(num_augmentations):
                try:
                    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    aug_image_name = f"{base_name}_aug{i}.jpg"
                    aug_label_name = f"{base_name}_aug{i}.txt"

                    # Save the new augmented image
                    cv2.imwrite(os.path.join(output_image_dir, aug_image_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                    
                    # Save the new augmented label file
                    with open(os.path.join(output_label_dir, aug_label_name), 'w') as f:
                        for j, bbox in enumerate(transformed_bboxes):
                            class_id = transformed['class_labels'][j]
                            f.write(f"{class_id} {' '.join(map(str, bbox))}\n")
                except Exception as e:
                    print(f"Error augmenting {image_file} (run {i}): {e}")
            
    print(f"✅ Augmentation complete! Augmented files saved in '{output_dir}'.")

if __name__ == '__main__':
    # --- CHANGE THESE PATHS ---
    # The input folder is the 'train' folder within your processed dataset
    processed_train_path = 'processed_data/train'
    
    # The output folder is a new, separate folder for the augmented data
    augmented_data_output_path = 'augmented_training_data'

    # Run the augmentation process
    # This will create 5 augmented versions of each training image
    augment_and_save_dataset(processed_train_path, augmented_data_output_path, num_augmentations=5)