# augment_data.py

import os
import cv2
import albumentations as A
import random
import matplotlib.pyplot as plt

# --- CONFIGURATION ---


SOURCE_IMAGE_DIR = "source_data/images"
SOURCE_LABEL_DIR = "source_data/labels"


AUGMENTED_IMAGE_DIR = "augmented_data/images"
AUGMENTED_LABEL_DIR = "augmented_data/labels"

# New augmented versions to create
NUM_AUGMENTATIONS_PER_IMAGE = 5

CLASS_NAMES = ['hardhat', 'vest', 'worker']

# --- END OF CONFIGURATION ---


def load_yolo_annotations(label_path, image_height, image_width):
    """Loads YOLO annotations from a file and de-normalizes them."""
    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            bboxes.append([x_center, y_center, width, height])
            class_labels.append(int(class_id))
    return bboxes, class_labels

def save_yolo_annotations(label_path, bboxes, class_labels):
    """Saves transformed annotations in YOLO format."""
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.75),
    A.GaussianBlur(blur_limit=(3, 7), p=0.25),
    A.GaussNoise(p=0.25),
    A.ToGray(p=0.25),

    A.RandomSizedBBoxSafeCrop(width=640, height=640, erosion_rate=0.2, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


def augment_and_save():
    """Main function to run the augmentation pipeline."""
    os.makedirs(AUGMENTED_IMAGE_DIR, exist_ok=True)
    os.makedirs(AUGMENTED_LABEL_DIR, exist_ok=True)

    image_files = os.listdir(SOURCE_IMAGE_DIR)

    print(f"Starting augmentation... Generating {NUM_AUGMENTATIONS_PER_IMAGE} new samples per image.")
    
    for img_name in image_files:
        basename, ext = os.path.splitext(img_name)
        img_path = os.path.join(SOURCE_IMAGE_DIR, img_name)
        label_path = os.path.join(SOURCE_LABEL_DIR, basename + '.txt')

        if not os.path.exists(label_path):
            continue

        # Load the original image and its annotations
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        bboxes, class_labels = load_yolo_annotations(label_path, h, w)

        # Generate N augmented versions
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            # Create new unique filename for the augmented data
            new_basename = f"{basename}_aug_{i}"
            new_img_path = os.path.join(AUGMENTED_IMAGE_DIR, new_basename + ext)
            new_label_path = os.path.join(AUGMENTED_LABEL_DIR, new_basename + '.txt')

            # Save the new image and its corresponding labels
            cv2.imwrite(new_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            save_yolo_annotations(new_label_path, aug_bboxes, aug_class_labels)

    print("\nAugmentation complete!")
    print(f"New augmented data is saved in '{os.path.dirname(AUGMENTED_IMAGE_DIR)}'")



