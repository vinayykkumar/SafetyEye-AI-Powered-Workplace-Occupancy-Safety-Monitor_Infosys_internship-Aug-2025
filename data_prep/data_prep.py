import os
import cv2
import pandas as pd
from pathlib import Path

# Data Prep: Load and process YOLO-formatted images
def load_yolo_images_labels(images_dir, labels_dir):
    images = list(Path(images_dir).glob('*.jpg'))
    labels = list(Path(labels_dir).glob('*.txt'))
    return images, labels

# Example usage:
# imgs, lbls = load_yolo_images_labels('data/train/images', 'data/train/labels')
