# # Auto-orient / rotate ,resize,and normalization

import cv2
import numpy as np
from PIL import Image, ExifTags
from typing import Tuple, List

def auto_orient(image_path: str) -> np.ndarray:
    
    try:
        # Open the image with PIL to access EXIF data
        pil_image = Image.open(image_path)
        exif = pil_image._getexif()

        if exif is None:
            # If no EXIF data, return the image read by OpenCV
            return cv2.imread(image_path)
        
        # Get the orientation tag ID (usually 274)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        
        exif_orientation = exif.get(orientation)
        
        # Apply the correct rotation based on the EXIF tag
        if exif_orientation == 2:
            pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif exif_orientation == 3:
            pil_image = pil_image.transpose(Image.ROTATE_180)
        elif exif_orientation == 4:
            pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
        elif exif_orientation == 5:
            pil_image = pil_image.transpose(Image.TRANSPOSE)
        elif exif_orientation == 6:
            pil_image = pil_image.transpose(Image.ROTATE_270)
        elif exif_orientation == 7:
            pil_image = pil_image.transpose(Image.TRANSVERSE)
        elif exif_orientation == 8:
            pil_image = pil_image.transpose(Image.ROTATE_90)
        
        # Convert the corrected PIL image to a NumPy array in BGR format
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception:
        # Fallback to simple OpenCV read if any error occurs
        return cv2.imread(image_path)
    
def preprocess_image_and_label(image_path: str, label_path: str, target_size: Tuple[int, int] = (416, 416)) -> Tuple[np.ndarray, np.ndarray]:
    
    try:
        # 1. Auto-orient the image
        img = auto_orient(image_path)
        if img is None:
            return None, None
            
        original_height, original_width = img.shape[:2]

        # 2. Resize the image to the target size
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # 3. Read and scale the labels
        with open(label_path, 'r') as f:
            labels = [line.strip().split() for line in f.readlines()]
        
        scaled_labels = []
        for label in labels:
            class_id = int(label[0])
            x_center_norm, y_center_norm, width_norm, height_norm = map(float, label[1:])
            
            # Convert normalized coordinates to pixel coordinates
            x_center = x_center_norm * original_width
            y_center = y_center_norm * original_height
            width = width_norm * original_width
            height = height_norm * original_height

            # Scale pixel coordinates to the new image size
            scaled_x_center = x_center * (target_size[0] / original_width)
            scaled_y_center = y_center * (target_size[1] / original_height)
            scaled_width = width * (target_size[0] / original_width)
            scaled_height = height * (target_size[1] / original_height)

            # Convert back to new normalized coordinates
            new_x_center_norm = scaled_x_center / target_size[0]
            new_y_center_norm = scaled_y_center / target_size[1]
            new_width_norm = scaled_width / target_size[0]
            new_height_norm = scaled_height / target_size[1]

            scaled_labels.append([class_id, new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm])
        
        # 4. Normalize the image pixel values
        normalized_img = resized_img.astype('float32') / 255.0

        return normalized_img, np.array(scaled_labels)

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None