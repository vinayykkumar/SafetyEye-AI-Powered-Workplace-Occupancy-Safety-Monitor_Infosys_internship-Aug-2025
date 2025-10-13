import os
import cv2
import numpy as np

# Paths
AUG_IMAGE_DIR = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_images"
AUG_LABEL_DIR = r"C:\Users\mkr19\Desktop\SafetyEye\data\train\aug_labels"

# Function to normalize image
def normalize_image(img):
    """
    Input: img (OpenCV image, dtype=uint8, range 0-255)
    Output: normalized image (float32, range 0-1)
    """
    img = img.astype(np.float32) / 255.0  # scale to [0,1]
    return img

# Example: Load all augmented images and normalize in memory
normalized_images = []
image_files = [f for f in os.listdir(AUG_IMAGE_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

for img_file in image_files:
    img_path = os.path.join(AUG_IMAGE_DIR, img_file)
    img = cv2.imread(img_path)  # Load image (0-255)
    img_norm = normalize_image(img)  # Normalize to [0,1]
    normalized_images.append(img_norm)

    # Optional: check
    print(f"{img_file}: min={img_norm.min():.3f}, max={img_norm.max():.3f}")

print("✅ All images normalized in memory. Ready for model training!")
