import cv2
import os
import albumentations as A
from tqdm import tqdm

# Input and output directories
input_dir = r"C:\safetyeye\preprocessed\train\images"
output_dir = r"C:\safetyeye\preprocessed\augmented\train\images"

# Make sure output folder exists
os.makedirs(output_dir, exist_ok=True)

# Define augmentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5),
    A.RandomCrop(width=224, height=224, p=0.3),
    A.ColorJitter(p=0.3),
])

# Loop through all images
for img_name in tqdm(os.listdir(input_dir)):
    img_path = os.path.join(input_dir, img_name)

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        continue
    
    # Apply augmentation multiple times
    for i in range(3):  # Generate 3 augmented versions per image
        augmented = transform(image=image)
        aug_img = augmented["image"]

        # Save augmented image
        new_name = f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
        cv2.imwrite(os.path.join(output_dir, new_name), aug_img)
