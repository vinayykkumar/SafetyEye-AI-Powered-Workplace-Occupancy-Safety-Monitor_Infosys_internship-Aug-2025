import albumentations as A
import cv2
import os

# Example, if images are in unique_cleaned folder:
input_folder = 'data/unique_cleaned/images'
output_folder = 'data/augmented/images'

os.makedirs(output_folder, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Rotate(limit=20, p=0.5),

], p=1.0)


for fname in os.listdir(input_folder):
    img_path = os.path.join(input_folder, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Unable to read image {fname}")
        continue
    augmented = transform(image=img)['image']
    cv2.imwrite(os.path.join(output_folder, fname), augmented)

print("Augmentation complete with ±20 degree tilt and grayscale applied.")
