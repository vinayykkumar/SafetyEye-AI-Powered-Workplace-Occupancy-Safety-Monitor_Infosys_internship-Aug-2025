import os
import cv2
from collections import Counter

DATASET_PATH = "data/processed"

print("Checking image sizes...")

all_sizes = []

for split in ["train", "val", "test"]:
    img_dir = os.path.join(DATASET_PATH, split, "images")
    
    if not os.path.exists(img_dir):
        print(f" {split}/images not found")
        continue
    
    img_files = [f for f in os.listdir(img_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"{split}: {len(img_files)} images")
    
    
    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is not None:
            h, w = img.shape[:2]
            all_sizes.append((w, h))

if all_sizes:
    size_counts = Counter(all_sizes)
    print(f"\nImage sizes found:")
    for size, count in size_counts.most_common():
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    print(f"\nTotal unique sizes: {len(size_counts)}")
else:
    print(" No images found!")
