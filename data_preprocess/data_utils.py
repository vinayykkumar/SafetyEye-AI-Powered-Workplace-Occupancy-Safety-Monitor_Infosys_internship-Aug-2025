# pairs the images and labels and removes duplicate images
import os
import glob
import hashlib
from typing import List, Tuple

def get_image_label_paths(base_dir: str) -> List[Tuple[str, str]]:
    """Pairs image and label files from a base directory."""
    image_dir = os.path.join(base_dir, 'images')
    label_dir = os.path.join(base_dir, 'labels')
    
    if not os.path.isdir(image_dir) or not os.path.isdir(label_dir):
        print(f"Error: 'images' or 'labels' directory not found in {base_dir}")
        return []
    
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*')))
    paired_data = []
    
    for img_path in image_paths:
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{img_basename}.txt")
        
        if os.path.exists(label_path):
            paired_data.append((img_path, label_path))
        else:
            print(f"Warning: No matching label found for {img_path}")
            
    return paired_data

def remove_duplicates(data_paths: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Removes duplicates based on file content using MD5 hashing."""
    unique_hashes = set()
    unique_data = []
    
    for img_path, label_path in data_paths:
        try:
            with open(img_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except IOError:
            print(f"Skipping corrupted or unreadable file: {img_path}")
            continue
            
        if file_hash not in unique_hashes:
            unique_hashes.add(file_hash)
            unique_data.append((img_path, label_path))
        else:
            print(f"Removed duplicate image: {img_path}")
            # You can also add os.remove(img_path) here if you want to delete the file
            
    return unique_data
    