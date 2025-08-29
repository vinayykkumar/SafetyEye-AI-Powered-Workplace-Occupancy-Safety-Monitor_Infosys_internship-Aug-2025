import os
import shutil
from PIL import Image
import imagehash

image_dir = 'data/cleaned/images'
label_dir = 'data/cleaned/labels'
unique_dir = 'data/unique_cleaned/images'
unique_label_dir = 'data/unique_cleaned/labels'

os.makedirs(unique_dir, exist_ok=True)
os.makedirs(unique_label_dir, exist_ok=True)

hashes = {}
duplicates = []

def is_similar(hash1, hash2, threshold=5):
    return hash1 - hash2 < threshold

for fname in os.listdir(image_dir):
    fpath = os.path.join(image_dir, fname)
    img = Image.open(fpath)
    imghash = imagehash.phash(img)

    found_similar = False
    for h in hashes:
        if is_similar(imghash, h):
            found_similar = True
            duplicates.append(fname)
            break

    if not found_similar:
        hashes[imghash] = fname
        shutil.copy(fpath, os.path.join(unique_dir, fname))
        label_name = os.path.splitext(fname)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(unique_label_dir, label_name))

print(f"Removed {len(duplicates)} visually similar duplicate images. Cleaned data saved in data/unique_cleaned/")
