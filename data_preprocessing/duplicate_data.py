# deduplicate_data.py

import os
import shutil
from PIL import Image
import imagehash
from tqdm import tqdm


SOURCE_IMAGES_DIR = "raw_unfiltered/images"
SOURCE_LABELS_DIR = "raw_unfiltered/labels"

DEST_IMAGES_DIR = "source_data/images"
DEST_LABELS_DIR = "source_data/labels"

SIMILARITY_THRESHOLD = 7




def main():
    print("----- Starting Data Deduplication Pipeline -----")

    os.makedirs(DEST_IMAGES_DIR, exist_ok=True)
    os.makedirs(DEST_LABELS_DIR, exist_ok=True)

    try:
        image_files = sorted(os.listdir(SOURCE_IMAGES_DIR))
    except FileNotFoundError:
        print(f"[ERROR] Source image directory not found: '{SOURCE_IMAGES_DIR}'")
        return

    kept_image_hashes = []
    files_kept = 0
    files_skipped = 0

    print(f"Scanning {len(image_files)} images to remove near-duplicates...")

    for filename in tqdm(image_files, desc="Deduplicating Images"):
        basename, _ = os.path.splitext(filename)
        source_image_path = os.path.join(SOURCE_IMAGES_DIR, filename)
        

        label_filename = basename + '.txt'
        source_label_path = os.path.join(SOURCE_LABELS_DIR, label_filename)
        if not os.path.exists(source_label_path):
            continue # Skip if no label

        try:

            with Image.open(source_image_path) as img:
                current_hash = imagehash.phash(img)
        except Exception as e:
            print(f"\n[WARNING] Could not process image {filename}, skipping. Error: {e}")
            continue

        is_duplicate = False
        for kept_hash in kept_image_hashes:
            if (current_hash - kept_hash) < SIMILARITY_THRESHOLD:
                is_duplicate = True
                break
        
        if not is_duplicate:

            kept_image_hashes.append(current_hash)

            shutil.copy(source_image_path, os.path.join(DEST_IMAGES_DIR, filename))
            shutil.copy(source_label_path, os.path.join(DEST_LABELS_DIR, label_filename))
            
            files_kept += 1
        else:
            files_skipped += 1

    print("\n----- Data Deduplication Complete! -----")
    print(f"Total files processed: {len(image_files)}")
    print(f"Files kept: {files_kept}")
    print(f"Files skipped (duplicates): {files_skipped}")
    print(f"Clean, unique data is now ready in the '{os.path.dirname(DEST_IMAGES_DIR)}' directory.")

if __name__ == "__main__":
    main()