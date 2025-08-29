import os

# Base directory of your dataset
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../archive/data"))
splits = ["train", "valid", "test"]

for split in splits:
    img_dir = os.path.join(BASE_DIR, split, "images")
    lbl_dir = os.path.join(BASE_DIR, split, "labels")
    
    print(f"Checking: {img_dir}")
    print(f"Checking: {lbl_dir}")

    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"[⚠️] {split} folder missing, skipping...")
        continue

    num_imgs = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))])
    num_lbls = len([f for f in os.listdir(lbl_dir) if f.lower().endswith('.txt')])

    print(f"[✅] {split}: {num_imgs} images, {num_lbls} labels")
