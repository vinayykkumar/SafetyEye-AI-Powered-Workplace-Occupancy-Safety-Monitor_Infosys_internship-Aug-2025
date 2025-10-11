import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_dataset(raw_dir, processed_dir, test_size=0.2, val_size=0.1):
    images = [f for f in os.listdir(os.path.join(raw_dir, "images")) if f.endswith((".jpg", ".png"))]

    train_imgs, test_imgs = train_test_split(images, test_size=test_size, random_state=42)
    train_imgs, val_imgs = train_test_split(train_imgs, test_size=val_size, random_state=42)

    for split, files in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        img_dir = os.path.join(processed_dir, split, "images")
        lbl_dir = os.path.join(processed_dir, split, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for f in files:
            label = f.rsplit(".", 1)[0] + ".txt"
            src_img = os.path.join(raw_dir, "images", f)
            src_lbl = os.path.join(raw_dir, "labels", label)
            dst_img = os.path.join(img_dir, f)
            dst_lbl = os.path.join(lbl_dir, label)
            if os.path.exists(src_lbl):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_lbl, dst_lbl)
            else:
                print(f"[WARNING] Skipping image (no label found): {f}")

    print("[INFO] Dataset prepared and split successfully!")

if __name__ == "__main__":
    prepare_dataset(raw_dir="data/unique_cleaned", processed_dir="data/processed")
