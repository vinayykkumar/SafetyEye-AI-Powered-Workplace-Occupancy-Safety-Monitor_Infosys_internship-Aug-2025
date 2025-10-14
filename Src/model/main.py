import os
import cv2
import torch
import random
import shutil
from dataprep import preprocess_image_cv2, preprocess_image_torch

def process_and_split_dataset(input_folder, output_folder, use_torch=False, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    os.makedirs(output_folder, exist_ok=True)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_folder, split, "images"), exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    total = len(image_files)
    random.shuffle(image_files)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    splits = {
        "train": image_files[:train_end],
        "val": image_files[train_end:val_end],
        "test": image_files[val_end:]
    }

    processed, failed = 0, 0

    for split, files in splits.items():
        for filename in files:
            image_path = os.path.join(input_folder, filename)
            try:
                if use_torch:
                    img_tensor = preprocess_image_torch(image_path)
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255).astype("uint8")
                else:
                    img_np = preprocess_image_cv2(image_path, target_size=(224, 224))
                    img_np = (img_np * 255).astype("uint8")

                save_path = os.path.join(output_folder, split, "images", filename)
                cv2.imwrite(save_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

                processed += 1
                print(f"✅ {split} | {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                failed += 1

    print("\n------ Processing Summary ------")
    print(f"📂 Input folder : {input_folder}")
    print(f"💾 Output folder: {output_folder}")
    print(f"🖼️ Total images : {total}")
    print(f"✅ Processed    : {processed}")
    print(f"❌ Failed       : {failed}")
    print("--------------------------------")
    print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

def main():
    input_folder = r"D:/CompanyProject/SafetyEye/Datasets"
    output_folder = r"D:/CompanyProject/SafetyEye/preprocessing/splitted_dataset"

    process_and_split_dataset(input_folder, output_folder, use_torch=False)

if __name__ == "__main__":
    main()
