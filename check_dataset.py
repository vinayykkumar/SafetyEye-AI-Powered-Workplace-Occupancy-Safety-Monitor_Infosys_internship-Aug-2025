import os

# 🔹 Change this to your dataset path
dataset_path = r"C:/Users/yuvan/SAFETYEYE/dataset"

# Define image and label folders
train_img = os.path.join(dataset_path, "images/train")
val_img = os.path.join(dataset_path, "images/val")
train_lbl = os.path.join(dataset_path, "labels/train")
val_lbl = os.path.join(dataset_path, "labels/val")

def check_dataset(img_folder, lbl_folder, split_name):
    print(f"\n🔎 Checking {split_name} dataset...")
    img_files = {os.path.splitext(f)[0] for f in os.listdir(img_folder) if f.endswith(('.jpg', '.png', '.jpeg'))}
    lbl_files = {os.path.splitext(f)[0] for f in os.listdir(lbl_folder) if f.endswith('.txt')}

    # Missing labels
    missing_labels = img_files - lbl_files
    if missing_labels:
        print(f"❌ {len(missing_labels)} images missing labels:", list(missing_labels)[:5])
    else:
        print("✅ All images have labels")

    # Missing images
    missing_images = lbl_files - img_files
    if missing_images:
        print(f"❌ {len(missing_images)} labels missing images:", list(missing_images)[:5])
    else:
        print("✅ All labels have images")

    # Empty labels
    empty_labels = []
    for file in os.listdir(lbl_folder):
        if file.endswith(".txt"):
            path = os.path.join(lbl_folder, file)
            if os.path.getsize(path) == 0:
                empty_labels.append(file)
    if empty_labels:
        print(f"⚠️ {len(empty_labels)} empty label files:", empty_labels[:5])
    else:
        print("✅ No empty labels")

# Run checks
check_dataset(train_img, train_lbl, "TRAIN")
check_dataset(val_img, val_lbl, "VAL")
