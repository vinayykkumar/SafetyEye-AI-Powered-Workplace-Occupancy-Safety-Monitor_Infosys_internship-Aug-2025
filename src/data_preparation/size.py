import os
from PIL import Image

# Root dataset path
root_path = "../../archive/data"

# Folders to check
splits = ["train/images", "valid/images", "test/images"]

for split in splits:
    folder = os.path.join(root_path, split)
    if not os.path.exists(folder):
        print(f"[⚠️] {split} folder missing, skipping...")
        continue

    print(f"\n🔍 Checking {split}...")
    sizes = {}
    for file in os.listdir(folder):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(folder, file)
            try:
                with Image.open(path) as img:
                    size = img.size  # (width, height)
                    sizes[size] = sizes.get(size, 0) + 1
            except Exception as e:
                print(f"[Error] Could not open {file}: {e}")

    if len(sizes) == 1:
        print("✅ YES — All images have the same size:", list(sizes.keys())[0])
    else:
        print("❌ NO — Images have different sizes:")
        for size, count in sizes.items():
            print(f"   {size} -> {count} images")
