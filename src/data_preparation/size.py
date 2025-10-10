import os
from PIL import Image

# Dynamically set dataset root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "archive", "data")

splits = ["train/images", "valid/images", "test/images"]

for split in splits:
    folder = os.path.join(DATASET_PATH, split)
    if not os.path.exists(folder):
        print(f"[⚠] {split} folder missing, skipping...")
        continue

    print(f"\n🔍 Checking {split} for image size consistency...")
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
        print(f"✅ YES — All images in {split} have the same size:", list(sizes.keys())[0])
    else:
        print(f"❌ NO — Images in {split} have different sizes:")
        for size, count in sizes.items():
            print(f"   {size} -> {count} images")
