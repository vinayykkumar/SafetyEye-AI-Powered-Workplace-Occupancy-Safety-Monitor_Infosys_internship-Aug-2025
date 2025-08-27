import os
from collections import Counter

# Paths
DATASET_PATH = "data/processed"
SPLITS = ["train", "val", "test"]  

def count_classes(lbl_dir):
    counter = Counter()
    if not os.path.exists(lbl_dir):
        print(f"⚠️ Missing: {lbl_dir}")
        return counter

    for lbl_file in os.listdir(lbl_dir):
        if lbl_file.endswith(".txt"):
            with open(os.path.join(lbl_dir, lbl_file), "r") as f:
                for line in f:
                    cls_id = line.strip().split()[0]
                    counter[cls_id] += 1
    return counter


print("📊 Checking class balance...\n")
for split in SPLITS:
    lbl_dir = os.path.join(DATASET_PATH, split, "labels")
    counter = count_classes(lbl_dir)

    if not counter:
        continue

    total = sum(counter.values())
    print(f"📂 {split.upper()} - Total labels: {total}")
    for cls_id, count in counter.items():
        pct = (count / total) * 100
        print(f"   Class {cls_id}: {count} ({pct:.2f}%)")
    print()
