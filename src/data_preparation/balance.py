import os
from collections import Counter
import matplotlib.pyplot as plt

# ✅ List your YOLO labels folders here
labels_folders = [
    "../../archive/data/train/labels",
    "../../archive/data/train/labels_aug3"
]

# Count objects per class
class_counts = Counter()

for folder in labels_folders:
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            with open(os.path.join(folder, file), "r") as f:
                for line in f:
                    class_id = int(float(line.split()[0]))  # convert '0.0' → 0
                    class_counts[class_id] += 1

# Print counts
print("📊 Class distribution:")
for cls, count in sorted(class_counts.items()):
    print(f"Class {cls}: {count} objects")

# Plot distribution
classes = list(class_counts.keys())
counts = list(class_counts.values())

plt.figure(figsize=(10,5))
plt.bar(classes, counts, color='skyblue')
plt.xlabel("Class ID")
plt.ylabel("Number of Objects")
plt.title("YOLO Dataset Class Distribution")
plt.show()
