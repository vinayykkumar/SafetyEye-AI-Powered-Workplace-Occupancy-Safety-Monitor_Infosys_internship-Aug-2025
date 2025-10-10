import os
import random
from collections import Counter

labels_dir = "path/to/labels"
target_class = 5
target_count = 10000  # desired count

# Count current instances of class 5
class_counts = Counter()
label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]

for lf in label_files:
    with open(os.path.join(labels_dir, lf)) as f:
        for line in f:
            cls = int(line.split()[0])
            class_counts[cls] += 1

print("Before:", class_counts)

# Number of class 5 objects to remove
remove_count = class_counts[target_class] - target_count
print(f"Need to remove {remove_count} objects of class {target_class}")

if remove_count > 0:
    removed = 0
    for lf in label_files:
        path = os.path.join(labels_dir, lf)
        with open(path) as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            cls = int(line.split()[0])
            if cls == target_class and removed < remove_count:
                if random.random() < (remove_count / class_counts[target_class]):
                    removed += 1
                    continue  # skip this object
            new_lines.append(line)

        # overwrite label file
        with open(path, "w") as f:
            f.writelines(new_lines)

print("Removed:", removed)

# Recount
class_counts = Counter()
for lf in label_files:
    with open(os.path.join(labels_dir, lf)) as f:
        for line in f:
            cls = int(line.split()[0])
            class_counts[cls] += 1

print("After:", class_counts)
