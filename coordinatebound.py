import os

# ---------------- CONFIG ----------------
dataset_root = r"C:\Users\gaikw\OneDrive\Desktop\SafetyEye\processed\safetyeye_v1"
num_classes = 14   # total valid classes (0 to 13)
folders = ["train/labels", "valid/labels", "test/labels"]  # adjust if needed
log_file = os.path.join(dataset_root, "label_cleaning_log.txt")
# ----------------------------------------

def clean_label_file(label_path, num_classes):
    removed_lines = 0
    fixed_lines = 0
    new_lines = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            removed_lines += 1
            continue

        try:
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:]))
        except:
            removed_lines += 1
            continue

        # Remove invalid class IDs
        if cls_id < 0 or cls_id >= num_classes:
            removed_lines += 1
            continue

        # Clip coordinates to [0,1]
        new_coords = [min(max(c, 0), 1) for c in coords]
        if new_coords != coords:
            fixed_lines += 1

        new_lines.append(f"{cls_id} {' '.join(map(str, new_coords))}\n")

    # Write cleaned labels back
    with open(label_path, "w") as f:
        f.writelines(new_lines)

    return removed_lines, fixed_lines

# ---------------- MAIN ----------------
total_removed = 0
total_fixed = 0

with open(log_file, "w") as log:
    for folder in folders:
        full_path = os.path.join(dataset_root, folder)
        if not os.path.exists(full_path):
            continue
        for file_name in os.listdir(full_path):
            if file_name.endswith(".txt"):
                label_path = os.path.join(full_path, file_name)
                removed, fixed = clean_label_file(label_path, num_classes)
                total_removed += removed
                total_fixed += fixed
                if removed or fixed:
                    log.write(f"{label_path}: removed {removed}, fixed {fixed}\n")

print(f"Done! Total removed lines: {total_removed}, total fixed lines: {total_fixed}")
print(f"See log: {log_file}")

#This script makes sure all your YOLO labels are:

#Correctly formatted
#Valid class IDs

#Normalized bounding boxes