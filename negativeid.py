import os

data_dir = "C:/Users/gaikw/OneDrive/Desktop/SafetyEye/processed/safetyeye_v1/train/labels"
nc = 14  # number of classes

removed_count = 0

for file in os.listdir(data_dir):
    if file.endswith(".txt"):
        path = os.path.join(data_dir, file)
        new_lines = []
        remove_file = False

        with open(path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    remove_file = True
                    break
                class_id, x, y, w, h = parts
                class_id = int(float(class_id))
                x, y, w, h = map(float, [x, y, w, h])

                if class_id < 0 or class_id >= nc or not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    remove_file = True
                    break

                new_lines.append(f"{class_id} {x} {y} {w} {h}\n")

        if remove_file:
            os.remove(path)
            removed_count += 1
        else:
            with open(path, "w") as f:
                f.writelines(new_lines)

print(f"Cleaning finished. Removed {removed_count} label files.")

#removes entire label files if they contain invalid classes or out-of-bound boxes.