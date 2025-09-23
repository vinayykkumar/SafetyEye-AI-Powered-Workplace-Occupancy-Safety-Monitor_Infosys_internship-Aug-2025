import os

# ⚙️ SETTINGS
labels_dir = r"C:\Users\yuvan\SAFETYEYE\Dataset\labels"  # path to your labels folder
keep_classes = [0, 1]   # change to [] if you want to keep ALL 10 classes

def clean_labels(path, keep_classes):
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                with open(file_path, "r") as f:
                    lines = f.readlines()

                cleaned = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        cls = int(parts[0])
                        # Keep only allowed classes (or all if keep_classes is empty)
                        if not keep_classes or cls in keep_classes:
                            cleaned.append(line)

                # Overwrite file with cleaned content
                with open(file_path, "w") as f:
                    f.writelines(cleaned)

                # If file becomes empty → delete it
                if not cleaned:
                    print(f"⚠️ Removed empty label file: {file_path}")
                    os.remove(file_path)

    print("✅ Label cleaning complete!")

# Run
clean_labels(labels_dir, keep_classes)
