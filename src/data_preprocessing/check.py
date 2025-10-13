import os, glob, collections

# Update these paths to your dataset folders
sets = {
    "train": r"C:\Users\mkr19\Desktop\SafetyEye\data\train",
    "val":   r"C:\Users\mkr19\Desktop\SafetyEye\data\valid",
    "test":  r"C:\Users\mkr19\Desktop\SafetyEye\data\test"
}

def analyze(set_path):
    label_dir = os.path.join(set_path, "labels")
    img_dir   = os.path.join(set_path, "images")
    counts = collections.Counter()
    img_with_labels = 0
    total_imgs = len(glob.glob(f"{img_dir}/*"))
    background_imgs = 0

    for f in glob.glob(f"{label_dir}/*.txt"):
        with open(f) as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        if not lines:
            background_imgs += 1
            continue
        img_with_labels += 1
        for line in lines:
            cls = int(line.split()[0])
            counts[cls] += 1

    label_files = set(os.path.basename(p).replace('.txt','') for p in glob.glob(f"{label_dir}/*.txt"))
    img_files   = set(os.path.basename(p).split('.')[0] for p in glob.glob(f"{img_dir}/*"))
    no_label_images = len(img_files - label_files)
    background_imgs += no_label_images

    print(f"\n=== {os.path.basename(set_path)} ===")
    print(f"Total images: {total_imgs}")
    print(f"Images with >=1 label: {img_with_labels}")
    print(f"Background / no-label images: {background_imgs}")
    print("Labels per class (counts):")
    for k in sorted(counts.keys()):
        print(f"  class {k}: {counts[k]}")
    return counts

# Run analysis for each set
for name, path in sets.items():
    analyze(path)
