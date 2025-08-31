import os

# Root project folder
root = r"D:\SAFETY_EYE"   # <-- your path

# Folder structure
folders = [
    ".github/workflows",
    ".vscode",
    "alerts",
    "dashboard/components",
    "data/processed",
    "data/raw",
    "data_preprocess",
    "detection",
    "models",
    "notebooks",
    "outputs",
    "src"
]

# Files to create
files = [
    ".github/workflows/ci-cd.yml",
    ".vscode/settings.json",
    "dashboard/app.py",
    "dashboard/utils.py",
    "data/data_preparation.py",
    "data/preprocess.py",
    "data/splitter.py",
    "data_preprocess/image.png",
    "data_preprocess/videos.mp4",
    "augmentation.py",
    "evaluate.py",
    "train.py",
    ".dockerignore",
    "config.yaml",
    "Dockerfile",
    "main.py",
    "README.md",
    "requirements.txt"
]

# Create folders
for folder in folders:
    os.makedirs(os.path.join(root, folder), exist_ok=True)

# Create empty files
for file in files:
    file_path = os.path.join(root, file)
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("")

print(f"✅ Project structure created at: {root}")