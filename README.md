# PPE Detection with YOLOv8

## Modules
- **data_prep**: Load & process YOLO-formatted images
- **model_training**: Train YOLOv8 for PPE detection
- **detection**: Real-time PPE violation spotting
- **alerts**: Send notifications for missing gear
- **dashboard**: Live view + compliance stats

## Setup
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```
pip install -r requirements.txt
   ```
3. Prepare your data in YOLO format under `data/`.
4. Edit and run scripts in each module as needed.
