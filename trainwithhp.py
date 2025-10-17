if __name__ == "__main__":
    from ultralytics import YOLO

    # Load pretrained model
    model = YOLO("runs/detect/ppe_yolov8s/weights/best.pt")

    # Start training
    model.train(
         data=r"C:\Users\gaikw\OneDrive\Desktop\SafetyEye\processed\safetyeye_v1\data.yaml",  # your dataset YAML
    epochs=20,                 # short fine-tuning
    imgsz=718,                 # image size
    batch=16,                  # adjust for GPU memory
    lr0=0.001,                 # initial learning rate
    lrf=0.1,                   # final learning rate factor
    optimizer="AdamW",         # better generalization
    weight_decay=0.0005,       # regularization
    momentum=0.937,            # for AdamW
    augment=True,              # flips, mosaic, HSV
    mixup=0.2,                 # mixup augmentation
    copy_paste=0.1,            # copy-paste augmentation for rare classes
    patience=5,                # early stopping
    save_period=5,             # save weights every 5 epochs
    project="runs/detect/fine_tuned_v2",  # custom folder
    name="ppe_best_v2",                   # subfolder for this experiment
    exist_ok=True,                         # overwrite only this folder if exists
    device=0,                               # GPU
    save=True,
    verbose=True
    )
