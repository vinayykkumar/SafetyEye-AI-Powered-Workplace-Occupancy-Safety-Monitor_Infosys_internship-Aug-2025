from ultralytics import YOLO
import json

def save_validation_metrics(model_path, data_yaml, output_file="validation_metrics.json"):
    # Load the trained YOLO model
    model = YOLO(model_path)

    # Run validation on the dataset specified in data.yaml
    metrics = model.val(data=data_yaml)

    # Extract required metrics from the results dictionary safely
    results_dict = getattr(metrics, 'results_dict', {})
    speed_dict = getattr(metrics, 'speed', {})

    # Prepare dictionary of metrics to save
    metrics_to_save = {
        "precision(B)": float(results_dict.get("metrics/precision(B)", 0)),
        "recall(B)": float(results_dict.get("metrics/recall(B)", 0)),
        "mAP50(B)": float(results_dict.get("metrics/mAP50(B)", 0)),
        "mAP50-95(B)": float(results_dict.get("metrics/mAP50-95(B)", 0)),
        "speed_preprocess": float(speed_dict.get("preprocess", 0)),
        "speed_inference": float(speed_dict.get("inference", 0)),
        "speed_postprocess": float(speed_dict.get("postprocess", 0))
    }

    # Save the metrics dictionary as pretty JSON
    with open(output_file, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"Validation metrics saved to {output_file}")
    print(metrics_to_save)

if __name__ == "__main__":
    model_path = "../models/trained_model.pt"  # Ensure path is correct to your saved weights
    data_yaml = "configs/data.yaml"
    save_validation_metrics(model_path, data_yaml)
