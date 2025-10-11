from ultralytics import YOLO
import cv2
import numpy as np
from plyer import notification
import time
import json
import os

from src.sort import Sort



def save_violation_log(log_list, filename="violation_log.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            old_logs = json.load(f)
        combined_logs = old_logs + log_list
    else:
        combined_logs = log_list
    unique_logs = [dict(t) for t in {tuple(d.items()) for d in combined_logs}]
    with open(filename, 'w') as f:
        json.dump(unique_logs, f, indent=4)


def run_realtime_detection(conf_threshold=0.6):
    model_path = "../models/trained_model.pt"
    model = YOLO(model_path)
    class_names = model.names
    class_name_map = {name: idx for idx, name in class_names.items()}
    helmet_class_id = class_name_map.get('helmet', 0)
    vest_class_id = class_name_map.get('vest', 1)
    color_map = {
        'helmet': (0, 255, 0),  # Green BGR
        'vest': (255, 0, 0),  # Blue
        'worker': (0, 255, 255)  # Yellow
    }

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    tracker = Sort()

    # Alert cooldown dictionary with per-track-id timestamps
    alert_cooldowns = {
        "helmet": 10,
        "vest": 10,
    }
    last_alert_time = {
        "helmet": {},
        "vest": {},
    }

    violation_logs = []

    print("Starting real-time detection and tracking. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb_frame, conf=conf_threshold)

        detected_boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        detected_classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes.cls) > 0 else []
        confidences = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes.conf) > 0 else []

        detections = []
        for box, conf in zip(detected_boxes, confidences):
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf])
        detections = np.array(detections)

        tracked_objects = tracker.update(detections)

        def iou(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            return interArea / float(boxAArea + boxBArea - interArea)

        tracked_labels = []
        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = obj.astype(int)
            best_iou = 0
            best_idx = -1
            for i, det_box in enumerate(detected_boxes):
                curr_iou = iou([x1, y1, x2, y2], det_box)
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_idx = i
            if best_iou > 0.3 and best_idx != -1:
                cls_id = detected_classes[best_idx]
                conf = confidences[best_idx]
            else:
                cls_id = -1
                conf = 0
            tracked_labels.append((track_id, cls_id, conf, (x1, y1, x2, y2)))

        current_time = time.time()
        current_time_str = time.strftime("%Y-%m-%d %H:%M:%S")

        # Group tracked labels by track_id
        labels_by_id = {}
        for track_id, cls_id, conf, box in tracked_labels:
            if track_id not in labels_by_id:
                labels_by_id[track_id] = []
            labels_by_id[track_id].append(cls_id)

        # Check violations per track_id and alert with cooldown per track id
        for track_id, classes in labels_by_id.items():
            classes_set = set(classes)
            helmet_present = helmet_class_id in classes_set
            vest_present = vest_class_id in classes_set

            if not helmet_present:
                last_alert = last_alert_time["helmet"].get(track_id, 0)
                if current_time - last_alert > alert_cooldowns["helmet"]:
                    print(f"ALERT: Helmet missing for ID {track_id}")
                    notification.notify(title="PPE Alert", message=f"Helmet missing for ID {track_id}", timeout=3)
                    last_alert_time["helmet"][track_id] = current_time
                    violation_logs.append({"Time": current_time_str, "ID": track_id, "Type": "Helmet Missing"})

            if not vest_present:
                last_alert = last_alert_time["vest"].get(track_id, 0)
                if current_time - last_alert > alert_cooldowns["vest"]:
                    print(f"ALERT: Vest missing for ID {track_id}")
                    notification.notify(title="PPE Alert", message=f"Vest missing for ID {track_id}", timeout=3)
                    last_alert_time["vest"][track_id] = current_time
                    violation_logs.append({"Time": current_time_str, "ID": track_id, "Type": "Vest Missing"})

        # Draw bounding boxes and labels with track IDs
        for track_id, cls_id, conf, box in tracked_labels:
            x1, y1, x2, y2 = box
            class_name = model.names[int(cls_id)] if cls_id != -1 else "Unknown"
            color = color_map.get(class_name, (255, 255, 255))
            label = f"{class_name} ID: {track_id} {conf:.2f}" if cls_id != -1 else f"ID: {track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("PPE Real-Time Detection with SORT", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting real-time detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the violations log
    if violation_logs:
        save_violation_log(violation_logs)


if __name__ == "__main__":
    run_realtime_detection()
