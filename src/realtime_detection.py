import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from src.config import Config
from src.violation_logger import ViolationLogger
from mediapipe.python.solutions import pose as mp_pose
import logging

# Setup logging
logger = logging.getLogger(__name__)

class RealTimeDetector:
    def __init__(self):
        Config.validate_paths()
        if not os.path.exists(Config.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {Config.MODEL_PATH}")
        self.session = ort.InferenceSession(Config.MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.logger = ViolationLogger()
        self.class_names = Config.DETECTION_CLASSES
        self.input_size = Config.INPUT_SIZE
        self.frame_dir = os.path.join(Config.LOG_DIR, "frames")
        os.makedirs(self.frame_dir, exist_ok=True)
        self.mp_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        logger.info(f"Model input names: {[inp.name for inp in self.session.get_inputs()]}")

    def preprocess_frame(self, frame):
        try:
            img = cv2.resize(frame, self.input_size)
            img = img.astype(np.float32) / 255.0
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0)
            return img
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            return None

    def postprocess_output(self, output, frame_shape):
        # Same as dashboard's postprocess_output
        try:
            predictions = output[0]
            logger.info(f"Predictions shape: {predictions.shape}")

            if len(predictions.shape) == 3:
                if predictions.shape[2] == 84 and predictions.shape[1] == 8400:
                    boxes = predictions[0, :, :4]
                    scores = predictions[0, :, 4:4 + Config.TOTAL_CLASSES]
                    scores = np.max(scores, axis=1)
                    class_ids = np.argmax(scores, axis=1)
                    predictions = predictions[0]
                elif predictions.shape[1] == 6 or predictions.shape[2] == 6:
                    predictions = predictions.transpose(0, 2, 1) if predictions.shape[1] == 6 else predictions
                    boxes = predictions[:, :, :4]
                    scores = predictions[:, :, 4]
                    class_ids = predictions[:, :, 5].astype(int)
                else:
                    predictions = predictions.transpose(0, 2, 1) if predictions.shape[1] == 4 + Config.TOTAL_CLASSES else predictions
                    boxes = predictions[:, :, :4]
                    scores = predictions[:, :, 4:4 + Config.TOTAL_CLASSES]
                    scores = np.max(scores, axis=2)
                    class_ids = np.argmax(predictions[:, :, 4:], axis=2)
            else:
                raise ValueError("Unexpected predictions shape")

            mask = scores > Config.CONFIDENCE_THRESHOLD
            logger.info(f"Detections above threshold: {np.sum(mask)}")
            boxes = boxes[mask]
            scores = scores[mask]
            class_ids = class_ids[mask]

            if len(boxes) == 0:
                return np.array([]), np.array([]), np.array([])

            boxes[:, [0, 2]] = boxes[:, [0, 2]] * frame_shape[1] / self.input_size[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * frame_shape[0] / self.input_size[0]
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            class_ids = np.clip(class_ids, 0, Config.TOTAL_CLASSES - 1)
            return boxes, scores, class_ids
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            return np.array([]), np.array([]), np.array([])

    def draw_overlays(self, frame, boxes, scores, class_ids):
        # Same as dashboard's draw_overlays with debug logs
        violations = []
        total_persons = 0
        violation_counts = {cls: 0 for cls in Config.VIOLATION_CLASSES}
        logger.info(f"Processing {len(boxes)} detections")

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{self.class_names[class_id]}: {score:.2f}"
            severity = Config.SEVERITY_LEVELS.get(self.class_names[class_id], 'None')
            color = {'High': (255, 0, 0), 'Medium': (255, 255, 0), 'Low': (0, 0, 255), 'None': (0, 255, 0)}[severity]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({severity})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if self.class_names[class_id] == "Person":
                total_persons += 1
            elif self.class_names[class_id] in Config.VIOLATION_CLASSES and score > Config.ALERT_THRESHOLD:
                violations.append((self.class_names[class_id], score, severity, box))
                violation_counts[self.class_names[class_id]] += 1
        logger.info(f"Total persons: {total_persons}, Violations: {len(violations)}")
        return frame, violations, total_persons, violation_counts

    def analyze_posture(self, frame, boxes, class_ids):
        # Same as dashboard
        posture_violations = []
        for box, class_id in zip(boxes, class_ids):
            if self.class_names[class_id] == "Person":
                x1, y1, x2, y2 = map(int, box)
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size == 0:
                    continue
                results = self.mp_pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    if landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y > landmarks[mp_pose.PoseLandmark.LEFT_HIP].y + 0.2:
                        posture_violations.append(("Unsafe Posture: Bending", 0.9, "Medium", box))
        return posture_violations

    def process_video(self):
        cap = cv2.VideoCapture(Config.VIDEO_SOURCE)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video source: {Config.VIDEO_SOURCE}")

        frame_count = 0
        start_time = time.time()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            input_img = self.preprocess_frame(frame)
            if input_img is None:
                continue

            outputs = self.session.run(None, {"images": input_img})
            boxes, scores, class_ids = self.postprocess_output(outputs, frame.shape)
            frame, violations, total_persons, violation_counts = self.draw_overlays(frame, boxes, scores, class_ids)
            posture_violations = self.analyze_posture(frame, boxes, class_ids)
            violations.extend(posture_violations)

            # Log even if 0
            self.logger.log_violation("Person", total_persons, "None", frame_count, (0, 0, 0, 0), metadata={"total_persons": total_persons})
            for violation_type, count in violation_counts.items():
                if count > 0:
                    self.logger.log_violation(violation_type, count, Config.SEVERITY_LEVELS.get(violation_type, "None"), frame_count, (0, 0, 0, 0), metadata={"violation_count": count})
            for violation in violations:
                self.logger.log_violation(violation[0], violation[1], violation[2], frame_count, violation[3])

            frame_path = os.path.join(self.frame_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)

        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        self.logger.log_performance(fps, frame_count)
        cap.release()

# Usage: detector = RealTimeDetector(); detector.process_video()