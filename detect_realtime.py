from ultralytics import YOLO
import cv2

def run_realtime_detection(conf_threshold=0.5):
    model_path = 'models/trained_model.pt'
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    print("Starting real-time detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert BGR to RGB before inference (YOLO expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = model(rgb_frame, conf=conf_threshold)

        annotated_frame = results[0].plot()

        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        cv2.imshow('SafetyEye Real-time Detection', annotated_frame_bgr)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting real-time detection.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realtime_detection()
