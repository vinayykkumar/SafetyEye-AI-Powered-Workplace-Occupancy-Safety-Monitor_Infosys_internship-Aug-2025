from ultralytics import YOLO
import cv2

def detect_on_video(video_path, output_path=None, conf_threshold=0.6):
    model = YOLO("../models/trained_model.pt")
    cap = cv2.VideoCapture(video_path)

    # Optional: configure output video
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    else:
        out = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf_threshold)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes else []
        classes = results[0].boxes.cls.cpu().numpy() if len(results[0].boxes.cls) > 0 else []
        confs = results[0].boxes.conf.cpu().numpy() if len(results[0].boxes.conf) > 0 else []

        for box, cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls_id)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("PPE Video Detection", frame)
        if out:
            out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting video detection.")
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_on_video("data/videos/istockphoto-1036333520-640_adpp_is.mp4", output_path="data/videos/output_demo.mp4", conf_threshold=0.5)
