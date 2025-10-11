from src.detect_realtime import run_realtime_detection
from src.dashboard import run_dashboard
from src.detect_video import detect_on_video  # sahi import

import threading


def run_video_detection():
    detect_on_video(
        video_path="data/videos/istockphoto-1036333520-640_adpp_is.mp4",
        output_path="data/videos/output_demo.mp4",
        conf_threshold=0.5
    )


if __name__ == "__main__":
    t0 = threading.Thread(target=run_video_detection)  # video detect thread
    t1 = threading.Thread(target=run_realtime_detection)
    t2 = threading.Thread(target=run_dashboard)

    t0.start()
    t0.join()  # video detection poora hone ke baad baaki start hoga

    t1.start()
    t2.start()

    t1.join()
    t2.join()
