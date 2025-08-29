from src.detect_realtime import run_realtime_detection
from src.dashboard import run_dashboard
import threading

if __name__ == "__main__":
    t1 = threading.Thread(target=run_realtime_detection)
    t2 = threading.Thread(target=run_dashboard)
    t1.start()
    t2.start()
