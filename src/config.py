import os

class Config:
    # Define base directory relative to config.py
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Paths
    MODEL_PATH = os.path.join(BASE_DIR, 'outputs', 'runs', 'detect', 'yolov8l_ppe_detection_100_epochs', 'weights', 'best.onnx')
    DATASET_PATH = os.path.join(BASE_DIR, 'dataset')
    LOG_DIR = os.path.join(BASE_DIR, 'logs')
    VIOLATION_LOG_FILE = os.path.join(LOG_DIR, 'violations.csv')
    DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard')

    # Detection settings
    DETECTION_CLASSES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
                         'NO-Safety Vest', 'Person', 'Safety Cone',
                         'Safety Vest', 'machinery', 'vehicle']
    TOTAL_CLASSES = len(DETECTION_CLASSES)
    CONFIDENCE_THRESHOLD = 0.4
    INPUT_SIZE = (640, 640)
    
    # Violation rules
    VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
    ALERT_THRESHOLD = 0.5
    SEVERITY_LEVELS = {
        'NO-Hardhat': 'High',
        'NO-Mask': 'Low',
        'NO-Safety Vest': 'Medium',
        'Unsafe Posture: Bending': 'Medium'
    }

    # Video settings
    VIDEO_SOURCE = 0
    FPS = 15

    # Dashboard settings
    DASHBOARD_PORT = 8501

    @staticmethod
    def validate_paths():
        """Validate that critical paths exist."""
        os.makedirs(Config.LOG_DIR, exist_ok=True)  # Ensure LOG_DIR exists early
        for path in [Config.MODEL_PATH, Config.DATASET_PATH, Config.LOG_DIR, Config.DASHBOARD_DIR]:
            if not os.path.exists(os.path.dirname(path)):
                raise FileNotFoundError(f"Directory does not exist: {os.path.dirname(path)}")