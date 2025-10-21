import os

def check_project_structure():
    print("🔍 Checking your project structure...")
    
    # Check root folder
    root_files = os.listdir('.')
    print(f"\n📁 Root folder files: {root_files}")
    
    # Check detection folder
    detection_path = './detection'
    if os.path.exists(detection_path):
        detection_files = os.listdir(detection_path)
        print(f"\n📁 Detection folder files: {detection_files}")
        
        # Check for Python files in detection
        py_files = [f for f in detection_files if f.endswith('.py')]
        print(f"🐍 Python files in detection: {py_files}")
    else:
        print(f"\n❌ Detection folder not found at: {detection_path}")
    
    # Check dashboard folder
    dashboard_path = './dashboard'
    if os.path.exists(dashboard_path):
        dashboard_files = os.listdir(dashboard_path)
        print(f"\n📁 Dashboard folder files: {dashboard_files}")

if __name__ == "__main__":
    check_project_structure()