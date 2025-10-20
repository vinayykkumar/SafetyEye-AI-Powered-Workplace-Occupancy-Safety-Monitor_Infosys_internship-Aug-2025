# find_real_code.py
import os

def find_working_files():
    print("🔍 Finding your actual working files...")
    
    # Check predictions folder
    print(f"\n📁 PREDICTIONS Folder:")
    predictions_path = './predictions'
    if os.path.exists(predictions_path):
        files = os.listdir(predictions_path)
        py_files = [f for f in files if f.endswith('.py')]
        ipynb_files = [f for f in files if f.endswith('.ipynb')]
        
        print(f"   Python files: {py_files}")
        print(f"   Notebook files: {ipynb_files}")
        print(f"   All files: {files}")
    else:
        print("   ❌ Predictions folder not found")
    
    # Check runs folder (where your models are)
    print(f"\n📁 RUNS Folder (your models):")
    runs_path = './runs'
    if os.path.exists(runs_path):
        # Walk through all subdirectories to find model files
        model_files = []
        for root, dirs, files in os.walk(runs_path):
            for file in files:
                if file.endswith('.pt'):  # YOLO model files
                    model_files.append(os.path.join(root, file))
        
        print(f"   Model files found:")
        for model in model_files:
            print(f"   ✅ {model}")
        
        if not model_files:
            print("   ❌ No .pt model files found in runs/")
    else:
        print("   ❌ Runs folder not found")
    
    # Check what's in root that might be working code
    print(f"\n📁 ROOT Folder - Potential working files:")
    root_files = os.listdir('.')
    working_files = []
    
    for file in root_files:
        if file.endswith('.py') and any(keyword in file.lower() for keyword in 
                                      ['detect', 'predict', 'real', 'time', 'ppe', 'safety']):
            working_files.append(file)
        elif file.endswith('.ipynb') and any(keyword in file.lower() for keyword in 
                                           ['detect', 'predict', 'real', 'time', 'ppe']):
            working_files.append(file)
    
    print(f"   Potential working scripts: {working_files}")

if __name__ == "__main__":
    find_working_files()