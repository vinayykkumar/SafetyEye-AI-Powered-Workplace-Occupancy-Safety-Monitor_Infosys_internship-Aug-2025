from ultralytics import YOLO
from PIL import Image
import os

def main():
    # --- Configuration ---
    MODEL_PATH = r'runs\detect\yolov8n_ppe_gpu_run_2\weights\best.pt'
    IMAGE_PATH = 'test_images/test_image.jpg'  
    

    CONFIDENCE_THRESHOLD = 0.40 
    # --- End of Configuration ---

    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
        print("Error: Model or image file not found. Please check paths.")
        return

    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)

    # --- Rule Engine Logic ---
    hardhat_detected = False
    vest_detected = False
    class_names = model.names
    
    for r in results:
        for box in r.boxes:
      
            confidence = float(box.conf[0])
            
            # --- APPLY THE THRESHOLD ---
            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                
                if class_name == 'hardhat':
                    hardhat_detected = True
                elif class_name == 'vest':
                    vest_detected = True

    # --- Print Compliance Summary to Terminal ---
    print("\n--- Safety Compliance Check ---")
    if hardhat_detected:
        print(f"✅ Hardhat Status: [COMPLIANT] - Hardhat detected.")
    else:
        print(f"❌ Hardhat Status: [VIOLATION] - No hardhat detected (or confidence too low).")
        
    if vest_detected:
        print(f"✅ Vest Status:    [COMPLIANT] - Vest detected.")
    else:
        print(f"❌ Vest Status:    [VIOLATION] - No vest detected (or confidence too low).")
    
    print("---------------------------------\n")

    # --- Visualization ---
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save('prediction_result.jpg')
    print("Result image saved to 'prediction_result.jpg'")


if __name__ == '__main__':
    main()