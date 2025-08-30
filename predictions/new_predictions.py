# new_predictions.py (v7 with Fallback Logic)

from ultralytics import YOLO
from PIL import Image
import os

# ...(calculate_iou function remains the same)...
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

def main():
    # --- Configuration ---
    MODEL_PATH = r'runs\detect\yolov8n_ppe_gpu_run_12\weights\best.pt'
    IMAGE_PATH = r'test_images\test_image2.jpeg'
    CONFIDENCE_THRESHOLD = 0.65
    IOU_THRESHOLD_HELMET = 0.1
    IOU_THRESHOLD_VEST = 0.3
    # --- End of Configuration ---

    if not os.path.exists(MODEL_PATH) or not os.path.exists(IMAGE_PATH):
        print("Error: Model or image file not found.")
        return

    model = YOLO(MODEL_PATH)
    results = model(IMAGE_PATH)
    class_names = model.names

    # --- Rule Engine Logic ---
    persons = []
    hardhats = []
    vests = []

    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                class_name = class_names[int(box.cls[0])]
                bbox = box.xyxy[0].cpu().numpy()
                if class_name == 'person':
                    persons.append(bbox)
                elif class_name == 'hardhat':
                    hardhats.append(bbox)
                elif class_name == 'vest':
                    vests.append(bbox)
    
    # --- [NEW] Advanced Compliance Summary ---
    print("\n--- Safety Compliance Check ---")

    if persons: # If we detected persons, use the advanced per-person logic
        print("Per-person analysis enabled:")
        for i, person_box in enumerate(persons):
            person_id = i + 1
            has_hardhat = False
            has_vest = False

            for hardhat_box in hardhats:
                if calculate_iou(person_box, hardhat_box) > IOU_THRESHOLD_HELMET:
                    has_hardhat = True
                    break
            for vest_box in vests:
                if calculate_iou(person_box, vest_box) > IOU_THRESHOLD_VEST:
                    has_vest = True
                    break
            
            print(f"\nPerson #{person_id}:")
            if has_hardhat: print("✅ Hardhat: [COMPLIANT]")
            else: print("❌ Hardhat: [VIOLATION] - MISSING HARDHAT")
            if has_vest: print("✅ Vest:    [COMPLIANT]")
            else: print("❌ Vest:    [VIOLATION] - MISSING VEST")

    elif hardhats or vests: # If NO persons, but PPE is detected, fallback to simple logic
        print("⚠️ WARNING: Person detection failed. Falling back to image-level check.")
        if hardhats:
            print(f"✅ Hardhat Status: [COMPLIANT] - {len(hardhats)} hardhat(s) detected.")
        else:
            print(f"❌ Hardhat Status: [VIOLATION] - No hardhats detected.")
        if vests:
            print(f"✅ Vest Status:    [COMPLIANT] - {len(vests)} vest(s) detected.")
        else:
            print(f"❌ Vest Status:    [VIOLATION] - No vests detected.")
    else: # If nothing is detected at all
        print("No persons or PPE detected meeting the confidence threshold.")

    print("\n------------------------------------------\n")

    # --- Visualization ---
    for r in results:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
        im.save('prediction_result.jpg')
    print("Result image saved to 'prediction_result.jpg'")

if __name__ == '__main__':
    main()