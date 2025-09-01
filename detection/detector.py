"""
YOLO-based PPE Detection System for SafetyAI
Handles model inference and detection pipeline
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
from src.utils import draw_bounding_boxes, get_device

class SafetyDetector:
    """Main detection class for PPE compliance checking"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        """
        Initialize the Safety Detector
        
        Args:
            model_path: Path to trained YOLO model
            confidence_threshold: Minimum confidence for detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = get_device()
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Class names (update based on your dataset)
        self.class_names = [
            'hard-hat', 'safety-vest', 'gloves', 'safety-boots',
            'face-mask', 'goggles', 'ear-protection'
        ]
        
    def detect_single_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect PPE in a single image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(image, conf=self.confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(confidence),
                        'class': class_id,
                        'class_name': self.class_names[class_id]
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_from_file(self, image_path: str) -> List[Dict]:
        """Detect PPE from image file"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.detect_single_image(image)
    
    def detect_from_directory(self, directory: str) -> Dict[str, List[Dict]]:
        """Detect PPE in all images in directory"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(directory).glob(f'*{ext}'))
            image_files.extend(Path(directory).glob(f'*{ext.upper()}'))
        
        results = {}
        for image_file in image_files:
            detections = self.detect_from_file(str(image_file))
            results[str(image_file)] = detections
        
        return results
    
    def detect_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """Detect PPE in video file"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect PPE
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detect_single_image(rgb_frame)
            
            # Draw results
            result_frame = draw_bounding_boxes(frame, detections, self.class_names)
            
            if output_path:
                out.write(result_frame)
            else:
                cv2.imshow('Safety Detection', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def check_compliance(self, detections: List[Dict], required_ppe: List[str]) -> Dict[str, bool]:
        """
        Check PPE compliance based on detected items
        
        Args:
            detections: List of detected items
            required_ppe: List of required PPE items
            
        Returns:
            Dictionary with compliance status for each PPE type
        """
        detected_ppe = [det['class_name'] for det in detections]
        
        compliance = {}
        for ppe in required_ppe:
            compliance[ppe] = ppe in detected_ppe
        
        return compliance
    
    def batch_process(self, input_dir: str, output_dir: str) -> None:
        """Process multiple images and save results"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        results = self.detect_from_directory(input_dir)
        
        for image_path, detections in results.items():
            # Load and process image
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result_image = draw_bounding_boxes(image, detections, self.class_names)
            
            # Save result
            filename = Path(image_path).stem
            output_path = os.path.join(output_dir, f"{filename}_detected.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict[str, int]:
        """Get summary of detected PPE items"""
        summary = {class_name: 0 for class_name in self.class_names}
        
        for detection in detections:
            class_name = detection['class_name']
            summary[class_name] += 1
        
        return summary

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SafetyAI PPE Detection')
    parser.add_argument('--model', type=str, default='models/best_model.pt',
                       help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                       help='Image/video path or directory')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'batch'],
                       default='image', help='Detection mode')
    
    args = parser.parse_args()
    
    detector = SafetyDetector(args.model, args.conf)
    
    if args.mode == 'image':
        detections = detector.detect_from_file(args.source)
        print(f"Detected {len(detections)} items:")
        for det in detections:
            print(f"  {det['class_name']}: {det['confidence']:.2f}")
    
    elif args.mode == 'video':
        detector.detect_video(args.source, args.output)
    
    elif args.mode == 'batch':
        detector.batch_process(args.source, args.output)
        print(f"Processed images saved to {args.output}")

if __name__ == "__main__":
    main()
