import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import io
import base64
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Construction Site Object Detection - Image Annotator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown("""
<style>
    .class-legend {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .color-box {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border: 2px solid #333;
        border-radius: 3px;
    }
    .detection-stats {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_class_colors():
    """Define colors for different classes (BGR format for OpenCV)"""
    return {
        0: (0, 255, 0),      # Hardhat - Green
        1: (255, 255, 0),    # Mask - Yellow  
        2: (0, 0, 255),      # NO-Hardhat - Red
        3: (255, 0, 255),    # NO-Mask - Magenta
        4: (255, 165, 0),    # NO-Safety Vest - Orange
        5: (0, 255, 255),    # Person - Cyan
        6: (255, 255, 255),  # Safety Cone - White
        7: (0, 128, 255),    # Safety Vest - Blue
        8: (255, 140, 0),    # machinery - Dark Orange
        9: (255, 20, 147)    # vehicle - Deep Pink
    }

def get_class_colors_rgb():
    """Get class colors in RGB format for display"""
    bgr_colors = get_class_colors()
    return {k: (b, g, r) for k, (b, g, r) in bgr_colors.items()}

def print_debug_detections(detections, class_names, confidence_threshold):
    """Print raw detection data for debugging"""
    st.markdown("### 🐛 Debug Information")
    st.markdown(f"**Total raw detections found:** {len(detections)}")
    
    if len(detections) > 0:
        # Count unknown objects
        unknown_objects = []
        known_objects = []
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, conf, class_id = detection
            class_id = int(class_id)
            
            if class_id in class_names:
                class_name = class_names[class_id]
                known_objects.append((i+1, class_name, class_id, conf, [int(x1), int(y1), int(x2), int(y2)]))
            else:
                class_name = f"Unknown_{class_id}"
                unknown_objects.append((i+1, class_name, class_id, conf, [int(x1), int(y1), int(x2), int(y2)]))
        
        # Display known objects
        if known_objects:
            st.markdown("**Known Objects (in our class list):**")
            for i, class_name, class_id, conf, bbox in known_objects:
                status = "✅ Above threshold" if conf >= confidence_threshold else "❌ Below threshold"
                st.write(f"Detection {i}: {class_name} (ID: {class_id}) - Confidence: {conf:.3f} {status}")
                st.write(f"   Bbox: {bbox}")
        
        # Display unknown objects
        if unknown_objects:
            st.markdown("**🔍 Unknown Objects (NOT in our class list):**")
            st.warning(f"Found {len(unknown_objects)} objects with unknown class IDs!")
            for i, class_name, class_id, conf, bbox in unknown_objects:
                status = "✅ Above threshold" if conf >= confidence_threshold else "❌ Below threshold"
                st.error(f"Detection {i}: {class_name} (ID: {class_id}) - Confidence: {conf:.3f} {status}")
                st.write(f"   Bbox: {bbox}")
            
            st.info("💡 **Tip:** Unknown objects might indicate:")
            st.write("- The model was trained on different classes than expected")
            st.write("- Model is detecting classes beyond the 10 construction site classes")
            st.write("- There might be a mismatch between model and class definitions")
        else:
            st.success("✅ All detected objects have known class IDs")
    else:
        st.write("No detections found by YOLO model")
    
    st.markdown("---")

def draw_detections_on_image(image, detections, class_names, colors, selected_classes, confidence_threshold):
    """Draw bounding boxes and labels on image for selected classes"""
    annotated_image = image.copy()
    detection_stats = {}
    
    for detection in detections:
        x1, y1, x2, y2, conf, class_id = detection
        class_id = int(class_id)
        class_name = class_names.get(class_id, f"Unknown_{class_id}")
        
        # Only draw if confidence is above threshold and class is selected
        if conf >= confidence_threshold and class_id in selected_classes:
            color = colors.get(class_id, (255, 255, 255))
            
            # Update detection stats
            if class_name not in detection_stats:
                detection_stats[class_name] = 0
            detection_stats[class_name] += 1
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Create label with class name and confidence
            label = f"{class_name} {conf:.2f}"
            
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            cv2.rectangle(
                annotated_image,
                (int(x1), int(y1) - text_height - 10),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image, 
                label, 
                (int(x1), int(y1) - 5),
                font, 
                font_scale, 
                (0, 0, 0),  # Black text
                thickness
            )
    
    return annotated_image, detection_stats

def create_color_legend(class_names, colors_rgb):
    """Create a color legend for the classes"""
    st.markdown("### 🎨 Class Color Legend")
    
    # Create columns for better layout
    cols = st.columns(3)
    col_idx = 0
    
    for class_id, class_name in class_names.items():
        color = colors_rgb[class_id]
        color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        with cols[col_idx % 3]:
            st.markdown(
                f'<div style="display: flex; align-items: center; margin: 5px 0;">'
                f'<div style="width: 20px; height: 20px; background-color: {color_hex}; '
                f'border: 2px solid #333; border-radius: 3px; margin-right: 10px;"></div>'
                f'<span><strong>{class_name}</strong></span></div>',
                unsafe_allow_html=True
            )
        
        col_idx += 1

def convert_image_to_download_link(image, filename):
    """Convert image to downloadable format"""
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='PNG')
    img_bytes = img_buffer.getvalue()
    
    return img_bytes

def main():
    st.title("🎯 Construction Site Object Detection - Image Annotator")
    st.markdown("Upload an image to detect and visualize all construction site objects with bounding boxes")
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Model selection
    model_options = {
        "Custom Trained Model (Recommended)": "models/construction_best.pt",
        "YOLOv8n (Base Model)": "models/yolov8n.pt"
    }
    
    selected_model = st.sidebar.selectbox("Select YOLO Model", list(model_options.keys()), index=0)
    model_path = model_options[selected_model]
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please make sure you have the model files in the 'models' directory")
        st.stop()
    
    # Show model information
    if "Custom Trained" in selected_model:
        st.sidebar.success("🎯 Using custom trained model (100 epochs)")
        st.sidebar.info("This model was specifically trained on construction site safety data for optimal performance.")
    else:
        st.sidebar.info(f"Using {selected_model}")
    
    # Load model
    model = load_model(model_path)
    if model is None:
        st.stop()
    
    # Define class names and colors
    class_names = {
        0: "Hardhat", 1: "Mask", 2: "NO-Hardhat", 3: "NO-Mask", 
        4: "NO-Safety Vest", 5: "Person", 6: "Safety Cone", 
        7: "Safety Vest", 8: "machinery", 9: "vehicle"
    }
    
    colors = get_class_colors()
    colors_rgb = get_class_colors_rgb()
    
    # Detection confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Lower values detect more objects but may include false positives"
    )
    
    # Debug mode toggle
    debug_mode = st.sidebar.checkbox(
        "🐛 Debug Mode", 
        value=False,
        help="Show raw detection output and detailed debugging information"
    )
    
    st.sidebar.markdown("---")
    
    # Class selection
    st.sidebar.markdown("### 🏷️ Select Classes to Display")
    st.sidebar.markdown("Choose which object classes you want to see in the annotated image:")
    
    # Group classes by category for better UX
    safety_equipment = [0, 1, 7]  # Hardhat, Mask, Safety Vest
    violations = [2, 3, 4]        # NO-Hardhat, NO-Mask, NO-Safety Vest  
    people_objects = [5, 6]       # Person, Safety Cone
    vehicles_machinery = [8, 9]   # machinery, vehicle
    
    selected_classes = set()
    
    # Safety Equipment category
    st.sidebar.markdown("**🦺 Safety Equipment:**")
    for class_id in safety_equipment:
        if st.sidebar.checkbox(f"{class_names[class_id]}", value=True, key=f"class_{class_id}"):
            selected_classes.add(class_id)
    
    # Violations category  
    st.sidebar.markdown("**⚠️ Safety Violations:**")
    for class_id in violations:
        if st.sidebar.checkbox(f"{class_names[class_id]}", value=True, key=f"class_{class_id}"):
            selected_classes.add(class_id)
    
    # People & Objects category
    st.sidebar.markdown("**👥 People & Objects:**")
    for class_id in people_objects:
        if st.sidebar.checkbox(f"{class_names[class_id]}", value=True, key=f"class_{class_id}"):
            selected_classes.add(class_id)
    
    # Vehicles & Machinery category
    st.sidebar.markdown("**🚗 Vehicles & Machinery:**")
    for class_id in vehicles_machinery:
        if st.sidebar.checkbox(f"{class_names[class_id]}", value=True, key=f"class_{class_id}"):
            selected_classes.add(class_id)
    
    # Quick selection buttons
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🚀 Quick Select:**")
    col1, col2 = st.sidebar.columns(2)
    
    if col1.button("✅ Select All"):
        st.rerun()
    
    if col2.button("❌ Clear All"):
        st.rerun()
    
    # Image upload
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📁 Upload Image", 
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
    
    if uploaded_file is not None:
        # Read and display original image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to BGR for OpenCV processing
        if len(image_array.shape) == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        # Create two columns for before/after comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(image, use_container_width=True)
        
        # Run inference
        with st.spinner("🔍 Detecting objects..."):
            results = model(image_bgr)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            # Get detections
            detections = results[0].boxes.data.cpu().numpy()
            
            # Show debug information if enabled
            if debug_mode:
                print_debug_detections(detections, class_names, confidence_threshold)
                
                # Show selected classes info
                st.markdown("### 🎯 Class Selection Debug")
                st.write(f"**Selected classes:** {sorted(selected_classes)}")
                selected_names = [class_names[cid] for cid in sorted(selected_classes)]
                st.write(f"**Selected class names:** {selected_names}")
                st.markdown("---")
            
            # Draw annotations
            annotated_image, detection_stats = draw_detections_on_image(
                image_bgr, detections, class_names, colors, selected_classes, confidence_threshold
            )
            
            with col2:
                st.markdown("### 🎯 Annotated Image")
                # Convert BGR back to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, use_container_width=True)
            
            # Detection statistics
            st.markdown("---")
            st.markdown("### 📊 Detection Statistics")
            
            if detection_stats:
                # Display stats in columns
                cols = st.columns(min(len(detection_stats), 4))
                for idx, (class_name, count) in enumerate(detection_stats.items()):
                    with cols[idx % 4]:
                        st.metric(class_name, count)
                
                # Total detections
                total_detections = sum(detection_stats.values())
                st.success(f"✅ Total objects detected: **{total_detections}**")
                
                # Detailed breakdown
                with st.expander("📋 Detailed Detection Breakdown"):
                    for class_name, count in sorted(detection_stats.items()):
                        percentage = (count / total_detections) * 100
                        st.write(f"• **{class_name}**: {count} objects ({percentage:.1f}%)")
            else:
                st.info("ℹ️ No objects detected above the confidence threshold or no classes selected")
            
            # Download button
            st.markdown("---")
            st.markdown("### 💾 Download Annotated Image")
            
            if detection_stats:
                # Create download button
                img_bytes = convert_image_to_download_link(annotated_image, "annotated_image.png")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"construction_detection_{timestamp}.png"
                
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png",
                    help="Download the image with bounding boxes drawn around detected objects"
                )
            else:
                st.warning("⚠️ No objects detected to download. Try lowering the confidence threshold or selecting more classes.")
        
        else:
            # Debug info for no detections
            if debug_mode:
                st.markdown("### 🐛 Debug: No Detections Found")
                st.write(f"**YOLO Results Length:** {len(results)}")
                if len(results) > 0:
                    st.write(f"**Boxes Length:** {len(results[0].boxes) if results[0].boxes is not None else 'None'}")
                    st.write(f"**Results[0] Type:** {type(results[0])}")
                st.markdown("---")
            
            with col2:
                st.markdown("### 🎯 Annotated Image")
                st.info("ℹ️ No objects detected in the image")
            
            st.warning("⚠️ No objects were detected. Try:")
            st.markdown("""
            - Lowering the confidence threshold
            - Selecting more object classes  
            - Using a different image with construction site objects
            - Checking if the image contains the objects you're looking for
            """)
    
    # Information section
    st.markdown("---")
    create_color_legend(class_names, colors_rgb)
    
    st.markdown("---")
    st.markdown("### ℹ️ About This Tool")
    st.markdown("""
    **This image annotation tool helps you:**
    - 🎯 **Detect construction site objects** using state-of-the-art YOLO models
    - 🎨 **Visualize detections** with colored bounding boxes and labels
    - 🔧 **Customize detection** by selecting specific object classes
    - 📊 **Analyze results** with detailed statistics
    - 💾 **Save annotated images** for documentation or training
    
    **Supported Object Classes:**
    - **Safety Equipment**: Hard hats, masks, safety vests
    - **Safety Violations**: Workers without required PPE
    - **People & Objects**: Workers, safety cones
    - **Vehicles & Equipment**: Construction machinery, vehicles
    
    **Tips for Best Results:**
    - Use clear, well-lit images
    - Ensure objects are reasonably sized in the image
    - Experiment with confidence threshold settings
    - Try both YOLO models to see which works better for your images
    """)

if __name__ == "__main__":
    main()