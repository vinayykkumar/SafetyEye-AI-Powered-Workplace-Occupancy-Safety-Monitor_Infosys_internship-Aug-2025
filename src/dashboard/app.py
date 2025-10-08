import streamlit as st
import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import tempfile
import pandas as pd
import os

# --- CONFIGURATION ---
st.set_page_config(page_title="Construction Safety Dashboard", layout="wide")
st.title("🚧 Construction Safety Monitoring Dashboard")

# --- MODEL AND DATA PATHS ---
MODEL_PATH = Path("./runs/detect/ppe_final_tuned_model2/weights/best.pt")
LOG_FILE = Path("./outputs/violation_log.csv")

LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# --- INITIALIZE SESSION STATE ---
if 'violations_df' not in st.session_state:
    if LOG_FILE.exists():
        # Tell pandas to automatically convert the 'Timestamp' column to a datetime object
        st.session_state.violations_df = pd.read_csv(LOG_FILE, parse_dates=['Timestamp'])
    else:
        st.session_state.violations_df = pd.DataFrame(columns=['Timestamp', 'ViolationType', 'Confidence'])

# --- HELPER FUNCTIONS ---
def load_model(model_path):
    """Loads the YOLOv8 model from the specified path."""
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the YOLO model: {e}")
        return None

def save_log(timestamp, violation_type, confidence):
    """
    Appends a new violation log to the DataFrame in memory and saves the entire
    updated DataFrame back to the CSV file.
    """
    # Create a new DataFrame for the single new log entry
    new_log = pd.DataFrame(
        [[timestamp, violation_type, confidence]],
        columns=['Timestamp', 'ViolationType', 'Confidence']
    )

    # Concatenate the new log with the existing DataFrame stored in the session state
    st.session_state.violations_df = pd.concat(
        [st.session_state.violations_df, new_log],
        ignore_index=True
    )

    # Save the entire updated DataFrame to the CSV file
    st.session_state.violations_df.to_csv(LOG_FILE, index=False)


# Create this set outside your processing loop, perhaps in the main part of your app
# so it persists across frames for a single video run.
logged_violations_in_run = set()

def process_frame(frame, model, confidence_threshold, violation_classes):
    """Processes a single frame for violation detection and returns the annotated frame."""
    
    # 1. Use model.track() instead of model()
    # persist=True tells the tracker to remember objects between frames
    # Specify bytetrack.yaml as the tracker configuration
    results = model.track(frame, persist = True, tracker = "./src/dashboard/custom_tracker.yaml", verbose=False)
    
    # Check if there are any tracked objects
    if results[0].boxes.id is not None:
        # Get the boxes and track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        confs = results[0].boxes.conf.cpu().tolist()

        # Plot the tracks
        for box, track_id, cls, conf in zip(boxes, track_ids, clss, confs):
            if conf < confidence_threshold:
                continue

            class_name = model.names[cls]
            
            # Draw bounding box for all tracked objects
            x1, y1, x2, y2 = map(int, box)
            color = (0, 255, 0) # Green by default
            
            if class_name in violation_classes:
                color = (0, 0, 255) # Red for violation
                
                # 2. Check if this specific violation has already been logged for this person
                violation_key = (track_id, class_name)
                if violation_key not in logged_violations_in_run:
                    # Log the violation ONLY IF it's a new incident
                    save_log(pd.to_datetime('now'), class_name, conf)
                    logged_violations_in_run.add(violation_key) # Remember this incident
            
            # Add label with Class, ID, and Confidence
            label = f"ID:{track_id} {class_name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame

# --- UI SETUP ---
# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.025)
    
    VIOLATION_CLASSES = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}
    
    source_choice = st.radio("Select Video Source", ["Upload a Video File", "Live Webcam Feed"])

# --- MAIN APPLICATION LOGIC ---
model = load_model(MODEL_PATH)

if not model:
    st.stop()

# Create tabs for different sections
tab1, tab2 = st.tabs(["Live Monitoring", "Analytics Dashboard"])

# --- TAB 1: LIVE MONITORING ---
with tab1:
    st.header("Live Monitoring Feed")
    
    if source_choice == "Upload a Video File":
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = Path(tfile.name)
            
            st.video(str(video_path))
            if st.button("Start Processing", key="process_file"):
                # 1. Reset the main violations DataFrame to an empty one
                st.session_state.violations_df = pd.DataFrame(columns=['Timestamp', 'ViolationType', 'Confidence'])
                st.session_state.violations_df.to_csv(LOG_FILE, index=False)

                logged_violations_in_run.clear() 
                
                st.info("Video processing started...")

                video_stream = cv2.VideoCapture(str(video_path))
                frame_placeholder = st.empty()
                
                while video_stream.isOpened():
                    ret, frame = video_stream.read()
                    if not ret:
                        break
                    
                    annotated_frame = process_frame(frame, model, confidence_threshold, VIOLATION_CLASSES)
                    
                    # for v_type, v_conf in violations:
                    #     save_log(pd.to_datetime('now'), v_type, v_conf)
                    
                    frame_placeholder.image(annotated_frame, channels = "BGR", use_container_width = True, width = "stretch")
                
                video_stream.release()
                st.success("Video processing complete!")

    elif source_choice == "Live Webcam Feed":
        st.info("Starting webcam feed... Please grant camera permissions.")
        
        # A button to start/stop the webcam feed
        run = st.checkbox('Start Webcam', key='webcam_run')
        frame_placeholder = st.empty()
        
        cap = cv2.VideoCapture(0)
        
        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image from webcam.")
                break
            
            start_time = time.time()
            annotated_frame, violations = process_frame(frame, model, confidence_threshold, VIOLATION_CLASSES)
            end_time = time.time()
            
            # Calculate and display FPS
            fps = 1 / (end_time - start_time)
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if violations:
                cv2.putText(annotated_frame, "!!! VIOLATION !!!", (10, 70), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 0, 255), 2)
                for v_type, v_conf in violations:
                    save_log(pd.to_datetime('now'), v_type, v_conf)

            frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True, width="stretch")
            
        cap.release()

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
    st.header("Compliance Analytics")
    
    if st.session_state.violations_df.empty:
        st.info("No violations have been logged yet.")
    else:
        # Display summary statistics
        st.subheader("Violation Summary")
        violation_counts = st.session_state.violations_df['ViolationType'].value_counts()
        st.dataframe(violation_counts)
        
        # Display charts
        st.subheader("Violations by Type")
        st.bar_chart(violation_counts)
        
        st.subheader("Full Violation Log")
        # Display the full log, sorted by most recent
        st.dataframe(st.session_state.violations_df.sort_values(by="Timestamp", ascending=False))
        
        # Allow downloading the log file
        csv = st.session_state.violations_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Log as CSV",
            data=csv,
            file_name='violation_log.csv',
            mime='text/csv',
        )