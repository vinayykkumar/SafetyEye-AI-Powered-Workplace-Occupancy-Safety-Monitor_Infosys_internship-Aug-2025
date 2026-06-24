# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import os
# from datetime import datetime, timedelta
# import numpy as np

# MODEL_PATH = r"C:\SafetyEye\models\best.pt"
# model = YOLO(MODEL_PATH)
# VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
# OUTPUT_DIR = "outputs/violation_screenshots"

# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# COOLDOWN_SECONDS = 5
# last_screenshot_time = datetime.now()
# last_violation_alert_time = datetime.now()
# VIOLATION_ALERT_COOLDOWN_SECONDS = 3


# st.set_page_config(page_title="SafetyEye Dashboard", layout="wide")
# st.title("AI-Powered Workplace Safety Monitor ")
# live_feed_placeholder = st.empty()
# message_placeholder = st.empty()
# st.sidebar.title("Alerts & Screenshots")
# alert_placeholder = st.sidebar.empty()
# screenshot_gallery = st.sidebar.empty()


# def process_and_display_stream():
#     """
#     Handles video stream processing and updates the Streamlit UI.
#     """
#     global last_screenshot_time, last_violation_alert_time
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error(" Could not open webcam!")
#         return

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # --- 1. Real-time inference and custom drawing ---
#         results = model(frame, conf=0.5, verbose=False)
#         detections = results[0].boxes.cpu().numpy()

#         violation_detected = False
#         active_violations = [] # NEW: List to hold specific violation names
#         annotated_frame = frame.copy()

#         # Custom drawing logic
#         for box in detections:
#             x1, y1, x2, y2 = box.xyxy[0].astype(int)
#             cls = int(box.cls[0])
#             label = model.names[cls]
            
#             box_color = (0, 255, 0)
            
#             if label in VIOLATION_CLASSES:
#                 box_color = (0, 0, 255)
#                 violation_detected = True
#                 active_violations.append(label) # NEW: Add violation name to list
            
#             cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
#             cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#         # --- 2. Violation Detection & Alerts ---
#         current_time = datetime.now()
        
#         if violation_detected:
#             # Display alert message with a cooldown in the main section
#             if (current_time - last_violation_alert_time).total_seconds() > VIOLATION_ALERT_COOLDOWN_SECONDS:
#                 # NEW: Join all active violation names for a single, clear message
#                 violation_msg = ", ".join(set(active_violations)) 
#                 message_placeholder.error(f" **VIOLATION DETECTED:** {violation_msg}")
#                 last_violation_alert_time = current_time
            
#             # Save screenshot with a cooldown (existing logic)
#             if (current_time - last_screenshot_time).total_seconds() > COOLDOWN_SECONDS:
#                 timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#                 filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp}.jpg")
#                 cv2.imwrite(filename, annotated_frame)
#                 st.sidebar.success(f" Screenshot saved: {filename}")
#                 last_screenshot_time = current_time
#         else:
#             message_placeholder.empty()
#             # Reset alert time so it shows up immediately on next violation
#             last_violation_alert_time = current_time - timedelta(seconds=VIOLATION_ALERT_COOLDOWN_SECONDS)
        
#         # Display the live annotated frame on the dashboard
#         live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

#         # --- 3. Display screenshots in gallery ---
#         screenshot_files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
#         recent_screenshots = screenshot_files[:5]
        
#         with screenshot_gallery.container():
#             st.markdown("### Recent Violations")
#             for filename in recent_screenshots:
#                 st.image(os.path.join(OUTPUT_DIR, filename), caption=filename, use_container_width=True)


#     cap.release()

# # Run the app
# if __name__ == "__main__":
#     process_and_display_stream()





# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import os
# from datetime import datetime, timedelta
# import numpy as np
# from PIL import Image
# import io
# import csv # Import the CSV module
# import pandas as pd


# # ----------------------------------------
# # Global Variables and Model Loading
# # ----------------------------------------

# # Load the trained YOLOv8 model
# MODEL_PATH = r"C:\SafetyEye\models\best.pt"
# model = YOLO(MODEL_PATH)

# # Define violation classes and screenshot output directory
# VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']
# OUTPUT_DIR = "outputs/violation_screenshots"

# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# # Define the log file path
# LOG_FILE = "outputs/violation_log.csv"

# # Check if log file exists and write header if it doesn't
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Timestamp', 'ViolationType', 'ScreenshotPath'])

# # Cooldown variables
# COOLDOWN_SECONDS = 40 # Cooldown for saving screenshots
# last_screenshot_time = datetime.now()
# last_violation_alert_time = datetime.now()
# VIOLATION_ALERT_COOLDOWN_SECONDS = 3 # Cooldown for displaying the message

# # ----------------------------------------
# # Streamlit Dashboard UI
# # ----------------------------------------

# st.set_page_config(page_title="SafetyEye Dashboard", layout="wide")
# st.title("AI-Powered Workplace Safety Monitor ")

# # Placeholders for dynamic content
# live_feed_placeholder = st.empty()
# message_placeholder = st.empty()
# st.sidebar.title("Alerts & Screenshots")
# alert_placeholder = st.sidebar.empty()
# screenshot_gallery = st.sidebar.empty()

# # --- INPUT SELECTION IN SIDEBAR (MODIFIED DEFAULT) ---
# input_mode = st.sidebar.radio(
#     "Select Input Source:",
#     ('Photo', 'Webcam', 'Video'), # 'Photo' is now the default selection
#     index=0 # Explicitly set index to 0 for 'Photo'
# )

# uploaded_file = None
# if input_mode == 'Photo':
#     uploaded_file = st.sidebar.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
# elif input_mode == 'Video':
#     uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

# # ----------------------------------------
# # CORE DETECTION LOGIC FUNCTIONS (Same as before)
# # ----------------------------------------

# def display_recent_logs():
#     """Reads the CSV log and displays a table of recent violations."""
#     if os.path.exists(LOG_FILE):
#         try:
#             df = pd.read_csv(LOG_FILE)
#             st.markdown("---")
#             st.markdown("## Violation Log")
            
#             # Display the log, showing the most recent 10 entries
#             st.dataframe(df.tail(10).style.set_properties(**{'font-size': '10pt'}), use_container_width=True)
            
#         except pd.errors.EmptyDataError:
#             st.info("Log is empty. Run the monitor to detect and log violations.")
#         except Exception as e:
#             st.error(f"Error reading log file: {e}")

# def get_annotated_frame(frame):
#     """Performs inference and custom drawing on a single frame."""
#     results = model(frame, conf=0.5, verbose=False)
#     detections = results[0].boxes.cpu().numpy()

#     violation_detected = False
#     active_violations = [] 
#     annotated_frame = frame.copy()

#     for box in detections:
#         x1, y1, x2, y2 = box.xyxy[0].astype(int)
#         cls = int(box.cls[0])
#         label = model.names[cls]
        
#         box_color = (0, 255, 0) # Green
        
#         if label in VIOLATION_CLASSES:
#             box_color = (0, 0, 255) # Red
#             violation_detected = True
#             active_violations.append(label)
        
#         # Draw the box and label
#         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
#         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
#     return annotated_frame, violation_detected, active_violations

# def save_and_display_alert(annotated_frame, violation_detected, active_violations):
#     """Handles alert messages and screenshot saving with cooldowns."""
#     global last_screenshot_time, last_violation_alert_time
#     current_time = datetime.now()
    
#     if violation_detected:
#         # --- Display alert message with a cooldown in the main section ---
#         if (current_time - last_violation_alert_time).total_seconds() > VIOLATION_ALERT_COOLDOWN_SECONDS:
#             violation_msg = ", ".join(set(active_violations)) 
#             message_placeholder.error(f" **VIOLATION DETECTED:** {violation_msg}")
#             last_violation_alert_time = current_time
        
#         # --- Save screenshot with a cooldown ---
#         if (current_time - last_screenshot_time).total_seconds() > COOLDOWN_SECONDS:
#             timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#             filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp_str}.jpg")
#             cv2.imwrite(filename, annotated_frame)
            
#             # --- NEW: Logging the event ---
#             violation_types_str = ", ".join(set(active_violations))
#             log_entry = [timestamp_str, violation_types_str, filename]
            
#             with open(LOG_FILE, mode='a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(log_entry)
#             # --- END NEW: Logging the event ---
            
#             st.sidebar.success(f" Screenshot saved: {filename}")
#             last_screenshot_time = current_time
#     else:
#         message_placeholder.empty()
#         # Reset alert time so it shows up immediately on next violation
#         last_violation_alert_time = current_time - timedelta(seconds=VIOLATION_ALERT_COOLDOWN_SECONDS)

# def update_screenshot_gallery():
#     """Updates the sidebar gallery with recent screenshots."""
#     screenshot_files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
#     recent_screenshots = screenshot_files[:5]
    
#     with screenshot_gallery.container():
#         st.markdown("### Recent Violations")
#         for filename in recent_screenshots:
#             st.image(os.path.join(OUTPUT_DIR, filename), caption=filename, use_container_width=True)

# # ----------------------------------------
# # MAIN EXECUTION LOGIC
# # ----------------------------------------

# if __name__ == "__main__":
    
#     # Condition 1: User selects Webcam
#     if input_mode == 'Webcam':
#         alert_placeholder.info("Webcam active. Detecting live...")
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error(" Could not open webcam!")
#         else:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Processing and display logic for webcam
#                 annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
#                 save_and_display_alert(annotated_frame, violation_detected, active_violations)
#                 live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
#                 update_screenshot_gallery()

#             cap.release()

#     # Condition 2: User uploads a Photo or Video
#     elif uploaded_file is not None:
#         alert_placeholder.info(f"Processing {input_mode}...")
        
#         # --- Process Photo (MODIFIED LOGIC) ---
#         if input_mode == 'Photo':
#             image_bytes = uploaded_file.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             frame = np.array(image)
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
            
#             # For static photo, check and save alert immediately
#             if violation_detected:
#                  # Capture specific violation message for the sidebar alert
#                  violation_msg = ", ".join(set(active_violations))
#                  st.sidebar.error(f"Violation detected in photo: {violation_msg}") 
#                  save_and_display_alert(annotated_frame, violation_detected, active_violations)
#             else:
#                  st.sidebar.success("No violations in uploaded photo.")

#             live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
#             update_screenshot_gallery()

#         # --- Process Video ---
#         elif input_mode == 'Video':
#             # Save uploaded video to a temporary file for OpenCV to read
#             tfile = uploaded_file.name
#             with open(tfile, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             cap = cv2.VideoCapture(tfile)
#             if not cap.isOpened():
#                 st.error(" Could not open uploaded video!")
#             else:
#                 video_status = st.empty()
#                 frame_count = 0
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
                    
#                     annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
#                     save_and_display_alert(annotated_frame, violation_detected, active_violations)
#                     live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
#                     update_screenshot_gallery()
#                     frame_count += 1
#                     video_status.info(f"Processing frame {frame_count}...")
                
#                 video_status.success("Video processing complete!")
#                 cap.release()
#                 os.remove(tfile) # Clean up the temporary file

#     # Condition 3: No input selected (Initial Load State)
#     else:
#         # Display instruction message only if no mode is selected yet
#         alert_placeholder.info(f"Select a source in the sidebar.")
#         live_feed_placeholder.markdown("""
#             <div style='text-align: center; padding: 50px; border: 2px dashed #4B88FF; border-radius: 10px; margin-top: 20px;'>
#                 <h3>Select an **Input Source** (Photo, Webcam, or Video) in the sidebar to start the Safety Monitor.</h3>
#                 <p>No video stream is active until a source is selected.</p>
#             </div>
#         """, unsafe_allow_html=True)
#         update_screenshot_gallery()
#     # Display the log and charts below the main video feed
#     display_recent_logs() 


# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import os
# from datetime import datetime, timedelta
# import numpy as np
# from PIL import Image
# import io
# import csv
# import pandas as pd
# import base64
# import plotly.express as px # REQUIRED FOR CHARTS

# # ----------------------------------------
# # 1. DIRECTORY AND LOG FILE SETUP
# # ----------------------------------------

# # Define and create the root 'outputs' directory first.
# OUTPUT_DIR_ROOT = "outputs"
# if not os.path.exists(OUTPUT_DIR_ROOT):
#     os.makedirs(OUTPUT_DIR_ROOT)

# # Define the log file path and check/create the log file.
# LOG_FILE = os.path.join(OUTPUT_DIR_ROOT, "violation_log.csv")
# if not os.path.exists(LOG_FILE):
#     with open(LOG_FILE, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Timestamp', 'ViolationType', 'ScreenshotPath'])

# # Define the sub-directory for screenshots and ensure it exists.
# OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "violation_screenshots")
# if not os.path.exists(OUTPUT_DIR):
#     os.makedirs(OUTPUT_DIR)

# # ----------------------------------------
# # Global Variables and Model Loading
# # ----------------------------------------

# MODEL_PATH = r"C:\SafetyEye\models\best.pt"
# model = YOLO(MODEL_PATH)
# VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# # Cooldown variables
# COOLDOWN_SECONDS = 40 
# last_screenshot_time = datetime.now()
# last_violation_alert_time = datetime.now()
# VIOLATION_ALERT_COOLDOWN_SECONDS = 3 

# # ----------------------------------------
# # Streamlit Dashboard UI
# # ----------------------------------------

# st.set_page_config(page_title="SafetyEye Dashboard", layout="wide")
# st.title("AI-Powered Workplace Safety Monitor ")

# live_feed_placeholder = st.empty()
# message_placeholder = st.empty()
# st.sidebar.title("Alerts & Screenshots")
# alert_placeholder = st.sidebar.empty()
# screenshot_gallery = st.sidebar.empty()

# # --- INPUT SELECTION IN SIDEBAR ---
# input_mode = st.sidebar.radio(
#     "Select Input Source:",
#     ('Photo', 'Webcam', 'Video'),
#     index=0 
# )

# uploaded_file = None
# if input_mode == 'Photo':
#     uploaded_file = st.sidebar.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
# elif input_mode == 'Video':
#     uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

# # ----------------------------------------
# # CORE DETECTION AND LOGGING FUNCTIONS
# # ----------------------------------------

# # --- Data Loading Function (Crucial for all inputs) ---
# @st.cache_data(ttl=1) # Force Streamlit to re-read the CSV log every 1 second
# def load_log_data():
#     """Loads log data from CSV, designed to bypass Streamlit's default caching."""
#     if os.path.exists(LOG_FILE):
#         try:
#             df = pd.read_csv(LOG_FILE)
#             return df
#         except pd.errors.EmptyDataError:
#             return pd.DataFrame()
#         except Exception as e:
#             st.error(f"Error loading log data: {e}")
#             return pd.DataFrame()
#     return pd.DataFrame()


# def display_compliance_stats():
#     """Reads the log file and displays KPI metrics and a bar chart."""
#     df = load_log_data()
    
#     st.markdown("---")
#     st.markdown("## Compliance Statistics")

#     # --- KPI METRICS ---
#     total_violations = len(df)
    
#     # Calculate most frequent violation for KPI
#     violation_counts = df['ViolationType'].str.split(', ').explode().str.strip().value_counts()
    
#     col1, col2, col3 = st.columns(3)
#     col1.metric("Total Violations Recorded", total_violations)
    
#     if not violation_counts.empty:
#         most_common_violation = violation_counts.index[0]
#         most_common_count = violation_counts.iloc[0]
#         col2.metric("Most Frequent Violation", most_common_violation, f"Count: {most_common_count}")
#     else:
#         col2.metric("Most Frequent Violation", "N/A")
        
#     col3.metric("Current Compliance Rate", "N/A (Needs Occupancy Data)")

#     st.markdown("---")
    
#     # --- VIOLATION FREQUENCY CHART ---
#     st.markdown("## Incident Trends by Type")
#     if not violation_counts.empty:
#         # Convert series to DataFrame for Plotly chart
#         chart_df = violation_counts.reset_index()
#         chart_df.columns = ['Violation Type', 'Count']
        
#         fig = px.bar(chart_df, x='Violation Type', y='Count',
#                      title='Total Incidents by PPE Type',
#                      color='Violation Type',
#                      color_discrete_sequence=px.colors.qualitative.Bold)
        
#         st.plotly_chart(fig, use_container_width=True)
#     else:
#         st.info("Log violations to view incident trends.")


# def display_recent_logs():
#     """Reads the CSV log, displays a table of recent violations, and provides a download option."""
#     df = load_log_data()

#     st.markdown("---")
#     st.markdown("## Violation Log")
    
#     if not df.empty:
#         # Display the log, showing the most recent 10 entries
#         st.dataframe(df.tail(10).style.set_properties(**{'font-size': '10pt'}), use_container_width=True)
        
#         # --- ADD DOWNLOAD BUTTON ---
#         csv_data = df.to_csv(index=False)
#         b64 = base64.b64encode(csv_data.encode()).decode()
#         href = f'<a href="data:file/csv;base64,{b64}" download="violation_log.csv">Download Full Log (CSV)</a>'
#         st.markdown(href, unsafe_allow_html=True)
        
#     else:
#         st.info("Log is empty. Run the monitor to detect and log violations.")


# # ... (get_annotated_frame, save_and_display_alert, and update_screenshot_gallery functions remain the same)
# # The remaining functions are omitted for brevity but should be included in your full app.py file.
# # Note: The logging logic in save_and_display_alert() handles the actual writing of the log file.

# def get_annotated_frame(frame):
#     """Performs inference and custom drawing on a single frame."""
#     results = model(frame, conf=0.5, verbose=False)
#     detections = results[0].boxes.cpu().numpy()

#     violation_detected = False
#     active_violations = [] 
#     annotated_frame = frame.copy()

#     for box in detections:
#         x1, y1, x2, y2 = box.xyxy[0].astype(int)
#         cls = int(box.cls[0])
#         label = model.names[cls]
        
#         box_color = (0, 255, 0) # Green
        
#         if label in VIOLATION_CLASSES:
#             box_color = (0, 0, 255) # Red
#             violation_detected = True
#             active_violations.append(label)
        
#         # Draw the box and label
#         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
#         cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
#     return annotated_frame, violation_detected, active_violations

# def save_and_display_alert(annotated_frame, violation_detected, active_violations):
#     """Handles alert messages and screenshot saving with cooldowns."""
#     global last_screenshot_time, last_violation_alert_time
#     current_time = datetime.now()
    
#     if violation_detected:
#         # --- Display alert message with a cooldown in the main section ---
#         if (current_time - last_violation_alert_time).total_seconds() > VIOLATION_ALERT_COOLDOWN_SECONDS:
#             violation_msg = ", ".join(set(active_violations)) 
#             message_placeholder.error(f" **VIOLATION DETECTED:** {violation_msg}")
#             last_violation_alert_time = current_time
        
#         # --- Save screenshot and log event with cooldown ---
#         if (current_time - last_screenshot_time).total_seconds() > COOLDOWN_SECONDS:
#             timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
#             filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp_str}.jpg")
#             cv2.imwrite(filename, annotated_frame)
            
#             # --- Logging the event ---
#             violation_types_str = ", ".join(set(active_violations))
#             log_entry = [timestamp_str, violation_types_str, filename]
            
#             with open(LOG_FILE, mode='a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(log_entry)
#             # --- End Logging ---
            
#             st.sidebar.success(f" Screenshot saved: {filename}")
#             last_screenshot_time = current_time
#     else:
#         message_placeholder.empty()
#         last_violation_alert_time = current_time - timedelta(seconds=VIOLATION_ALERT_COOLDOWN_SECONDS)

# def update_screenshot_gallery():
#     """Updates the sidebar gallery with recent screenshots."""
#     screenshot_files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
#     recent_screenshots = screenshot_files[:5]
    
#     with screenshot_gallery.container():
#         st.markdown("### Recent Violations")
#         for filename in recent_screenshots:
#             st.image(os.path.join(OUTPUT_DIR, filename), caption=filename, use_container_width=True)


# # ----------------------------------------
# # MAIN EXECUTION LOGIC
# # ----------------------------------------

# if __name__ == "__main__":
    
#     # Condition 1: User selects Webcam
#     if input_mode == 'Webcam':
#         alert_placeholder.info("Webcam active. Detecting live...")
#         cap = cv2.VideoCapture(0)
#         if not cap.isOpened():
#             st.error(" Could not open webcam!")
#         else:
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
                
#                 # Processing and display logic for webcam
#                 annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
#                 save_and_display_alert(annotated_frame, violation_detected, active_violations)
#                 live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
#                 update_screenshot_gallery()
                
#                 # Update stats and logs on every loop iteration
#                 display_compliance_stats()
#                 display_recent_logs() 
                
#             cap.release()

#     # Condition 2: User uploads a Photo or Video
#     elif uploaded_file is not None:
#         alert_placeholder.info(f"Processing {input_mode}...")
        
#         # --- Process Photo ---
#         if input_mode == 'Photo':
#             image_bytes = uploaded_file.read()
#             image = Image.open(io.BytesIO(image_bytes))
#             frame = np.array(image)
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)

            
#             live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
#             # For static photo, check and save alert immediately
#             if violation_detected:
#                  violation_msg = ", ".join(set(active_violations))
#                  st.sidebar.error(f"Violation detected in photo: {violation_msg}") 
#                  save_and_display_alert(annotated_frame, violation_detected, active_violations)
#                  #st.rerun()
#             else:
#                  st.sidebar.success("No violations in uploaded photo.")

#             update_screenshot_gallery()

#         # --- Process Video ---
#         elif input_mode == 'Video':
#             tfile = uploaded_file.name
#             with open(tfile, "wb") as f:
#                 f.write(uploaded_file.getbuffer())
            
#             cap = cv2.VideoCapture(tfile)
#             if not cap.isOpened():
#                 st.error(" Could not open uploaded video!")
#             else:
#                 video_status = st.empty()
#                 frame_count = 0
                
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
                    
#                     annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
#                     save_and_display_alert(annotated_frame, violation_detected, active_violations)
#                     live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
#                     update_screenshot_gallery()
#                     frame_count += 1
#                     video_status.info(f"Processing frame {frame_count}...")
                
#                 video_status.success("Video processing complete!")
#                 cap.release()
#                 os.remove(tfile) 
    
#     # Final display of logs and stats after processing (or on initial load)
#     else:
#         st.sidebar.info(f"Select a source in the sidebar.")
#         live_feed_placeholder.markdown("""
#             <div style='text-align: center; padding: 50px; border: 2px dashed #4B88FF; border-radius: 10px; margin-top: 20px;'>
#                 <h3> Select an **Input Source** (Photo, Webcam, or Video) in the sidebar to start the Safety Monitor.</h3>
#                 <p>No video stream is active until a source is selected.</p>
#             </div>
#         """, unsafe_allow_html=True)
#         update_screenshot_gallery()

#     st.markdown("---")
#     display_compliance_stats()
#     display_recent_logs()


import streamlit as st
from ultralytics import YOLO
import cv2
import os
from datetime import datetime, timedelta
import numpy as np
from PIL import Image
import io
import csv
import pandas as pd
import base64
import plotly.express as px # REQUIRED FOR CHARTS
import smtplib # Module for sending emails
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ----------------------------------------
# 1. DIRECTORY AND LOG FILE SETUP
# ----------------------------------------

# Define and create the root 'outputs' directory first.
OUTPUT_DIR_ROOT = "outputs"
if not os.path.exists(OUTPUT_DIR_ROOT):
    os.makedirs(OUTPUT_DIR_ROOT)

# Define the log file path and check/create the log file.
LOG_FILE = os.path.join(OUTPUT_DIR_ROOT, "violation_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'ViolationType', 'ScreenshotPath'])

# Define the sub-directory for screenshots and ensure it exists.
OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "violation_screenshots")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ----------------------------------------
# Global Variables and Model Loading
# ----------------------------------------

MODEL_PATH = r"C:\SafetyEye\models\best.pt"
model = YOLO(MODEL_PATH)
VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

# Cooldown variables
COOLDOWN_SECONDS = 40 
last_screenshot_time = datetime.now()
last_violation_alert_time = datetime.now()
VIOLATION_ALERT_COOLDOWN_SECONDS = 3 

# --- Email Configuration ---
SENDER_EMAIL = "quickqart30@gmail.com"  # <<< REPLACE with your Gmail address
SENDER_PASSWORD = "...."  # <<< REPLACE with your App Password
EMAIL_COOLDOWN_SECONDS = 300 # Send email only once every 5 minutes (300 seconds)
last_email_time = datetime.now() - timedelta(seconds=EMAIL_COOLDOWN_SECONDS) 
SUMMARY_EMAIL_COOLDOWN_SECONDS = 86400 
last_email_time = datetime.now() - timedelta(seconds=SUMMARY_EMAIL_COOLDOWN_SECONDS) 

# --- Session State Initialization ---
if 'log_updated_flag' not in st.session_state:
    st.session_state.log_updated_flag = False

# ----------------------------------------
# Streamlit Dashboard UI & Input Setup
# ----------------------------------------

st.set_page_config(page_title="SafetyEye Dashboard", layout="wide")
st.title("AI-Powered Workplace Safety Monitor 👷")

live_feed_placeholder = st.empty()
message_placeholder = st.empty()
st.sidebar.title("Alerts & Screenshots")
alert_placeholder = st.sidebar.empty()
screenshot_gallery = st.sidebar.empty()

# --- INPUT SELECTION IN SIDEBAR ---
input_mode = st.sidebar.radio(
    "Select Input Source:",
    ('Photo', 'Webcam', 'Video'),
    index=0 
)

uploaded_file = None
if input_mode == 'Photo':
    uploaded_file = st.sidebar.file_uploader("Upload Photo", type=['jpg', 'jpeg', 'png'])
elif input_mode == 'Video':
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

# --- EMAIL CONFIGURATION IN SIDEBAR ---
st.sidebar.markdown("---")
st.sidebar.title("Email Notification")

enable_email = st.sidebar.checkbox(
    "Enable email notifications", 
    value=False # Set to False by default
)

recipient_email = st.sidebar.text_input(
    "Recipient Email", 
    value="admin_or_safety_officer@example.com",
    disabled=not enable_email
)

notification_type = st.sidebar.radio(
    "Select notification method:",
    ('Real-time Violation Alerts', 'Summary Report (CSV)'),
    disabled=not enable_email
)

if enable_email and notification_type == 'Real-time Violation Alerts':
    st.sidebar.warning("""
        HIGH EMAIL VOLUME WARNING: Email is sent once per 5-minute cooldown period for critical violations.
    """)

# ----------------------------------------
# CORE DETECTION AND LOGGING FUNCTIONS
# ----------------------------------------

# def send_email_alert(violation_types, frame_time_str, screenshot_path, recipient, email_enabled, alert_type):
#     """Sends a high-priority email alert only if enabled and in Real-time mode."""
#     global last_email_time
#     current_time = datetime.now()

#     cooldown_duration = EMAIL_COOLDOWN_SECONDS if alert_type == 'Real-time Violation Alerts' else SUMMARY_EMAIL_COOLDOWN_SECONDS
    
#     # if not email_enabled or alert_type != 'Real-time Violation Alerts':
#     #     return False
#     if not email_enabled:
#         return False

#     if (current_time - last_email_time).total_seconds() >= EMAIL_COOLDOWN_SECONDS:
        
#         msg = EmailMessage()
#         violation_list = ', '.join(set(violation_types))
        
#         msg['Subject'] = f"🚨 URGENT: HIGH-PRIORITY VIOLATION DETECTED ({violation_list})!"
#         msg['From'] = SENDER_EMAIL
#         msg['To'] = recipient
        
#         body = f"""
#         Dear Safety Officer,
#         A *HIGH-PRIORITY SAFETY VIOLATION* was instantly detected by the SafetyEye monitoring system.
#         --- ALERT DETAILS ---
#         1. *Violation Type(s):* {violation_list}
#         2. *Detection Time:* {frame_time_str}
#         3. *Image Frame:* Evidence saved to: {screenshot_path}
#         This email is sent once every {EMAIL_COOLDOWN_SECONDS / 60} minutes for critical alerts.
#         """
#         msg.set_content(body)

#         try:
#             with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
#                 smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
#                 smtp.send_message(msg)
            
#             last_email_time = current_time
#             st.sidebar.warning("📧 Email Alert Sent! (Next in 5 min)")
#             return True
        
#         except Exception as e:
#             st.sidebar.error(f"❌ Email Failed: {e}")
#             return False
#     return False

def send_email_alert(violation_types, frame_time_str, screenshot_path, recipient, enable_email, alert_type):
    """Sends an email alert based on the selected type (Real-time or Summary)."""
    global last_email_time
    current_time = datetime.now()

    # Determine which cooldown to use
    cooldown_duration = EMAIL_COOLDOWN_SECONDS if alert_type == 'Real-time Violation Alerts' else SUMMARY_EMAIL_COOLDOWN_SECONDS
    
    if not enable_email:
        return False
    
    if (current_time - last_email_time).total_seconds() < cooldown_duration:
        return False 
    
    # --- Proceed to Send ---
    
    # We MUST use MIMEMultipart if we are attaching a file (Summary Report)
    msg = EmailMessage() 
    
    violation_list = ', '.join(set(violation_types))
    
    # 2. Configure Content and Attachment based on type
    if alert_type == 'Real-time Violation Alerts':
        subject = f"URGENT: HIGH-PRIORITY VIOLATION DETECTED ({violation_list})!"
        body = f"""
        Dear Safety Officer,
        A **HIGH-PRIORITY SAFETY VIOLATION** was instantly detected by the SafetyEye monitoring system.
        --- ALERT DETAILS ---
        1. **Violation Type(s):** {violation_list}
        2. **Detection Time:** {frame_time_str}
        3. **Image Frame:** Evidence saved to: {screenshot_path}
        This email is sent once every {cooldown_duration / 60} minutes for critical alerts.
        """
        # For Real-time, use simple set_content since there is NO attachment here
        msg.set_content(body) 
        
    elif alert_type == 'Summary Report (CSV)':
        subject = f"DAILY SAFETY REPORT: Violation Summary ({frame_time_str})"
        body = "Please find the comprehensive CSV log of all recorded safety violations attached below."
        
        # FIX 1: Set the body content *before* attaching the file
        msg.set_content(body) 

        # --- CSV ATTACHMENT LOGIC ---
        if os.path.exists(LOG_FILE):
            try:
                with open(LOG_FILE, 'rb') as f:
                    file_data = f.read()
                    file_name = os.path.basename(LOG_FILE)
                
                # FIX 2: Add the attachment
                msg.add_attachment(file_data, 
                                   maintype='application', 
                                   subtype='octet-stream', 
                                   filename=file_name)
            except Exception as e:
                st.sidebar.error(f" Failed to attach log file: {e}")
                return False 
        else:
            st.sidebar.error(" Summary Report Failed: Log file not found.")
            return False
            
    # 3. Finalize and Send Email
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.send_message(msg)
        
        last_email_time = current_time
        st.sidebar.warning(f"📧 {alert_type} Sent! (Next in {cooldown_duration / 60} min)")
        return True
    
    except Exception as e:
        st.sidebar.error(f" Email Failed: {e}")
        return False
    return False

@st.cache_data(ttl=1)
def load_log_data():
    """Loads log data from CSV, designed to bypass Streamlit's default caching."""
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Error loading log data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def display_compliance_stats():
    """Reads the log file and displays KPI metrics and a bar chart."""
    df = load_log_data()
    
    st.markdown("---")
    st.markdown("## Compliance Statistics")

    total_violations = len(df)
    violation_counts = df['ViolationType'].str.split(', ').explode().str.strip().value_counts()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Violations Recorded", total_violations)
    
    if not violation_counts.empty:
        most_common_violation = violation_counts.index[0]
        most_common_count = violation_counts.iloc[0]
        col2.metric("Most Frequent Violation", most_common_violation, f"Count: {most_common_count}")
    else:
        col2.metric("Most Frequent Violation", "N/A")
        
    col3.metric("Current Compliance Rate", "N/A (Needs Occupancy Data)")

    st.markdown("---")
    
    st.markdown("## Incident Trends by Type")
    if not violation_counts.empty:
        chart_df = violation_counts.reset_index()
        chart_df.columns = ['Violation Type', 'Count']
        
        fig = px.bar(chart_df, x='Violation Type', y='Count',
                     title='Total Incidents by PPE Type',
                     color='Violation Type',
                     color_discrete_sequence=px.colors.qualitative.Bold)
        
        st.plotly_chart(fig, use_container_width=True,key="incident_trend_chart")
    else:
        st.info("Log violations to view incident trends.")


def send_summary_on_demand(recipient, is_enabled):
    """Generates the CSV summary and sends it via email instantly."""
    
    if not is_enabled:
        st.error("Please enable email notifications to use this feature.")
        return

    # Check if the log file exists and has data (optional, but good practice)
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) <= 100: # Check size > header size
        st.warning("Log is empty or too small to send a report.")
        return

    # This is essentially the summary logic, but we pass the log file directly
    violation_types = ["Safety Summary Report"] # A generic type for the subject line
    
    # Get current time for the email body timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Call the core send_email_alert function with SUMMAR Y_EMAIL_COOLDOWN_SECONDS set to 0
    # NOTE: We can modify the global cooldown logic for this special case, 
    # but the cleanest way is to ensure the check is bypassed.
    
    # To bypass the cooldown, we temporarily set last_email_time far in the past.
    global last_email_time
    original_last_email_time = last_email_time
    
    # Temporarily set the last email time to bypass the check immediately
    last_email_time = datetime.now() - timedelta(seconds=SUMMARY_EMAIL_COOLDOWN_SECONDS * 2) 

    # --- Force Email Sending ---
    send_email_alert(
        violation_types,
        current_time,
        LOG_FILE, # Pass the log file path as the 'screenshot' path
        recipient,
        is_enabled,
        'Summary Report (CSV)' # Pass the correct type
    )

    # Restore the original last_email_time to ensure the real-time cooldown is respected
    last_email_time = original_last_email_time 
    
    # Provide visual confirmation
    st.success(f"Report generation initiated. Check {recipient} shortly.")


def display_recent_logs():
    """Reads the CSV log, displays a table of recent violations, and provides a download option."""
    df = load_log_data()

    st.markdown("---")
    st.markdown("## Violation Log")
    
    if not df.empty:
        # st.dataframe(df.tail(10).style.set_properties({'font-size': '10pt'}), use_container_width=True)
        st.dataframe(df.tail(10), use_container_width=True)
        
        csv_data = df.to_csv(index=False)
        b64 = base64.b64encode(csv_data.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="violation_log.csv">📥 Download Full Log (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    else:
        st.info("Log is empty. Run the monitor to detect and log violations.")


def get_annotated_frame(frame):
    """Performs inference and custom drawing on a single frame."""
    results = model(frame, conf=0.5, verbose=False)
    detections = results[0].boxes.cpu().numpy()

    violation_detected = False
    active_violations = [] 
    annotated_frame = frame.copy()

    for box in detections:
        x1, y1, x2, y2 = box.xyxy[0].astype(int)
        cls = int(box.cls[0])
        label = model.names[cls]
        
        box_color = (0, 255, 0) # Green
        
        if label in VIOLATION_CLASSES:
            box_color = (0, 0, 255) # Red
            violation_detected = True
            active_violations.append(label)
        
        # Draw the box and label
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return annotated_frame, violation_detected, active_violations

def save_and_display_alert(annotated_frame, violation_detected, active_violations):
    """Handles alert messages and screenshot saving with cooldowns."""
    global last_screenshot_time, last_violation_alert_time
    current_time = datetime.now()
    
    if violation_detected:
        # --- Display alert message with a cooldown in the main section ---
        if (current_time - last_violation_alert_time).total_seconds() > VIOLATION_ALERT_COOLDOWN_SECONDS:
            violation_msg = ", ".join(set(active_violations)) 
            message_placeholder.error(f"*VIOLATION DETECTED:* {violation_msg}")
            last_violation_alert_time = current_time
        
        # --- Save screenshot and log event with cooldown ---
        if (current_time - last_screenshot_time).total_seconds() > COOLDOWN_SECONDS:
            timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp_str}.jpg")
            cv2.imwrite(filename, annotated_frame)
            
            # --- Logging the event ---
            violation_types_str = ", ".join(set(active_violations))
            log_entry = [timestamp_str, violation_types_str, filename]
            
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(log_entry)
            
            # --- Trigger Email Alert (After screenshot is guaranteed) ---
            send_email_alert(
                active_violations, 
                timestamp_str, 
                filename,
                recipient_email,    # Pass the user input
                enable_email,       # Pass the checkbox state
                notification_type   # Pass the radio button state
            )
            
            st.sidebar.success(f"Screenshot saved: {filename}")
            last_screenshot_time = current_time
    else:
        message_placeholder.empty()
        last_violation_alert_time = current_time - timedelta(seconds=VIOLATION_ALERT_COOLDOWN_SECONDS)

def update_screenshot_gallery():
    """Updates the sidebar gallery with recent screenshots."""
    screenshot_files = sorted(os.listdir(OUTPUT_DIR), reverse=True)
    recent_screenshots = screenshot_files[:5]
    
    with screenshot_gallery.container():
        st.markdown("### Recent Violations")
        for filename in recent_screenshots:
            st.image(os.path.join(OUTPUT_DIR, filename), caption=filename, use_container_width=True)


# ----------------------------------------
# MAIN EXECUTION LOGIC
# ----------------------------------------

if __name__ == "__main__":
    
    # Condition 1: User selects Webcam
    if input_mode == 'Webcam':
        alert_placeholder.info("Webcam active. Detecting live...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam!")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Processing and display logic for webcam
                annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
                save_and_display_alert(annotated_frame, violation_detected, active_violations)
                live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                update_screenshot_gallery()
                
                # # Update stats and logs on every loop iteration
                # display_compliance_stats()
                # display_recent_logs() 
                
            cap.release()

    # Condition 2: User uploads a Photo or Video
    elif uploaded_file is not None:
        alert_placeholder.info(f"Processing {input_mode}...")
        
        # --- Process Photo ---
        if input_mode == 'Photo':
            image_bytes = uploaded_file.read()
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)

            live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
            
            # For static photo, check and save alert immediately
            if violation_detected:
                 violation_msg = ", ".join(set(active_violations))
                 st.sidebar.error(f"Violation detected in photo: {violation_msg}") 
                 
                 # The logging function (which sets the log and calls email)
                 save_and_display_alert(annotated_frame, violation_detected, active_violations)
                 
                 # Since st.rerun() is REMOVED, the log update is relying on cache update speed.
                 
            else:
                 st.sidebar.success("No violations in uploaded photo.")

            update_screenshot_gallery()

        # --- Process Video ---
        elif input_mode == 'Video':
            tfile = uploaded_file.name
            with open(tfile, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            cap = cv2.VideoCapture(tfile)
            if not cap.isOpened():
                st.error(" Could not open uploaded video!")
            else:
                video_status = st.empty()
                frame_count = 0
                while cap.isOpened():
                        
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated_frame, violation_detected, active_violations = get_annotated_frame(frame)
                    save_and_display_alert(annotated_frame, violation_detected, active_violations)
                    live_feed_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                    update_screenshot_gallery()
                    frame_count += 1
                    video_status.info(f"Processing frame {frame_count}...")
                
                video_status.success("Video processing complete!")
                cap.release()
                os.remove(tfile) 
    
    # Final display of logs and stats after processing (or on initial load)
    else:
        st.sidebar.info(f"Select a source in the sidebar.")
        live_feed_placeholder.markdown("""
            <div style='text-align: center; padding: 50px; border: 2px dashed #4B88FF; border-radius: 10px; margin-top: 20px;'>
                <h3>▶ Select an *Input Source* (Photo, Webcam, or Video) in the sidebar to start the Safety Monitor.</h3>
                <p>No video stream is active until a source is selected.</p>
            </div>
        """, unsafe_allow_html=True)
        update_screenshot_gallery()

    st.markdown("---")
    # display_compliance_stats()
    # display_recent_logs()
     # --- ADD THE DOWNLOAD BUTTON AND ON-DEMAND EMAIL BUTTON ---
    col_log, col_button = st.columns([3, 1])
    
    with col_log:
        display_compliance_stats()
        display_recent_logs()
        
    with col_button:
        st.markdown("<br><br><br><br>", unsafe_allow_html=True) # Spacer
        # Add the button that calls the send_summary_on_demand function
        st.button(
            " Send Summary Report Now", 
            on_click=send_summary_on_demand, 
            args=(recipient_email, enable_email)
        )
