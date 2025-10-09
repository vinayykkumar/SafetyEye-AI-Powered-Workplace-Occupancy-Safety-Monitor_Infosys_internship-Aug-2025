# =========================================================
# AI-Powered Construction Safety Monitoring & Analytics
# =========================================================
import time
import streamlit as st
import cv2, os, io, shutil, tempfile, base64
import numpy as np
import pandas as pd
from PIL import Image
import plotly.io as pio
from datetime import datetime, timedelta
from ultralytics import YOLO
import plotly.express as px
import smtplib
from email.message import EmailMessage
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import Counter

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SafetyEye Dashboard", layout="wide", page_icon="👷")

st.title("🛡 AI-Powered Construction Safety Monitor")

# ---------------- MODEL & PATHS ----------------
MODEL_PATH = r"C:\SafetyEye\models\best.pt"  # Change as per your path
model = YOLO(MODEL_PATH)

OUTPUT_DIR_ROOT = "outputs_new"
OUTPUT_DIR = os.path.join(OUTPUT_DIR_ROOT, "violation_screenshots")
LOG_FILE = os.path.join(OUTPUT_DIR_ROOT, "violation_log.csv")
VIOLATION_CLASSES = ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']

os.makedirs(OUTPUT_DIR_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=['Timestamp', 'ViolationType', 'PersonID', 'ScreenshotPath']).to_csv(LOG_FILE, index=False)

# ---------------- EMAIL CONFIG ----------------
SENDER_EMAIL = "quickqart30@gmail.com"
SENDER_PASSWORD = "....."
EMAIL_COOLDOWN_SECONDS = 60
last_batch_email_time = datetime.now() - timedelta(seconds=EMAIL_COOLDOWN_SECONDS)
batch_violations, batch_screenshots = [], []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📥 Input & Settings")
input_mode = st.sidebar.radio("Input Source", ["Photo", "Webcam", "Video"])
uploaded_file = None
if input_mode == "Photo":
    uploaded_file = st.sidebar.file_uploader("Upload Photo", type=['jpg','jpeg','png'])
elif input_mode == "Video":
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4','avi','mov'])

st.sidebar.markdown("---")
st.sidebar.title("⚙ Detection Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.title("📧 Email Notifications")
enable_email = st.sidebar.checkbox("Enable Email", value=False)
recipient_email = st.sidebar.text_input("Recipient Email", "admin@example.com", disabled=not enable_email)
notification_type = st.sidebar.radio(
    "Notification Type",
    ['Real-time Violation Alerts', 'Summary Report (CSV)'],
    disabled=not enable_email
)



# ---------------- PLACEHOLDERS ----------------
live_feed_placeholder = st.empty()
alert_placeholder = st.empty()
screenshot_gallery = st.empty()

# ---------------- DEEPSORT ----------------
tracker = DeepSort(max_age=30)
person_violation_tracker = {}

# ---------------- HELPER FUNCTIONS ----------------
def load_log_data():
    if os.path.exists(LOG_FILE):
        try:
            df = pd.read_csv(LOG_FILE)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
            return df.dropna(subset=["Timestamp"])
        except Exception:
            return pd.DataFrame(columns=['Timestamp', 'ViolationType', 'PersonID', 'ScreenshotPath'])
    return pd.DataFrame(columns=['Timestamp', 'ViolationType', 'PersonID', 'ScreenshotPath'])

def get_annotated_frame(frame, conf=0.5):
    results = model(frame, conf=conf, verbose=False)
    annotated_frame = frame.copy()
    violation_detected = False
    active_violations = []
    detections = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        color = (0,255,0)
        if label in VIOLATION_CLASSES:
            color = (0,0,255)
            violation_detected = True
            active_violations.append(label)
        cv2.rectangle(annotated_frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(annotated_frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
        detections.append(([x1,y1,x2-x1,y2-y1], 1.0, label))

    tracks = tracker.update_tracks(detections, frame=annotated_frame)
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        bbox = track.to_ltrb()
        cv2.putText(annotated_frame, f"ID:{track_id}", (int(bbox[0]), int(bbox[1]-25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    return annotated_frame, violation_detected, active_violations, tracks

def save_and_log_violation(frame, violations, tracks):
    global person_violation_tracker
    timestamp = datetime.now()
    filename = None
    persons_to_log = []

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        last_time = person_violation_tracker.get(track_id, datetime.min)
        if (timestamp - last_time).total_seconds() > 40:
            persons_to_log.append(track_id)
            person_violation_tracker[track_id] = timestamp

    if persons_to_log and violations:
        filename = os.path.join(OUTPUT_DIR, f"violation_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg")
        cv2.imwrite(filename, frame)
        for track_id in persons_to_log:
            pd.DataFrame([[timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                           ", ".join(set(violations)),
                           track_id,
                           filename]],
                         columns=['Timestamp','ViolationType','PersonID','ScreenshotPath']).to_csv(LOG_FILE, mode='a', header=False, index=False)
        alert_placeholder.error(f"🚨 Violation Detected: {', '.join(set(violations))} for IDs: {', '.join(map(str, persons_to_log))}")

    return filename, timestamp.strftime("%Y-%m-%d %H:%M:%S")

# ---------------- EMAIL FUNCTIONS ----------------
def send_batch_email():
    pass  # Implement same as your code for batch emails

def send_summary_report():
    pass  # Implement same as your code for summary CSV report

# ---------------- DASHBOARD TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📹 Monitoring", "📊 Compliance Stats", "📜 Violation Logs","📅 Reports & Data Export","⚙ Settings"])

# --- TAB 1: Monitoring ---
# --- TAB 1: Monitoring ---
with tab1:
    st.markdown("### 🎥 Monitoring Zone")

    # Initialize session_state variables
    if "monitoring_state" not in st.session_state:
        st.session_state.monitoring_state = "stopped"  # options: stopped, running, paused

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("▶ Start/Resume"):
            st.session_state.monitoring_state = "running"
    with col2:
        if st.button("⏸ Pause"):
            st.session_state.monitoring_state = "paused"
    with col3:
        if st.button("⏹ End"):
            st.session_state.monitoring_state = "stopped"

    live_feed_placeholder = st.empty()  # placeholder for video frames

    if input_mode == "Webcam" and st.session_state.monitoring_state != "stopped":
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            while True:
                if st.session_state.monitoring_state == "paused":
                    time.sleep(0.1)
                    continue
                elif st.session_state.monitoring_state == "stopped":
                    break

                ret, frame = cap.read()
                if not ret: break
                frame_resized = cv2.resize(frame, (640,480))
                annotated_frame, violation_detected, violations, tracks = get_annotated_frame(frame_resized, conf_threshold)
                if violation_detected:
                    screenshot_path, _ = save_and_log_violation(annotated_frame, violations, tracks)
                live_feed_placeholder.image(annotated_frame, channels="BGR", width="stretch")
        cap.release()

    elif input_mode == "Photo" and uploaded_file:
        image = Image.open(io.BytesIO(uploaded_file.read()))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        annotated_frame, violation_detected, violations, tracks = get_annotated_frame(frame, conf_threshold)
        live_feed_placeholder.image(annotated_frame, channels="BGR", width="stretch")
        if violation_detected:
            save_and_log_violation(annotated_frame, violations, tracks)

    elif input_mode == "Video" and uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()
        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            if st.session_state.monitoring_state == "paused":
                time.sleep(0.1)
                continue
            elif st.session_state.monitoring_state == "stopped":
                break

            ret, frame = cap.read()
            if not ret: break
            frame_resized = cv2.resize(frame, (640,480))
            annotated_frame, violation_detected, violations, tracks = get_annotated_frame(frame_resized, conf_threshold)
            if violation_detected:
                save_and_log_violation(annotated_frame, violations, tracks)
            live_feed_placeholder.image(annotated_frame, channels="BGR", width="stretch")
        cap.release()
        os.remove(tfile.name)


# --- TAB 2: Compliance Stats ---
# ---------------- TAB 2 : Compliance Stats ----------------
with tab2:
    st.markdown("### 📊 Compliance Statistics")

    df = load_log_data()
    if df.empty:
        st.info("No data to show compliance stats.")
    else:
        # Convert Timestamp to datetime
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        # ---------------- FILTER SECTION ----------------
        st.markdown("#### 🔍 Filter Options")
        col_f1, col_f2, col_f3 = st.columns(3)

        filter_mode = col_f1.selectbox(
            "Filter By Date",
            ["All Time", "Specific Date", "Date Range", "Last Month", "Last Year"],
            key="compliance_filter_mode"
        )

        # Apply date filter
        if filter_mode == "Specific Date":
            selected_date = col_f2.date_input("Select Date", datetime.now().date(), key="specific_date")
            filtered_df = df[df["Timestamp"].dt.date == selected_date]

        elif filter_mode == "Date Range":
            start_date = col_f2.date_input("Start Date", datetime.now().date() - timedelta(days=7), key="start_date")
            end_date = col_f3.date_input("End Date", datetime.now().date(), key="end_date")
            filtered_df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

        elif filter_mode == "Last Month":
            today = datetime.now().date()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            filtered_df = df[(df["Timestamp"].dt.date >= first_day_prev_month) & (df["Timestamp"].dt.date <= last_day_prev_month)]

        elif filter_mode == "Last Year":
            today = datetime.now().date()
            start_last_year = datetime(today.year - 1, 1, 1).date()
            end_last_year = datetime(today.year - 1, 12, 31).date()
            filtered_df = df[(df["Timestamp"].dt.date >= start_last_year) & (df["Timestamp"].dt.date <= end_last_year)]

        else:
            filtered_df = df.copy()  # All Time

        # ---------------- DISPLAY STATS ----------------
        if filtered_df.empty:
            st.info("No data available for selected filters.")
        else:
            # Violation stats
            violation_counts = filtered_df['ViolationType'].str.split(',').explode().str.strip().value_counts()
            total_violations = len(filtered_df)
            total_persons = len(filtered_df['PersonID'].unique())
            compliance_rate = f"{round((total_persons - total_violations) / total_persons * 100, 2) if total_persons else 0}%"

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Violations", total_violations)
            col2.metric("Most Frequent Violation", violation_counts.index[0] if not violation_counts.empty else "N/A")
            col3.metric("Compliance Rate", compliance_rate)

            # ---------------- BAR CHART ----------------
            chart_df = violation_counts.reset_index()
            chart_df.columns = ['Violation Type', 'Count']
            fig = px.bar(
                chart_df,
                x='Violation Type',
                y='Count',
                color='Violation Type',
                color_discrete_sequence=px.colors.qualitative.Bold,
                title=f"Violations ({filter_mode})"
            )
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='black')
            st.plotly_chart(fig, use_container_width=True)

            # Save chart and CSV temporarily
            chart_path = "compliance_chart.png"
            csv_path = "compliance_report.csv"
            pio.write_image(fig, chart_path, format="png")
            filtered_df.to_csv(csv_path, index=False)

            # ---------------- SEND EMAIL BUTTON ----------------
            st.markdown("---")
            st.markdown("### 📧 Send Compliance Report")
            if st.button("📤 Send Compliance Report to Admin Email"):
                try:
                    msg = EmailMessage()
                    msg['Subject'] = f"📊 Compliance Report ({filter_mode}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    msg['From'] = SENDER_EMAIL
                    msg['To'] = SENDER_EMAIL
                    msg.set_content("Attached is the latest compliance report with graph and data summary.")

                    # Attach chart
                    with open(chart_path, 'rb') as f:
                        msg.add_attachment(f.read(), maintype='image', subtype='png', filename="compliance_chart.png")

                    # Attach CSV
                    with open(csv_path, 'rb') as f:
                        msg.add_attachment(f.read(), maintype='text', subtype='csv', filename="compliance_report.csv")

                    # Send Email
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                        smtp.send_message(msg)

                    st.success(f"✅ Compliance report sent successfully to {SENDER_EMAIL}!")

                except Exception as e:
                    st.error(f"❌ Failed to send report: {e}")

with tab3:
    st.markdown("### 📜 Recent Violation Logs")

    df = load_log_data()  # Load data

    if not df.empty:
        # ---------------- DATE RANGE FILTER ----------------
        st.subheader("🔎 Filter Logs")
        col1, col2 = st.columns(2)

        # Convert Timestamp column to datetime
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        with col1:
            start_date = st.date_input("Start Date", value=df['Timestamp'].min().date())
        with col2:
            end_date = st.date_input("End Date", value=df['Timestamp'].max().date())

        # Apply date filter
        mask = (df['Timestamp'].dt.date >= start_date) & (df['Timestamp'].dt.date <= end_date)
        filtered_df = df.loc[mask]

        # ---------------- VIOLATION TYPE FILTER ----------------
        if 'ViolationType' in df.columns:
            all_violations = sorted(df['ViolationType'].dropna().unique().tolist())
            selected_violations = st.multiselect(
                "Select Violation Types",
                options=all_violations,
                default=all_violations
            )

            filtered_df = filtered_df[filtered_df['ViolationType'].isin(selected_violations)]

        # ---------------- DISPLAY DATA ----------------
        if not filtered_df.empty:
            st.markdown(f"### Showing {len(filtered_df)} records")
            st.dataframe(filtered_df.tail(10))

            # ---------------- DOWNLOAD CSV ----------------
            csv_data = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv_data.encode()).decode()
            st.markdown(
                f'<a href="data:file/csv;base64,{b64}" download="filtered_violation_log.csv">📥 Download Filtered Log</a>',
                unsafe_allow_html=True
            )

            # ---------------- EMAIL REPORT ----------------
            st.markdown("---")
            st.subheader("📧 Send Report to Admin")

            if st.button("📤 Send Report to Admin"):
                csv_path = f"filtered_violation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                filtered_df.to_csv(csv_path, index=False)

                try:
                    msg = EmailMessage()
                    msg['Subject'] = f"📋 Safety Violation Report ({start_date} → {end_date})"
                    msg['From'] = SENDER_EMAIL
                    msg['To'] = SENDER_EMAIL
                    msg.set_content(
                        f"""Hello Admin,

Please find attached the latest Safety Violation Report.

🗓 Date Range: {start_date} → {end_date}
🚨 Violations Included: {', '.join(selected_violations)}
📊 Total Records: {len(filtered_df)}

Best regards,
Safety Monitoring Dashboard
"""
                    )

                    with open(csv_path, 'rb') as f:
                        msg.add_attachment(
                            f.read(),
                            maintype='text',
                            subtype='csv',
                            filename=os.path.basename(csv_path)
                        )

                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                        smtp.send_message(msg)

                    st.success(f"✅ Report sent successfully to {SENDER_EMAIL}!")

                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")

        else:
            st.warning("⚠ No violations found for the selected filters.")
    else:
        st.info("No violation logs available yet.")

# --- TAB 4: Reports ---
with tab4:
    st.markdown("## 📅 Reports & Data Export")

    # Load data
    df = load_log_data()

    if df.empty:
        st.info("⚠ No data available to generate reports yet.")
    else:
        # Convert timestamp to datetime safely
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

        st.markdown("### 🔍 Select Report Type")
        col1, col2, col3 = st.columns(3)

        report_mode = col1.selectbox(
            "Report Type",
            ["Daily", "Date Range", "Last Month", "Last Year"],
            key="tab4_report_mode"
        )

        # Initialize date variables
        start_date, end_date = None, None

        # ---------------- FILTER MODES ----------------
        if report_mode == "Daily":
            selected_date = col2.date_input("Select Date", datetime.now().date(), key="tab4_daily_date")
            start_date = end_date = selected_date

        elif report_mode == "Date Range":
            start_date = col2.date_input("Start Date", datetime.now().date() - timedelta(days=7), key="tab4_range_start")
            end_date = col3.date_input("End Date", datetime.now().date(), key="tab4_range_end")

        elif report_mode == "Last Month":
            today = datetime.now().date()
            first_day_prev_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
            last_day_prev_month = today.replace(day=1) - timedelta(days=1)
            start_date, end_date = first_day_prev_month, last_day_prev_month

        elif report_mode == "Last Year":
            today = datetime.now().date()
            start_date = datetime(today.year - 1, 1, 1).date()
            end_date = datetime(today.year - 1, 12, 31).date()

        # ---------------- FILTER DATA ----------------
        filtered_df = df[(df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date)]

        st.markdown("---")
        st.markdown(f"### 📊 Data Summary ({start_date} → {end_date})")

        if filtered_df.empty:
            st.warning(f"No data found between {start_date} and {end_date}.")
        else:
            st.success(f"✅ Found {len(filtered_df)} violation records.")
            st.dataframe(filtered_df.tail(10), use_container_width=True)

            # ---------------- REPORT GENERATION ----------------
            zip_generated = False
            csv_path = None
            chart_path = None
            zip_path = None

            if st.button("📥 Generate Report ZIP", key="tab4_generate_zip"):
                try:
                    range_label = f"{start_date}to{end_date}".replace(":", "-")
                    report_dir = os.path.join(OUTPUT_DIR_ROOT, f"report_{range_label}")
                    images_dir = os.path.join(report_dir, "Images")
                    visual_dir = os.path.join(report_dir, "VisualReports")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(visual_dir, exist_ok=True)

                    # 1️⃣ Save CSV
                    csv_path = os.path.join(report_dir, "violation_data.csv")
                    filtered_df.to_csv(csv_path, index=False)

                    # 2️⃣ Copy screenshots if column exists
                    if "ScreenshotPath" in filtered_df.columns:
                        for path in filtered_df["ScreenshotPath"].dropna():
                            if os.path.exists(path):
                                shutil.copy(path, os.path.join(images_dir, os.path.basename(path)))

                    # 3️⃣ Generate chart if column exists
                    if "ViolationType" in filtered_df.columns:
                        violation_counts = (
                            filtered_df["ViolationType"].str.split(",").explode().value_counts()
                        )
                        if not violation_counts.empty:
                            chart_df = violation_counts.reset_index()
                            chart_df.columns = ["Violation Type", "Count"]
                            fig = px.bar(
                                chart_df,
                                x="Violation Type",
                                y="Count",
                                color="Violation Type",
                                color_discrete_sequence=px.colors.qualitative.Bold,
                                title=f"Violations ({start_date} → {end_date})"
                            )
                            chart_path = os.path.join(visual_dir, f"chart_{range_label}.png")
                            fig.write_image(chart_path)

                    # 4️⃣ Create ZIP
                    zip_path = shutil.make_archive(report_dir, 'zip', report_dir)
                    zip_generated = True

                    # Download link
                    with open(zip_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="SafetyEye_Report_{range_label}.zip">📦 Download Full Report (ZIP)</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("✅ Report ZIP generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating report: {e}")

            # ---------------- SEND EMAIL ----------------
            if zip_generated and st.button("📤 Send Report ZIP to Admin Email", key="tab4_send_email"):
                try:
                    msg = EmailMessage()
                    msg["Subject"] = f"📅 SafetyEye Report ({report_mode}) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    msg["From"] = SENDER_EMAIL
                    msg["To"] = SENDER_EMAIL
                    msg.set_content(f"Attached is the generated SafetyEye Report for {start_date} to {end_date}.")

                    # Attach ZIP
                    with open(zip_path, "rb") as f:
                        msg.add_attachment(f.read(), maintype="application", subtype="zip", filename=os.path.basename(zip_path))

                    # Send Email
                    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                        smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
                        smtp.send_message(msg)

                    st.success(f"✅ Report successfully emailed to {SENDER_EMAIL}")

                except Exception as e:
                    st.error(f"❌ Failed to send email: {e}")
# ---------------- Sidebar Settings Button ----------------
with tab5:
    st.markdown("## ⚙ Settings")

    # ---------------- Manage Violations ----------------
    st.markdown("### 🧹 Manage Violations")
    st.info("Use the button below to clear all violations and reset logs.")

    if st.button("🗑 Clear All Violations", key="clear_violations"):
        try:
            if os.path.exists(LOG_FILE):
                pd.DataFrame(columns=['Timestamp', 'ViolationType', 'PersonID', 'ScreenshotPath']).to_csv(LOG_FILE, index=False)
            shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            st.success("✅ All violations cleared successfully!")
        except Exception as e:
            st.error(f"❌ Failed to clear violations: {e}")

    # ---------------- Email Configuration ----------------
    st.markdown("### 📧 Email Configuration")
    sender_email_input = st.text_input("Sender Email", value=SENDER_EMAIL)
    sender_password_input = st.text_input("Sender Password", type="password", value=SENDER_PASSWORD)
    if st.button("💾 Save Email Settings"):
        SENDER_EMAIL = sender_email_input
        SENDER_PASSWORD = sender_password_input
        st.success("✅ Email settings saved!")

    # ---------------- Report Preferences ----------------
    st.markdown("### 📊 Report Preferences")
    default_report_mode = st.selectbox(
        "Default Report Type",
        ["Daily", "Date Range", "Last Month", "Last Year"],
        index=0
    )
    default_csv_name = st.text_input("Default CSV Filename", "violation_report.csv")
    if st.button("💾 Save Report Preferences"):
        st.success(f"✅ Report preferences saved! Default mode: {default_report_mode}, CSV name: {default_csv_name}")