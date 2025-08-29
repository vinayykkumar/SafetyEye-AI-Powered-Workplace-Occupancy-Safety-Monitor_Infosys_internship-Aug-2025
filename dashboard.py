import streamlit as st
import pandas as pd

def run_dashboard():
    st.title("👁️ SafetyEye Dashboard")
    st.subheader("Real-time Workplace Safety Monitor")

    st.video("sample_feed.mp4")  
    stats = {"Total Frames": 1500, "Violations Detected": 12, "Helmet Compliance": "92%"}
    st.write("### Compliance Stats")
    st.json(stats)

    violation_log = pd.DataFrame([
        {"Time": "10:05 AM", "Type": "No Helmet"},
        {"Time": "10:15 AM", "Type": "No Vest"},
    ])
    st.write("### Violation Logs")
    st.table(violation_log)

if __name__ == "__main__":
    run_dashboard()
