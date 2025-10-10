import pandas as pd
from src.config import Config

def generate_violation_report(df):
    if df.empty:
        total_persons = 0
    else:
        total_persons = df['confidence'].sum() if 'confidence' in df.columns and df['violation'].str.contains('Person', na=False).any() else 0
    
    violation_filter = Config.VIOLATION_CLASSES + ["Unsafe Posture: Bending"]
    violation_df = df[df["violation"].isin(violation_filter)].copy()
    
    if violation_df.empty:
        return f"No violations detected. Total persons monitored: {total_persons}."
    
    report_lines = [f"Total persons monitored: {total_persons}."]
    violation_counts = violation_df["violation"].value_counts()
    for violation, count in violation_counts.items():
        item = violation.replace("NO-", "").replace("Hardhat", "Hard hat")
        timestamps = df[df["violation"] == violation]["timestamp"].dropna().tolist()
        timestamp_str = ", ".join(timestamps[:3]) + f" (and {len(timestamps)-3} more)" if len(timestamps) > 3 else ", ".join(timestamps)
        report_lines.append(f"- Person detected not wearing {item}: {count} time{'s' if count > 1 else ''} at {timestamp_str}")
    return "\n".join(report_lines)