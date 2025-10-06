# check_csv_violations.py
import csv
import os
from datetime import datetime

def check_csv_violations():
    print("🔍 Checking CSV violations...")
    
    csv_path = 'violations/violations_log.csv'
    if os.path.exists(csv_path):
        print(f"✅ CSV violations file exists: {csv_path}")
        
        with open(csv_path, 'r', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
            
        print(f"📊 Total violations in CSV: {len(rows) - 1}")  # minus header
        
        if len(rows) > 1:
            print("📋 Recent violations:")
            for row in rows[-5:]:  # last 5 rows
                print(f"   - {row}")
    else:
        print(f"❌ CSV violations file NOT found: {csv_path}")

if __name__ == "__main__":
    check_csv_violations()