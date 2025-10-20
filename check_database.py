# check_database.py
import sqlite3
import os

def check_database():
    print("🔍 Checking database status...")
    
    # Check if violations folder exists
    violations_dir = 'violations'
    if os.path.exists(violations_dir):
        print(f"✅ Violations directory exists: {violations_dir}")
        print(f"📁 Files in violations directory: {os.listdir(violations_dir)}")
    else:
        print(f"❌ Violations directory NOT found: {violations_dir}")
    
    # Check if database file exists
    db_path = 'violations/violations.db'
    if os.path.exists(db_path):
        print(f"✅ Database file exists: {db_path}")
        
        # Check what's in the database
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if violations table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='violations'")
            table_exists = cursor.fetchone()
            
            if table_exists:
                print("✅ Violations table exists")
                
                # Count violations
                cursor.execute("SELECT COUNT(*) FROM violations")
                count = cursor.fetchone()[0]
                print(f"📊 Total violations in database: {count}")
                
                # Show recent violations
                cursor.execute("SELECT * FROM violations ORDER BY timestamp DESC LIMIT 5")
                recent = cursor.fetchall()
                print(f"📋 Recent violations:")
                for violation in recent:
                    print(f"   - {violation}")
            else:
                print("❌ Violations table does NOT exist")
                
            conn.close()
            
        except Exception as e:
            print(f"❌ Error accessing database: {e}")
    else:
        print(f"❌ Database file NOT found: {db_path}")

if __name__ == "__main__":
    check_database()