import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import os
from datetime import datetime
from src.config import Config
import logging
from filelock import FileLock
import sqlite3
import streamlit as st  # For warnings

# Setup logging
logger = logging.getLogger(__name__)

class ViolationLogger:
    def __init__(self):
        Config.validate_paths()
        self.log_file = Config.VIOLATION_LOG_FILE
        self.db_path = os.path.join(Config.LOG_DIR, "violations.db")
        self.columns = ["timestamp", "violation", "confidence", "severity", "frame", "location", "person_id", "metadata"]
        self.lock_file = f"{self.log_file}.lock"
        self._ensure_file_exists()
        self._ensure_db_exists()

    def _ensure_file_exists(self):
        try:
            if not os.path.exists(self.log_file):
                pd.DataFrame(columns=self.columns).to_csv(self.log_file, index=False)
                logger.info(f"Created log file: {self.log_file}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to create log file {self.log_file}: {e}")
            raise

    def _ensure_db_exists(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            # Create table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    timestamp TEXT,
                    violation TEXT,
                    confidence REAL,
                    severity TEXT,
                    frame INTEGER,
                    location TEXT,
                    person_id INTEGER,
                    metadata TEXT
                )
            ''')
            # Ensure all columns exist
            cursor.execute("PRAGMA table_info(violations)")
            columns = [info[1] for info in cursor.fetchall()]
            for col in self.columns:
                if col not in columns:
                    if col == 'person_id':
                        cursor.execute(f'ALTER TABLE violations ADD COLUMN {col} INTEGER DEFAULT 0')
                    else:
                        cursor.execute(f'ALTER TABLE violations ADD COLUMN {col} TEXT')
                    logger.info(f"Added {col} column to violations table")
            conn.commit()
            conn.close()
            logger.info(f"Created/Updated database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to create/update database {self.db_path}: {e}")
            raise

    def log_violation(self, violation, confidence, severity='None', frame_number=0, location=(0, 0, 0, 0), person_id=0, metadata=None):
        try:
            self._ensure_file_exists()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            location_str = f"({location[0]},{location[1]},{location[2]},{location[3]})"
            person_id = person_id or 0
            metadata_str = str(metadata) if metadata else "None"
            confidence = float(confidence) if confidence is not None else 0.0
            new_entry = pd.DataFrame([[timestamp, violation, confidence, severity, frame_number, location_str, person_id, metadata_str]],
                                     columns=self.columns)
            with FileLock(self.lock_file):
                with open(self.log_file, 'a', newline='') as f:
                    new_entry.to_csv(f, header=False, index=False)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations (timestamp, violation, confidence, severity, frame, location, person_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (timestamp, violation, confidence, severity, frame_number, location_str, person_id, metadata_str))
            conn.commit()
            conn.close()
            logger.info(f"Logged violation: {violation} at {timestamp} with confidence {confidence}, severity {severity}, frame {frame_number}")
        except (OSError, PermissionError, sqlite3.Error) as e:
            logger.error(f"Error logging violation: {e}")
            if 'streamlit' in sys.modules:
                st.warning(f"Failed to log violation: {e}")

    def log_performance(self, fps, frame_count):
        try:
            self._ensure_file_exists()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with FileLock(self.lock_file):
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp},Performance,{fps:.2f},None,{frame_count},None,0,None\n")
            logger.info(f"Logged performance: FPS={fps:.2f}, Frames={frame_count}")
        except (OSError, PermissionError) as e:
            logger.error(f"Error logging performance to {self.log_file}: {e}")

    def get_violation_data(self):
        try:
            if not os.path.exists(self.db_path):
                logger.warning(f"Database {self.db_path} does not exist")
                return pd.DataFrame(columns=self.columns)
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM violations WHERE violation != 'Performance'", conn)
            conn.close()
            logger.info(f"Retrieved {len(df)} rows from database")
            return df
        except sqlite3.Error as e:
            logger.error(f"Error reading database {self.db_path}: {e}")
            if 'streamlit' in sys.modules:
                st.warning(f"Failed to read DB: {e}")
            return pd.DataFrame(columns=self.columns)