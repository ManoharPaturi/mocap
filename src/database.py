import json
import uuid
import time
import threading
import queue
import csv
import io
from datetime import datetime
from config import DB_TYPE, DB_PATH, POSTGRES_CONNECTION
from src.calculations import Calculations

# Conditional imports
if DB_TYPE == 'postgres':
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

# ... imports remain same
from config import DB_TYPE, DB_PATH, POSTGRES_CONNECTION
from src.calculations import Calculations

# Conditional imports
if DB_TYPE == 'postgres':
    import psycopg2
    import psycopg2.extras
else:
    import sqlite3

class MocapDB:
    def __init__(self):
        self.db_type = DB_TYPE
        self.queue = queue.Queue()
        self.running = False
        self.session_id = None
        self.current_table = None # Track current table name
        self.thread = None
        self._init_db()

    def _get_connection(self):
        """Get database connection based on type."""
        if self.db_type == 'postgres':
            return psycopg2.connect(**POSTGRES_CONNECTION)
        else:
            return sqlite3.connect(DB_PATH)

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Sessions Master Table
        # id: UUID, start_time: ISO timestamp, table_name: storage table
        schema = '''CREATE TABLE IF NOT EXISTS sessions
                     (id TEXT PRIMARY KEY, 
                      start_time TEXT,
                      table_name TEXT)'''
        c.execute(schema)
        
        # Schema Migration: Add table_name if missing
        try:
            if self.db_type == 'sqlite':
                c.execute("ALTER TABLE sessions ADD COLUMN table_name TEXT")
            else:
                c.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS table_name TEXT")
        except: pass
            
        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_type.upper()}")

    def start_recording(self):
        """Start a new recording session."""
        self.session_id = str(uuid.uuid4())
        timestamp = datetime.now()
        start_time = timestamp.isoformat()
        
        # Create Table Name: session_YYYYMMDD_HHMMSS
        date_str = timestamp.strftime("%Y%m%d_%H%M%S")
        self.current_table = f"session_{date_str}"
        
        conn = self._get_connection()
        c = conn.cursor()
        
        # 1. Create Dynamic Table for this session
        create_table_sql = f'''CREATE TABLE IF NOT EXISTS {self.current_table}
                             (id {("SERIAL PRIMARY KEY" if self.db_type == 'postgres' else "INTEGER PRIMARY KEY AUTOINCREMENT")},
                              timestamp REAL,
                              pose_data TEXT,
                              face_data TEXT,
                              hand_data TEXT,
                              derived_data TEXT)'''
        c.execute(create_table_sql)
        
        # 2. Register Session
        insert_sql = "INSERT INTO sessions (id, start_time, table_name) VALUES (%s, %s, %s)" if self.db_type == 'postgres' else \
                     "INSERT INTO sessions (id, start_time, table_name) VALUES (?, ?, ?)"
        c.execute(insert_sql, (self.session_id, start_time, self.current_table))
        
        conn.commit()
        conn.close()
        
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
        print(f"Recording Started: {self.session_id} -> Table: {self.current_table}")
        return self.session_id

    def stop_recording(self):
        """Stop the current recording session."""
        self.running = False
        if self.thread:
            self.thread.join()
        saved_id = self.session_id
        self.session_id = None
        self.current_table = None
        print("Recording Stopped.")
        return saved_id

    def save_frame(self, results):
        """Queue a frame of results for saving."""
        if not self.running or not self.session_id:
            return
            
        # Serialize raw landmarks
        pose_data = self._serialize_landmarks(results.get('pose'), 'pose_landmarks')
        face_data = self._serialize_landmarks(results.get('face'), 'face_landmarks')
        hand_data = self._serialize_landmarks(results.get('hand'), 'hand_landmarks')
        
        # Calculate Derived Metrics
        derived_data = []
        
        for p_idx, person_pose in enumerate(pose_data):
            p_metrics = Calculations.get_body_metrics(person_pose)
            if p_idx < len(face_data):
                f_metrics = Calculations.get_face_metrics(face_data[p_idx])
                p_metrics.update(f_metrics)
            derived_data.append(p_metrics)
            
        data = {
            'timestamp': time.time(),
            'pose': pose_data,
            'face': face_data,
            'hand': hand_data,
            'derived': derived_data
        }
        self.queue.put(data)

    def _serialize_landmarks(self, result, attr_name):
        """Convert MediaPipe results to JSON-serializable list."""
        if not result or not getattr(result, attr_name):
            return []
        
        all_people = []
        for person_landmarks in getattr(result, attr_name):
            person_data = []
            for lm in person_landmarks:
                lm_dict = {'x': round(lm.x, 5), 'y': round(lm.y, 5), 'z': round(lm.z, 5)}
                if hasattr(lm, 'visibility'):
                    lm_dict['v'] = round(lm.visibility, 5)
                person_data.append(lm_dict)
            all_people.append(person_data)
        return all_people

    def _worker_loop(self):
        """Background thread to batch insert data into specific session table."""
        conn = self._get_connection()
        table_name = self.current_table # Capture for this thread
        
        while self.running or not self.queue.empty():
            try:
                batch = []
                while len(batch) < 50:
                    try:
                        item = self.queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                if not batch:
                    if not self.running: 
                        break
                    continue

                c = conn.cursor()
                for item in batch:
                    # Insert into the dynamically named table
                    # Note: Using python f-string for table name is safe here as it's internally generated from date
                    insert_sql = f'''INSERT INTO {table_name} 
                                     (timestamp, pose_data, face_data, hand_data, derived_data) 
                                     VALUES (%s, %s, %s, %s, %s)''' if self.db_type == 'postgres' else \
                                 f'''INSERT INTO {table_name} 
                                     (timestamp, pose_data, face_data, hand_data, derived_data) 
                                     VALUES (?, ?, ?, ?, ?)'''
                                     
                    c.execute(insert_sql, (
                                   item['timestamp'], 
                                   json.dumps(item['pose']), 
                                   json.dumps(item['face']), 
                                   json.dumps(item['hand']),
                                   json.dumps(item['derived'])))
                conn.commit()
                
            except Exception as e:
                print(f"DB Error: {e}")
        
        conn.close()

    def export_latest_session_csv(self):
        """Export the most recent session to a CSV string."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Get latest session AND its table name
        c.execute("SELECT id, table_name FROM sessions ORDER BY start_time DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        session_id, table_name = row
        
        # Handle fallback for old sessions (before refactor)
        if not table_name:
            # Fallback to old 'frames' table logic (omitted for brevity, assuming new sessions)
             query = "SELECT timestamp, pose_data, face_data, hand_data, derived_data FROM frames WHERE session_id = %s ORDER BY timestamp ASC" if self.db_type == 'postgres' else \
                    "SELECT timestamp, pose_data, face_data, hand_data, derived_data FROM frames WHERE session_id = ? ORDER BY timestamp ASC"
             c.execute(query, (session_id,))
        else:
            # Query the dynamic table
            query = f"SELECT timestamp, pose_data, face_data, hand_data, derived_data FROM {table_name} ORDER BY timestamp ASC"
            c.execute(query)
            
        rows = c.fetchall()
        conn.close()
        
        if not rows:
            return None
            
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header - Dynamic based on what metrics we find? 
        # For simplicity, we define a standard set + the raw nodes
        header = ['Timestamp', 'Type', 'Person_Index', 'Data_Key', 'Value_X_or_Metric', 'Value_Y', 'Value_Z', 'Confidence']
        writer.writerow(header)
        
        for row in rows:
            timestamp = row[0]
            pose_json = row[1]
            face_json = row[2]
            hand_json = row[3]
            derived_json = row[4] if len(row) > 4 and row[4] else "[]"
            
            # Helper to write raw landmarks
            def write_landmarks(data, l_type):
                people = json.loads(data)
                for p_idx, person in enumerate(people):
                    for l_idx, lm in enumerate(person):
                        writer.writerow([
                            timestamp, 
                            l_type, 
                            p_idx, 
                            l_idx, # Data Key = Landmark Index
                            lm['x'], 
                            lm['y'], 
                            lm['z'], 
                            lm.get('v', '')
                        ])

            # Export Raw
            write_landmarks(pose_json, 'POSE')
            write_landmarks(face_json, 'FACE')
            write_landmarks(hand_json, 'HAND')
            
            # Export Derived Metrics
            try:
                derived = json.loads(derived_json)
                for p_idx, metrics in enumerate(derived):
                    for key, value in metrics.items():
                        writer.writerow([
                            timestamp,
                            'METRIC',
                            p_idx,
                            key, # Data Key = Metric Name
                            value, # Value stored in X column
                            '', '', '' 
                        ])
            except:
                pass
            
        return output.getvalue()
