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
        
        # 1. Create Dynamic Table for this session (Expanded for Multi-Cam)
        create_table_sql = f'''CREATE TABLE IF NOT EXISTS {self.current_table}
                             (id {("SERIAL PRIMARY KEY" if self.db_type == 'postgres' else "INTEGER PRIMARY KEY AUTOINCREMENT")},
                              timestamp REAL,
                              -- PC1 (Local)
                              pose_data TEXT,
                              face_data TEXT,
                              hand_data TEXT,
                              derived_data TEXT,
                              -- PC2 (Remote)
                              pc2_pose_data TEXT,
                              pc2_face_data TEXT,
                              pc2_hand_data TEXT,
                              pc2_derived_data TEXT,
                              -- 3D / Combined
                              pose_3d_data TEXT,
                              combined_derived_data TEXT)'''
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
        """Queue a single-camera frame (legacy/PC1-only mode)."""
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
            'type': 'single',
            'timestamp': time.time(),
            'pose': pose_data,
            'face': face_data,
            'hand': hand_data,
            'derived': derived_data
        }
        self.queue.put(data)

    def save_synced_frame(self, timestamp, pc1_results, pc2_results, pose_3d):
        """Queue a synchronized multi-camera frame."""
        if not self.running or not self.session_id:
            return

        def process_results(results):
            if not results: return [], [], [], []
            msg_pose = results.get('pose')
            # Handle FrameData results which might have different structure or be raw dicts
            # If coming from FrameData, results is a dict with 'pose': SimpleNamespace/list...
            # Actually master_coordinator passes dicts now.
            
            # Helper to safely get from dict or object
            def get_attr(obj, attr):
                if isinstance(obj, dict): return obj.get(attr)
                return getattr(obj, attr, None)
            
            # Serialize
            pose = self._serialize_landmarks(results, 'pose') # Adjusted: expecting 'pose' key in dict
            face = self._serialize_landmarks(results, 'face')
            hand = self._serialize_landmarks(results, 'hand')
            
            derived = []
            for p_idx, person_pose in enumerate(pose):
                p_metrics = Calculations.get_body_metrics(person_pose)
                if p_idx < len(face):
                     # Need to convert face list back to dict format expected by Calculations
                     # _serialize_landmarks returns list of dicts {'x':...}
                     # Calculations expects list of dicts directly
                    f_metrics = Calculations.get_face_metrics(face[p_idx])
                    p_metrics.update(f_metrics)
                derived.append(p_metrics)
            return pose, face, hand, derived

        # PC1
        p1_pose, p1_face, p1_hand, p1_derived = process_results(pc1_results)
        
        # PC2
        p2_pose, p2_face, p2_hand, p2_derived = process_results(pc2_results)
        
        # 3D
        p3d_data = []
        if pose_3d and 'pose_3d' in pose_3d:
            # Flatten 3D dict to list for storage
            # Format: [{'id': 0, 'x': 1.2, ...}, ...]
            for lm_id, lm_data in pose_3d['pose_3d'].items():
                lm_data['id'] = lm_id
                p3d_data.append(lm_data)
        
        data = {
            'type': 'multi',
            'timestamp': timestamp,
            # PC1
            'pose': p1_pose, 'face': p1_face, 'hand': p1_hand, 'derived': p1_derived,
            # PC2
            'pc2_pose': p2_pose, 'pc2_face': p2_face, 'pc2_hand': p2_hand, 'pc2_derived': p2_derived,
            # 3D
            'pose_3d': p3d_data,
            'combined_derived': [] # Placeholder for future 3D metrics
        }
        self.queue.put(data)

    def _serialize_landmarks(self, results, key_or_attr):
        """Convert MediaPipe results or FrameData results to JSON-serializable list."""
        val = None
        
        # 1. EXTRACT DATA
        if isinstance(results, dict):
             val = results.get(key_or_attr)
             # Fallback: key 'pose' might contain object with 'pose_landmarks'
             if not val and key_or_attr == 'pose_landmarks':
                 val = results.get('pose')
        else:
             val = getattr(results, key_or_attr, None)

        if not val: return []
        
        # 2. NORMALIZE TO LIST OF PEOPLE [(lm1, lm2...), (lm1...)]
        all_people_landmarks = []
        
        # CASE A: MediaPipe Solution Output (Object)
        # It has .pose_landmarks or .face_landmarks attribute which is a LIST of NormalizedLandmarkList
        if hasattr(val, 'pose_landmarks'):
             all_people_landmarks = val.pose_landmarks
        elif hasattr(val, 'face_landmarks'):
             all_people_landmarks = val.face_landmarks
        elif hasattr(val, 'hand_landmarks'):
             all_people_landmarks = val.hand_landmarks
             
        # CASE B: Already a list (Direct access or deserialized)
        elif isinstance(val, list):
             all_people_landmarks = val
             
        # CASE C: Single NormalizedLandmarkList (Rare, but possible in some MP versions)
        elif hasattr(val, 'landmark'): # It's a single set of landmarks
             all_people_landmarks = [val]
             
        else:
             # Unknown type or empty
             return []

        if not all_people_landmarks: return []

        # 3. SERIALIZE EACH PERSON
        serialized_people = []
        
        for person in all_people_landmarks:
             # 'person' is a list of landmarks (or NormalizedLandmarkList)
             # OR 'person' is a dict (if already serialized)?
             
             # Check if 'person' is actually a full result object (nested error case)
             if hasattr(person, 'pose_landmarks'): 
                 continue # Skip invalid nesting
                 
             person_data = []
             
             # Iterate landmarks in this person
             try:
                 for lm in person:
                     lm_dict = {}
                     # Handle Obj vs Dict
                     if isinstance(lm, dict):
                         lm_dict = {
                             'x': round(lm.get('x',0), 5), 
                             'y': round(lm.get('y',0), 5), 
                             'z': round(lm.get('z',0), 5)
                         }
                         if 'v' in lm: lm_dict['v'] = lm['v']
                         elif 'visibility' in lm: lm_dict['v'] = lm['visibility']
                     else:
                         # MediaPipe Landmark Object
                         lm_dict = {
                             'x': round(lm.x, 5), 
                             'y': round(lm.y, 5), 
                             'z': round(lm.z, 5)
                         }
                         if hasattr(lm, 'visibility'):
                             lm_dict['v'] = round(lm.visibility, 5)
                     
                     person_data.append(lm_dict)
                 serialized_people.append(person_data)
             except TypeError:
                 pass # Not iterable

        return serialized_people

    def _worker_loop(self):
        """Background thread to batch insert data."""
        conn = self._get_connection()
        table_name = self.current_table
        
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
                    if not self.running: break
                    continue

                c = conn.cursor()
                for item in batch:
                    if item.get('type') == 'multi':
                         # Multi-camera insert (PC1 + PC2 + 3D)
                         cols = "(timestamp, pose_data, face_data, hand_data, derived_data, pc2_pose_data, pc2_face_data, pc2_hand_data, pc2_derived_data, pose_3d_data, combined_derived_data)"
                         vals = "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?" if self.db_type == 'sqlite' else "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"
                         
                         c.execute(f"INSERT INTO {table_name} {cols} VALUES ({vals})", (
                              item['timestamp'],
                              json.dumps(item['pose']), json.dumps(item['face']), json.dumps(item['hand']), json.dumps(item['derived']),
                              json.dumps(item['pc2_pose']), json.dumps(item['pc2_face']), json.dumps(item['pc2_hand']), json.dumps(item['pc2_derived']),
                              json.dumps(item['pose_3d']), json.dumps(item['combined_derived'])
                         ))
                    else:
                         # Single camera insert (PC1 only) - Fill others with NULL/Empty
                         cols = "(timestamp, pose_data, face_data, hand_data, derived_data)"
                         vals = "?, ?, ?, ?, ?" if self.db_type == 'sqlite' else "%s, %s, %s, %s, %s"
                         
                         c.execute(f"INSERT INTO {table_name} {cols} VALUES ({vals})", (
                              item['timestamp'],
                              json.dumps(item['pose']), json.dumps(item['face']), json.dumps(item['hand']), json.dumps(item['derived'])
                         ))
                         
                conn.commit()
            except Exception as e:
                print(f"DB Error: {e}")
        conn.close()

    def export_latest_session_csv(self):
        """Export the most recent session to a CSV string."""
        conn = self._get_connection()
        c = conn.cursor()
        
        # Get latest session
        c.execute("SELECT id, table_name FROM sessions ORDER BY start_time DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        session_id, table_name = row
        if not table_name:
             # Fallback logic omitted
             conn.close()
             return None
             
        # Check columns to decide query type
        # SQLite pragma
        if self.db_type == 'sqlite':
             c.execute(f"PRAGMA table_info({table_name})")
             cols = [info[1] for info in c.fetchall()]
        else:
             # Postgres assumption
             cols = [] # TODO
        
        is_multi = 'pc2_pose_data' in cols
        
        if is_multi:
             query = f"SELECT timestamp, pose_data, derived_data, pc2_pose_data, pc2_derived_data, pose_3d_data FROM {table_name} ORDER BY timestamp ASC"
        else:
             query = f"SELECT timestamp, pose_data, derived_data FROM {table_name} ORDER BY timestamp ASC"
             
        c.execute(query)
        rows = c.fetchall()
        conn.close()
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # CSV Header
        header = ['Timestamp', 'Source', 'Person', 'Key', 'Value_X', 'Value_Y', 'Value_Z', 'Conf']
        writer.writerow(header)
        
        for row in rows:
            ts = row[0]
            
            # Helper
            def write_data(source, pose_json, derived_json):
                 if pose_json:
                      try:
                           people = json.loads(pose_json)
                           for p_i, p in enumerate(people):
                                for l_i, lm in enumerate(p):
                                     writer.writerow([ts, source, p_i, l_i, lm.get('x'), lm.get('y'), lm.get('z'), lm.get('v','')])
                      except: pass
                 if derived_json:
                      try:
                           derived = json.loads(derived_json)
                           for p_i, m in enumerate(derived):
                                for k, v in m.items():
                                     writer.writerow([ts, source, p_i, k, v, '', '', ''])
                      except: pass

            # PC1
            write_data('PC1', row[1], row[2])
            
            if is_multi:
                 # PC2
                 write_data('PC2', row[3], row[4])
                 
                 # 3D
                 p3d_json = row[5]
                 if p3d_json:
                      try:
                           pts = json.loads(p3d_json)
                           # list of dicts {'id':..., 'x':..., ...}
                           for Pt in pts:
                                writer.writerow([ts, '3D', 0, Pt.get('id'), Pt.get('x'), Pt.get('y'), Pt.get('z'), Pt.get('visibility')])
                      except: pass
                      
        return output.getvalue()
