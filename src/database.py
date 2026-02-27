import json
import uuid
import time
import threading
import queue
import csv
import io
from typing import Optional
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
                              combined_derived_data TEXT,
                              kinematics_flat_data TEXT,
                              confidence_data TEXT)'''
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
        joint_confidence = {}
        if pose_3d and 'pose_3d' in pose_3d:
            # Flatten 3D dict to list for storage
            # Format: [{'id': 0, 'x': 1.2, ...}, ...]
            for lm_id, lm_data in pose_3d['pose_3d'].items():
                row = dict(lm_data)
                row['id'] = lm_id
                p3d_data.append(row)
                joint_confidence[int(lm_id)] = {
                    'confidence': float(lm_data.get('visibility', 0.0)),
                    'method': lm_data.get('method', 'unknown'),
                    'reproj_error': lm_data.get('reproj_error')
                }

        combined_derived = {
            'kinematics_3d': pose_3d.get('kinematics_3d', {}) if pose_3d else {},
            'low_reliability_landmarks': pose_3d.get('low_reliability_landmarks', []) if pose_3d else [],
            'joint_confidence': joint_confidence,
            'timestamp_ns': pose_3d.get('timestamp_ns') if pose_3d else None
        }

        kinematics_flat = {}
        confidence_data = {
            'joint_confidence': joint_confidence,
            'low_reliability_landmarks': combined_derived['low_reliability_landmarks']
        }
        if pose_3d and 'kinematics_3d' in pose_3d:
            kinematics_flat = pose_3d['kinematics_3d'].get('flat_export', {})
        
        data = {
            'type': 'multi',
            'timestamp': timestamp,
            # PC1
            'pose': p1_pose, 'face': p1_face, 'hand': p1_hand, 'derived': p1_derived,
            # PC2
            'pc2_pose': p2_pose, 'pc2_face': p2_face, 'pc2_hand': p2_hand, 'pc2_derived': p2_derived,
            # 3D
            'pose_3d': p3d_data,
            'combined_derived': combined_derived,
            'kinematics_flat': kinematics_flat,
            'confidence_data': confidence_data,
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
                        cols = "(timestamp, pose_data, face_data, hand_data, derived_data, pc2_pose_data, pc2_face_data, pc2_hand_data, pc2_derived_data, pose_3d_data, combined_derived_data, kinematics_flat_data, confidence_data)"
                        vals = "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?" if self.db_type == 'sqlite' else "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s"

                        c.execute(f"INSERT INTO {table_name} {cols} VALUES ({vals})", (
                            item['timestamp'],
                            json.dumps(item['pose']), json.dumps(item['face']), json.dumps(item['hand']), json.dumps(item['derived']),
                            json.dumps(item['pc2_pose']), json.dumps(item['pc2_face']), json.dumps(item['pc2_hand']), json.dumps(item['pc2_derived']),
                            json.dumps(item['pose_3d']), json.dumps(item['combined_derived']),
                            json.dumps(item.get('kinematics_flat', {})), json.dumps(item.get('confidence_data', {}))
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

        has_kinematics_flat = 'kinematics_flat_data' in cols

        if is_multi and has_kinematics_flat:
            query = f"SELECT timestamp, pose_data, derived_data, pc2_pose_data, pc2_derived_data, pose_3d_data, kinematics_flat_data FROM {table_name} ORDER BY timestamp ASC"
        elif is_multi:
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

                # Flat kinematics (optional)
                if has_kinematics_flat and len(row) > 6 and row[6]:
                    try:
                        kin = json.loads(row[6])
                        for k, v in kin.items():
                            writer.writerow([ts, 'KIN', 0, k, v, '', '', ''])
                    except: pass
                      
        return output.getvalue()

    # =========================================================================
    # Dataset Pipeline — JSON Export, Session Archive, Raw Frame Archival
    # =========================================================================

    def export_session_json(self, session_id: str = None) -> Optional[str]:
        """
        Export a session to a structured JSON string.
        
        If session_id is None, exports the latest session.
        Returns JSON string or None.
        """
        conn = self._get_connection()
        c = conn.cursor()

        if session_id:
            placeholder = '%s' if self.db_type == 'postgres' else '?'
            c.execute(f"SELECT id, start_time, table_name FROM sessions WHERE id = {placeholder}", (session_id,))
        else:
            c.execute("SELECT id, start_time, table_name FROM sessions ORDER BY start_time DESC LIMIT 1")
        
        row = c.fetchone()
        if not row:
            conn.close()
            return None
        
        sid, start_time, table_name = row
        if not table_name:
            conn.close()
            return None

        # Determine schema
        if self.db_type == 'sqlite':
            c.execute(f"PRAGMA table_info({table_name})")
            cols = [info[1] for info in c.fetchall()]
        else:
            cols = []

        # Fetch all rows
        c.execute(f"SELECT * FROM {table_name} ORDER BY timestamp ASC")
        rows = c.fetchall()
        conn.close()

        frames = []
        for row_data in rows:
            frame = {'id': row_data[0], 'timestamp': row_data[1]}
            for i, col_name in enumerate(cols):
                if i <= 1:
                    continue  # id and timestamp already added
                val = row_data[i] if i < len(row_data) else None
                if val and isinstance(val, str):
                    try:
                        frame[col_name] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        frame[col_name] = val
                else:
                    frame[col_name] = val
            frames.append(frame)

        export = {
            'session_id': sid,
            'start_time': start_time,
            'table_name': table_name,
            'total_frames': len(frames),
            'schema_columns': cols,
            'frames': frames
        }
        return json.dumps(export, indent=2, default=str)

    def archive_session(self, session_id: str = None, output_dir: str = 'results') -> Optional[str]:
        """
        Archive a session as a ZIP bundle containing:
          - session_metadata.json (session info)
          - data.csv (CSV export)
          - data.json (JSON export)
          - raw_frames/ (if raw frames were saved)
        
        Returns path to ZIP file or None.
        """
        import zipfile
        import os
        import glob

        conn = self._get_connection()
        c = conn.cursor()

        if session_id:
            placeholder = '%s' if self.db_type == 'postgres' else '?'
            c.execute(f"SELECT id, start_time, table_name FROM sessions WHERE id = {placeholder}", (session_id,))
        else:
            c.execute("SELECT id, start_time, table_name FROM sessions ORDER BY start_time DESC LIMIT 1")

        row = c.fetchone()
        conn.close()
        if not row:
            return None

        sid, start_time, table_name = row
        if not table_name:
            return None

        os.makedirs(output_dir, exist_ok=True)
        zip_name = f"{table_name}_archive.zip"
        zip_path = os.path.join(output_dir, zip_name)

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                # 1. Metadata
                metadata = {
                    'session_id': sid,
                    'start_time': start_time,
                    'table_name': table_name,
                    'archive_created': datetime.now().isoformat()
                }
                zf.writestr('session_metadata.json', json.dumps(metadata, indent=2))

                # 2. CSV export
                csv_data = self.export_latest_session_csv()
                if csv_data:
                    zf.writestr('data.csv', csv_data)

                # 3. JSON export
                json_data = self.export_session_json(sid)
                if json_data:
                    zf.writestr('data.json', json_data)

                # 4. Raw frames (if saved)
                raw_dir = os.path.join(output_dir, 'raw_frames', table_name)
                if os.path.isdir(raw_dir):
                    for fpath in sorted(glob.glob(os.path.join(raw_dir, '*.jpg'))):
                        arcname = os.path.join('raw_frames', os.path.basename(fpath))
                        zf.write(fpath, arcname)

            print(f"[Database] Session archived to {zip_path}")
            return zip_path
        except Exception as e:
            print(f"[Database] Archive error: {e}")
            return None

    def save_raw_frame(self, frame_bgr, frame_number: int, camera_id: str = 'local',
                       output_dir: str = 'results'):
        """
        Save a raw JPEG frame to disk for dataset archival.
        
        Args:
            frame_bgr: BGR numpy array
            frame_number: Sequential frame number
            camera_id: Camera identifier
            output_dir: Base output directory
        """
        import os
        import cv2 as _cv2
        
        if not self.current_table:
            return
        
        raw_dir = os.path.join(output_dir, 'raw_frames', self.current_table)
        os.makedirs(raw_dir, exist_ok=True)
        
        filename = f"{camera_id}_{frame_number:06d}.jpg"
        filepath = os.path.join(raw_dir, filename)
        
        try:
            _cv2.imwrite(filepath, frame_bgr, [_cv2.IMWRITE_JPEG_QUALITY, 95])
        except Exception as e:
            if frame_number % 100 == 0:
                print(f"[Database] Raw frame save error: {e}")

    def get_session_list(self) -> list:
        """Return list of all sessions with metadata."""
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT id, start_time, table_name FROM sessions ORDER BY start_time DESC")
        rows = c.fetchall()
        conn.close()
        return [
            {'id': r[0], 'start_time': r[1], 'table_name': r[2]}
            for r in rows
        ]
