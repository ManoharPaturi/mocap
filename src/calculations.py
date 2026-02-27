import numpy as np
from config import (
    VISIBILITY_MIN_METRIC, ANGLE_OUTLIER_BASE_THRESHOLD,
    ANGLE_OUTLIER_VELOCITY_COEFF, MAX_LINEAR_VELOCITY,
    SMOOTHING_ALPHA_DEFAULT, SMOOTHING_ALPHA_FACE,
    BODY_HEIGHT_MULTIPLIER, IPD_DEFAULT
)

class Calculations:
    POSE_IDX = {
        'left_shoulder': 11,
        'right_shoulder': 12,
        'left_elbow': 13,
        'right_elbow': 14,
        'left_wrist': 15,
        'right_wrist': 16,
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
    }

    @staticmethod
    def calculate_angle(a, b, c, vector_b_to_a=None):
        """
        Calculate 3D angle. 
        Optionally accept pre-calculated vector_b_to_a (v1) to allow for Spine-relative angles.
        Default: Angle between BA and BC.
        """
        # Vector 1 (BA)
        if vector_b_to_a is not None:
            ba = vector_b_to_a
        else:
            ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
            
        # Vector 2 (BC)
        bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
        
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        
        if norm_ba == 0 or norm_bc == 0:
            return 0.0
        
        # Calculate cosine using dot product
        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        
        # Clip to handle floating point errors slightly outside [-1, 1]
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return round(np.degrees(angle), 2)

    @staticmethod
    def calculate_distance(a, b):
        """Calculate 3D Euclidean distance."""
        pa = np.array([a['x'], a['y'], a['z']])
        pb = np.array([b['x'], b['y'], b['z']])
        return round(float(np.linalg.norm(pa - pb)), 4)

    @staticmethod
    def get_segment_vectors_from_pose_3d(pose_3d):
        """
        Return canonical body segment vectors from a 3D pose dict/list.
        Conventions:
            Upper Arm (R): Elbow - Shoulder
            Forearm (R): Wrist - Elbow
            Trunk Axis: MidShoulder - MidHip
            Pelvic Axis: RightHip - LeftHip
        """
        def p(index):
            if isinstance(pose_3d, dict):
                return pose_3d.get(index)
            if isinstance(pose_3d, list) and index < len(pose_3d):
                return pose_3d[index]
            return None

        ls = p(Calculations.POSE_IDX['left_shoulder'])
        rs = p(Calculations.POSE_IDX['right_shoulder'])
        re = p(Calculations.POSE_IDX['right_elbow'])
        rw = p(Calculations.POSE_IDX['right_wrist'])
        lh = p(Calculations.POSE_IDX['left_hip'])
        rh = p(Calculations.POSE_IDX['right_hip'])

        segments = {}
        if rs and re:
            segments['upper_arm_r'] = {
                'x': re['x'] - rs['x'],
                'y': re['y'] - rs['y'],
                'z': re['z'] - rs['z']
            }
        if re and rw:
            segments['forearm_r'] = {
                'x': rw['x'] - re['x'],
                'y': rw['y'] - re['y'],
                'z': rw['z'] - re['z']
            }
        if ls and rs and lh and rh:
            mid_shoulder = {
                'x': (ls['x'] + rs['x']) / 2.0,
                'y': (ls['y'] + rs['y']) / 2.0,
                'z': (ls['z'] + rs['z']) / 2.0,
            }
            mid_hip = {
                'x': (lh['x'] + rh['x']) / 2.0,
                'y': (lh['y'] + rh['y']) / 2.0,
                'z': (lh['z'] + rh['z']) / 2.0,
            }
            segments['trunk_axis'] = {
                'x': mid_shoulder['x'] - mid_hip['x'],
                'y': mid_shoulder['y'] - mid_hip['y'],
                'z': mid_shoulder['z'] - mid_hip['z']
            }
        if lh and rh:
            segments['pelvic_axis'] = {
                'x': rh['x'] - lh['x'],
                'y': rh['y'] - lh['y'],
                'z': rh['z'] - lh['z']
            }

        return segments

    @staticmethod
    def get_joint_angles_from_pose_3d(pose_3d):
        """Compute canonical 3D joint angles in degrees, clamped to [0, 180]."""
        def p(index):
            if isinstance(pose_3d, dict):
                return pose_3d.get(index)
            if isinstance(pose_3d, list) and index < len(pose_3d):
                return pose_3d[index]
            return None

        def angle(a_idx, b_idx, c_idx):
            a = p(a_idx)
            b = p(b_idx)
            c = p(c_idx)
            if not all([a, b, c]):
                return None
            value = Calculations.calculate_angle(a, b, c)
            return float(np.clip(value, 0.0, 180.0))

        angles = {}
        pairs = {
            'Angle_Elbow_L': ('left_shoulder', 'left_elbow', 'left_wrist'),
            'Angle_Elbow_R': ('right_shoulder', 'right_elbow', 'right_wrist'),
            'Angle_Knee_L': ('left_hip', 'left_knee', 'left_ankle'),
            'Angle_Knee_R': ('right_hip', 'right_knee', 'right_ankle'),
        }

        for key, (a_name, b_name, c_name) in pairs.items():
            value = angle(
                Calculations.POSE_IDX[a_name],
                Calculations.POSE_IDX[b_name],
                Calculations.POSE_IDX[c_name]
            )
            if value is not None:
                angles[key] = value

        return angles

    @staticmethod
    def get_body_metrics(pose_landmarks):
        """
        Calculate joint angles and limb lengths from MediaPipe Pose landmarks.
        Refined: Shoulders/Hips use Spine Vector for stability.
        """
        if not pose_landmarks or len(pose_landmarks) < 33:
            return {}

        metrics = {}
        lm = pose_landmarks
        min_vis = VISIBILITY_MIN_METRIC

        def v(indices):
            """Check if all landmarks in indices have sufficient visibility."""
            for idx in indices:
                if 'v' in lm[idx] and lm[idx]['v'] < min_vis:
                    return False
            return True

        # --- VIRTUAL LANDMARKS (Spine) ---
        # Mid-Hip
        mh_x = (lm[23]['x'] + lm[24]['x']) / 2
        mh_y = (lm[23]['y'] + lm[24]['y']) / 2
        mh_z = (lm[23]['z'] + lm[24]['z']) / 2
        
        # Mid-Shoulder
        ms_x = (lm[11]['x'] + lm[12]['x']) / 2
        ms_y = (lm[11]['y'] + lm[12]['y']) / 2
        ms_z = (lm[11]['z'] + lm[12]['z']) / 2
        
        # Spine Vector (MidHip -> MidShoulder) pointing UP
        spine_vec = np.array([ms_x - mh_x, ms_y - mh_y, ms_z - mh_z])

        # --- JOINT ANGLES (Degrees) ---
        
        # Elbows: Shoulder -> Elbow -> Wrist (Standard)
        if v([11,13,15]): metrics['Angle_Elbow_L'] = Calculations.calculate_angle(lm[11], lm[13], lm[15])
        if v([12,14,16]): metrics['Angle_Elbow_R'] = Calculations.calculate_angle(lm[12], lm[14], lm[16])
        
        # Shoulders: Spine -> Shoulder -> Elbow
        # Angle between Vertical Spine and Upper Arm
        if v([11,13]): 
            # We want angle at Shoulder. Vector1 = pre-calced Spine. Vector2 = Shoulder->Elbow.
            # calculate_angle logic: BA vs BC. We pass BA = Spine. B = Shoulder. C = Elbow.
            # Note: spine_vec is UP.
            metrics['Angle_Shoulder_L'] = Calculations.calculate_angle(None, lm[11], lm[13], vector_b_to_a=spine_vec)
        
        if v([12,14]): 
            metrics['Angle_Shoulder_R'] = Calculations.calculate_angle(None, lm[12], lm[14], vector_b_to_a=spine_vec)
        
        # Hips: Spine -> Hip -> Knee
        # Angle between Spine and Upper Leg
        # We use NEGATIVE spine vec (Down) for hips? Or just measure deviation from straight line?
        # Standard: Hip Flexion is angle between Trunk and Thigh.
        # Trunk vector = Spine (Up). Thigh vector = Hip->Knee (Down).
        # Extended leg = 180 deg. Flexed = 90 deg.
        spine_down = -spine_vec
        if v([23,25]): metrics['Angle_Hip_L'] = Calculations.calculate_angle(None, lm[23], lm[25], vector_b_to_a=spine_vec)
        if v([24,26]): metrics['Angle_Hip_R'] = Calculations.calculate_angle(None, lm[24], lm[26], vector_b_to_a=spine_vec)
        
        # Knees: Hip -> Knee -> Ankle (Standard)
        if v([23,25,27]): metrics['Angle_Knee_L'] = Calculations.calculate_angle(lm[23], lm[25], lm[27])
        if v([24,26,28]): metrics['Angle_Knee_R'] = Calculations.calculate_angle(lm[24], lm[26], lm[28])

        # --- LIMB LENGTHS (Normalized Units 0-1) ---
        if v([11,13]): metrics['Length_UpperArm_L'] = Calculations.calculate_distance(lm[11], lm[13])
        if v([13,15]): metrics['Length_LowerArm_L'] = Calculations.calculate_distance(lm[13], lm[15])
        if v([12,14]): metrics['Length_UpperArm_R'] = Calculations.calculate_distance(lm[12], lm[14])
        if v([14,16]): metrics['Length_LowerArm_R'] = Calculations.calculate_distance(lm[14], lm[16])
        
        if v([23,25]): metrics['Length_UpperLeg_L'] = Calculations.calculate_distance(lm[23], lm[25])
        if v([25,27]): metrics['Length_LowerLeg_L'] = Calculations.calculate_distance(lm[25], lm[27])
        if v([24,26]): metrics['Length_UpperLeg_R'] = Calculations.calculate_distance(lm[24], lm[26])
        if v([26,28]): metrics['Length_LowerLeg_R'] = Calculations.calculate_distance(lm[26], lm[28])
        
        if v([11,12]): metrics['Width_Shoulder'] = Calculations.calculate_distance(lm[11], lm[12])
        if v([23,24]): metrics['Width_Hip'] = Calculations.calculate_distance(lm[23], lm[24])

        return metrics

    @staticmethod
    def get_face_metrics(face_landmarks):
        """
        Calculate facial features (Mouth openness, etc.).
        Expects list of dicts.
        """
        # 468 landmarks default
        if not face_landmarks or len(face_landmarks) < 468:
            return {}
            
        metrics = {}
        lm = face_landmarks
        
        # --- MOUTH ---
        # Lips Vertical: Upper(13) <-> Lower(14)
        mouth_h = Calculations.calculate_distance(lm[13], lm[14])
        # Lips Horizontal: Left(61) <-> Right(291)
        mouth_w = Calculations.calculate_distance(lm[61], lm[291])
        
        # --- EYES (Interpupillary Distance - IPD) ---
        ipd = IPD_DEFAULT  # Default from config
        if 468 in lm and 473 in lm:
             ipd = Calculations.calculate_distance(lm[159], lm[386])
        else:
             ipd = Calculations.calculate_distance(lm[33], lm[263]) * 0.5 
             
        if ipd == 0: ipd = 1.0
        
        # Metrics normalized by IPD
        metrics['Face_Mouth_Openness'] = round(mouth_h / ipd, 4)
        metrics['Face_Smile_Ratio'] = round(mouth_w / (mouth_h + 1e-6), 4) # Ratio stays W/H
        
        # ... Eyes Openness (Ratio) ...
        # Left Eye: Vertical 159-145, Horizontal 33-133
        l_eye_v = Calculations.calculate_distance(lm[159], lm[145])
        l_eye_h = Calculations.calculate_distance(lm[33], lm[133])
        metrics['Face_Eye_L_Openness'] = round(l_eye_v / (l_eye_h + 1e-6), 4)
        
        # Right Eye: Vertical 386-374, Horizontal 362-263
        r_eye_v = Calculations.calculate_distance(lm[386], lm[374])
        r_eye_h = Calculations.calculate_distance(lm[362], lm[263])
        metrics['Face_Eye_R_Openness'] = round(r_eye_v / (r_eye_h + 1e-6), 4)

        return metrics

    @staticmethod
    def get_kinematics(current_lm, prev_lm, current_metrics, prev_metrics, dt):
        """
        Calculate instantaneous linear and angular velocities.
        Returns a dict of velocities.
        Units: Meters/sec (approx if coords normalized) or Units/sec, and Deg/sec.
        """
        if dt <= 0: return {}
        
        kinematics = {}
        
        # --- ANGULAR VELOCITY (Deg/s) ---
        # Keys like 'Angle_Elbow_L'
        for key, val in current_metrics.items():
            if key.startswith('Angle_') and key in prev_metrics:
                diff = val - prev_metrics[key]
                # Handle wrapping if needed (though joint angles usually don't wrap abruptly like heading)
                vel = diff / dt
                kinematics[f'Velocity_{key}'] = vel

        # --- LINEAR VELOCITY (Units/s) ---
        # Calculate for key joints: Wrists(15/16), Ankles(27/28), Hips(23/24)
        # current_lm is list of dicts {'x', 'y', 'z', 'v'}
        
        def calc_vel(idx, name):
            if idx < len(current_lm) and idx < len(prev_lm):
                c = current_lm[idx]
                p = prev_lm[idx]
                # Check visibility
                if c.get('v', 1) < VISIBILITY_MIN_METRIC or p.get('v', 1) < VISIBILITY_MIN_METRIC:
                    return
                
                # dist = sqrt(dx^2 + dy^2 + dz^2)
                dx = c['x'] - p['x']
                dy = c['y'] - p['y']
                dz = c['z'] - p['z']
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                
                vel = dist / dt
                
                # Sanity Check: Human motion cap (approx 20 km/h or 6 m/s for limbs)
                if vel > 6.0: 
                    return
                    
                kinematics[f'Velocity_{name}'] = vel

        calc_vel(15, 'Wrist_L')
        calc_vel(16, 'Wrist_R')
        calc_vel(27, 'Ankle_L')
        calc_vel(28, 'Ankle_R')
        
        return kinematics

    @staticmethod
    def normalize_metrics(metrics, pose_landmarks):
        """
        Normalize all limb lengths by body height (Nose to Mid-Hip).
        Adds 'Normalized_' keys to preserve Raw values.
        """
        if not metrics or not pose_landmarks:
            return metrics
            
        lm = pose_landmarks
        
        # Calculate Body Height Reference
        # User Suggestion: Hip Center -> Ankle Average (More stable than Head)
        
        # Mid-Hip
        hip_x = (lm[23]['x'] + lm[24]['x']) / 2
        hip_y = (lm[23]['y'] + lm[24]['y']) / 2
        hip_z = (lm[23]['z'] + lm[24]['z']) / 2
        mid_hip = {'x': hip_x, 'y': hip_y, 'z': hip_z}
        
        # Mid-Ankle
        ank_x = (lm[27]['x'] + lm[28]['x']) / 2
        ank_y = (lm[27]['y'] + lm[28]['y']) / 2
        ank_z = (lm[27]['z'] + lm[28]['z']) / 2
        mid_ankle = {'x': ank_x, 'y': ank_y, 'z': ank_z}
        
        # Height Ref = Distance from Mid-Hip to Mid-Ankle (Leg Length approx)
        # Taking "Body Height" usually means full height. 
        # But this reference is stable. Let's call it "BodyScale".
        height = Calculations.calculate_distance(mid_hip, mid_ankle) * BODY_HEIGHT_MULTIPLIER 
        # Multiply by 2.0 to approximate full stature (Legs ~ half height)? 
        # Or just use the raw distance as the unit. 
        # User said: "Scaling lengths by body height... Hip center -> ankle average".
        # Let's use the raw leg length as the unit "1.0".
        if height == 0: height = 1.0 # Avoid div by zero
        
        # Normalize all 'Length_' or 'Width_' metrics
        # Create NEW keys so we keep Raw (Meters) and Normalized (Ratio)
        for k, v in list(metrics.items()): # copy list safely
            if k.startswith('Length_') or k.startswith('Width_'):
                metrics[f"Normalized_{k}"] = round(v / height, 4)
                
        metrics['Body_Height'] = round(height, 4)
        return metrics
    
    @staticmethod
    def filter_and_smooth(current, prev, alpha=SMOOTHING_ALPHA_DEFAULT):
        """
        1. Outlier Rejection: Velocity-Dependent Threshold.
        2. Smoothing: Apply EMA (Exponential Moving Average).
        """
        if not prev:
            return current
            
        filtered = {}
        for k, v in current.items():
            # Only smooth Angles and Lengths
            if k not in prev:
                filtered[k] = v
                continue
                
            p_val = prev[k]
            
            # 1. Outlier Rejection (Angles only)
            if k.startswith('Angle_'):
                diff = abs(v - p_val)
                
                # Dynamic Threshold: 50 deg + k * velocity
                # If we have a stored velocity for this metric, use it.
                # Since 'Velocity_' keys might not be in 'prev' (or are computed *after* this),
                # we can approximate angular velocity by just looking at the raw diff if needed,
                # BUT the user suggested using 'angular_velocity_prev'.
                # Let's check if 'Velocity_{k}' exists in prev.
                vel_key = f"Velocity_{k}"
                threshold = ANGLE_OUTLIER_BASE_THRESHOLD
                if vel_key in prev:
                    threshold += ANGLE_OUTLIER_VELOCITY_COEFF * abs(prev[vel_key])
                
                if diff > threshold:
                    # Spike detected! Ignore new value, keep old.
                    filtered[k] = p_val 
                    continue
            
                if diff > threshold:
                    # Spike detected! Ignore new value, keep old.
                    filtered[k] = p_val 
                    continue
            
            # 2. EMA Smoothing
            # Default alpha 0.5
            a = alpha
            
            # Use smoother alpha for Face Metrics (micro-jitter)
            if k.startswith('Face_'):
                a = SMOOTHING_ALPHA_FACE  # Stronger smoothing for face
            
            smoothed_val = a * v + (1 - a) * p_val
            
            # Rounding
            if k.startswith('Angle_'):
                filtered[k] = round(smoothed_val, 2)
            else:
                filtered[k] = round(smoothed_val, 4)
                
        return filtered
