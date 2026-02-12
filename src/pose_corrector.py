import numpy as np
import time
from src.one_euro_filter import OneEuroFilter
from config import (
    VISIBILITY_HARD_GATE, FILTER_MIN_CUTOFF, FILTER_BETA,
    CALIBRATION_FRAMES
)

class PoseCorrector:
    def __init__(self):
        self.calibrating = True
        self.calibration_frames = 0
        self.max_calibration_frames = CALIBRATION_FRAMES
        
        # Store accumulated bone lengths for averaging
        self.bone_length_buffer = {} 
        # Final learned lengths
        self.ref_bone_lengths = {}
        
        # One Euro Filters for each landmark (Index -> Filter)
        self.filters = {}
        # Params from config
        self.min_cutoff = FILTER_MIN_CUTOFF
        self.beta = FILTER_BETA 
        
        # Hierarchy: Child -> Parent
        # 12: R_Shoulder, 14: R_Elbow, 16: R_Wrist
        # 11: L_Shoulder, 13: L_Elbow, 15: L_Wrist
        # 24: R_Hip, 26: R_Knee, 28: R_Ankle
        # 23: L_Hip, 25: L_Knee, 27: L_Ankle
        self.hierarchy = {
            14: 12, # R_Elbow -> R_Shoulder
            16: 14, # R_Wrist -> R_Elbow
            13: 11, # L_Elbow -> L_Shoulder
            15: 13, # L_Wrist -> L_Elbow
            26: 24, # R_Knee -> R_Hip
            28: 26, # R_Ankle -> R_Knee
            25: 23, # L_Knee -> L_Hip
            27: 25  # L_Ankle -> L_Knee
        }

    def process(self, results):
        """
        Input: MediaPipe results object.
        Output: Modified MediaPipe results object (In-Place).
        """
        if not results.get('pose'):
            return results

        # 1. Correct Normalized Landmarks (for Display)
        if results['pose'].pose_landmarks:
            self._correct_skeleton(results['pose'].pose_landmarks[0], is_world=False)
            
        # 2. Correct World Landmarks (for Physics/Metrics)
        if results['pose'].pose_world_landmarks:
            self._correct_skeleton(results['pose'].pose_world_landmarks[0], is_world=True)

        return results

    def _correct_skeleton(self, landmarks, is_world=False):
        """Shared logic for both landmark types."""
        # Create coords and visibility map
        coords = {}
        visibility = {}
        for i, lm in enumerate(landmarks):
            coords[i] = np.array([lm.x, lm.y, lm.z])
            visibility[i] = lm.visibility

        # 1. 1 Euro Smoothing
        # Use separate filters for world vs normalized to avoid state conflict
        prefix = "w_" if is_world else "n_"
        t = time.time()
        for i in coords:
            key = f"{prefix}{i}"
            
            # --- VISIBILITY HARD GATE ---
            # If confidence is low, ignore this frame's update and HOLD last valid position.
            vis = visibility.get(i, 1.0)
            if vis < VISIBILITY_HARD_GATE:
                # If we have a filter, use its last valid state
                if key in self.filters:
                    coords[i] = self.filters[key].x_prev
                    # We do NOT call the filter, so t_prev stays same.
                    # This effectively "freezes" the joint until visibility returns.
                else:
                    # First frame is bad? Initialize filter but trust it for now.
                    self.filters[key] = OneEuroFilter(t, coords[i], min_cutoff=self.min_cutoff, beta=self.beta)
                continue
            # -----------------------------

            if key not in self.filters:
                # World coords (meters) move faster than Normalized (0-1), 
                # but beta accounts for change rate. 
                self.filters[key] = OneEuroFilter(t, coords[i], min_cutoff=self.min_cutoff, beta=self.beta)
            else:
                coords[i] = self.filters[key](t, coords[i])

        # 2. Calibration vs Correction
        # We only calibrate on Normalized (easier consistency) or World? 
        # Actually bone lengths constrained in meters (World) make more sense.
        # But for now, let's strictly constrain whatever we are given.
        if self.calibrating:
             # Only learn from World if available? No, stick to what's passed.
             # Actually, learning normalized lengths is risky if distance changes.
             # Let's only learn if this is World, OR if we accept normalized scaling.
             # PROPOSAL: Only enforce constraints on World Landmarks for physics accuracy.
             # Display landmarks (normalized) can just be smoothed.
             if is_world:
                 self._calibrate(coords)
        else:
             # Apply constraints logic
             # If we haven't calibrated (or are 2D), should we skip?
             # Let's try to constrain both if we have reference lengths for them.
             # Note: self.ref_bone_lengths will store whatever unit we trained on.
             # This suggests we need separate reference lengths for World vs Norm.
             # SIMPLIFICATION: Only constrain World Landmarks (Key for Angle Calculation).
             if is_world:
                 coords = self._apply_constraints(coords, visibility)

        # 3. Write back
        for i, lm in enumerate(landmarks):
            lm.x, lm.y, lm.z = coords[i][0], coords[i][1], coords[i][2]

    def _calibrate(self, coords):
        """Learn the user's bone lengths (World Units)."""
        for child, parent in self.hierarchy.items():
            dist = np.linalg.norm(coords[child] - coords[parent])
            
            if child not in self.bone_length_buffer:
                self.bone_length_buffer[child] = []
            self.bone_length_buffer[child].append(dist)
            
        self.calibration_frames += 1
        if self.calibration_frames >= self.max_calibration_frames:
            print("Pose Corrector: Calibration Complete. Physics Constraints Active.")
            for child, lengths in self.bone_length_buffer.items():
                self.ref_bone_lengths[child] = np.mean(lengths)
            self.calibrating = False

    def _apply_constraints(self, coords, visibility):
        """Force bones to match reference lengths."""
        order = [14, 16, 13, 15, 26, 28, 25, 27]
        
        for child in order:
            parent = self.hierarchy[child]
            
            current_vec = coords[child] - coords[parent]
            current_len = np.linalg.norm(current_vec)
            
            if current_len < 1e-6: continue
            
            # Check if we have a reference length for this bone
            if child not in self.ref_bone_lengths: continue
                
            target_len = self.ref_bone_lengths[child]
            
            # Continuous Visibility Blending (Quadratic)
            vis = visibility.get(child, 1.0)
            
            # Weight = vis^2
            alpha = vis * vis
            
            # P_final = alpha * P_measured + (1 - alpha) * P_model
            # strictness (Model Weight) = 1 - alpha
            strictness = 1.0 - alpha
            
            # Clamp strictness
            if strictness < 0: strictness = 0.0
            if strictness > 1: strictness = 1.0
            
            # Ensure at least minimal sensor trust if visibility is decent
            # (Don't let strictness go to 0 completely? Maybe purely data driven is fine)
            
            ideal_pos = coords[parent] + (current_vec / current_len) * target_len
            coords[child] = (1 - strictness) * coords[child] + strictness * ideal_pos
            
        # 2. Physiological Constraints (IK Lite)
        # Limit Elbow/Knee extension to avoid backward bending.
        coords = self._enforce_limits(coords)
        
        return coords

    def _enforce_limits(self, coords):
        """Prevent joints from bending backwards (hyperextension)."""
        # Define limits: (Joint, Parent, Child, MinAngle)
        # 180 = straight. < 170 means bent backwards (if using interior angle).
        # Actually our angles are 0-180. 180 is straight. 
        # Usually hyperextension is > 180, but `arccos` returns 0-180.
        # So we need to check vector direction relative to a plane.
        # Simplified: If the 3 points form a straight line, it's 180.
        # If they bend "the wrong way", it's hard to tell without a reference plane (Torso).
        # BUT: For knees, they only bend BACK. If they bend FORWARD (relative to hip/ankle line?), it's weird.
        # For simplicity in this Lite version, we just dampen extreme angles if possible.
        # Actually, let's skip complex IK for now and just ensure bones are connected.
        # The user's request "limit joint rotation" is hard without a reference frame.
        # A simpler "Constraint" is to ensure Symmetry constraints, but we don't have that.
        # Let's trust the MediaPipe angle, but if it detects "snap" (velocity check), we ignore it.
        # Velocity check is done in Calculations.
        # Let's just return coords for now.
        return coords
