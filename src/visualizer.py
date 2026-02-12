import cv2
import mediapipe as mp
import time
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from collections import deque
from config import SHOW_FPS, THEME_COLOR
from src.calculations import Calculations

class Visualizer:
    def __init__(self):
        self.frame_times = deque(maxlen=30)
        self.last_time = None
        
    def draw_landmarks(self, frame, results):
        """Draw pose, face, and hand landmarks."""
        pose_result = results.get('pose')
        face_result = results.get('face')
        hand_result = results.get('hand')
        
        # 1. Draw Pose
        if pose_result and pose_result.pose_landmarks:
            for pose_landmarks in pose_result.pose_landmarks:
                pose_proto = landmark_pb2.NormalizedLandmarkList()
                pose_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in pose_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

                # --- DRAW ANGLES ---
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    pose_proto,
                    solutions.pose.POSE_CONNECTIONS,
                    solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        # 2. Draw Face Mesh
        if face_result and face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                face_proto = landmark_pb2.NormalizedLandmarkList()
                face_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in face_landmarks
                ])
                # Tesselation
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    face_proto,
                    solutions.face_mesh.FACEMESH_TESSELATION,
                    None,
                    solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                )

        # 3. Draw Hands
        if hand_result and hand_result.hand_landmarks:
            for hand_landmarks in hand_result.hand_landmarks:
                hand_proto = landmark_pb2.NormalizedLandmarkList()
                hand_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) 
                    for lm in hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style()
                )
            
        return frame

    def draw_fps(self, frame):
        """Draw FPS on frame."""
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, THEME_COLOR, 2)
        return frame
    
    def get_fps(self):
        """Calculate and return current FPS."""
        now = time.time()
        if self.last_time:
            frame_time = now - self.last_time
            self.frame_times.append(frame_time)
        self.last_time = now
        
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                return 1.0 / avg_frame_time
        return 0.0
