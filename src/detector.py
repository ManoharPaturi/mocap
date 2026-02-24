import mediapipe as mp
import cv2
import time
import numpy as np
import platform
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import (
    MODEL_PATHS, POSE_MODEL_COMPLEXITY, NUM_POSES, NUM_FACES, NUM_HANDS,
    MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE,
    MAX_FRAME_WIDTH, GAMMA_DEFAULT, GAMMA_MIN, GAMMA_MAX,
    FACE_EXPOSURE_TARGET, FACE_EXPOSURE_MIN_GAIN, FACE_EXPOSURE_MAX_GAIN,
    FACE_EXPOSURE_MIN_BRIGHTNESS, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE,
    ENABLE_FACE_DETECTION, ENABLE_HAND_DETECTION, ENABLE_FACE_EXPOSURE,
    ENABLE_ROI_CROPPING, ROI_EXPANSION_FACTOR, ROI_MIN_SIZE,
    ROI_TARGET_SIZE, ROI_SMOOTHING_ALPHA, PREFER_GPU_DELEGATE,
    INFERENCE_BACKEND
)

class MocapDetector:
    def __init__(self):
        self.enable_face = ENABLE_FACE_DETECTION
        self.enable_hand = ENABLE_HAND_DETECTION
        
        # Imaging Settings
        self.gamma = GAMMA_DEFAULT
        self.face_exposure = ENABLE_FACE_EXPOSURE
        self.last_face_rect = None # (x, y, w, h)
        self.last_timestamp_ms = 0
        
        # ROI Cropping
        self.enable_roi_cropping = ENABLE_ROI_CROPPING
        self.last_roi = None  # (x, y, w, h) in original frame coords
        self.roi_expansion = ROI_EXPANSION_FACTOR
        self.roi_min_size = ROI_MIN_SIZE
        self.roi_target_size = ROI_TARGET_SIZE
        self.roi_alpha = ROI_SMOOTHING_ALPHA
        
        # Select Pose Model based on config
        pose_model_path = MODEL_PATHS.get(POSE_MODEL_COMPLEXITY, MODEL_PATHS['LITE'])
        print(f"Loading Pose Model: {POSE_MODEL_COMPLEXITY} ({pose_model_path})")

        self.use_gpu_delegate = self._should_use_gpu_delegate()
        if self.use_gpu_delegate:
            print("[Detector] Using MediaPipe GPU delegate (Metal on macOS)")
        else:
            print("[Detector] Using CPU delegate")

        # 1. Pose Landmarker
        self.pose_landmarker = self._build_pose_landmarker(pose_model_path)

        # 2. Face Landmarker
        self.face_landmarker = self._build_face_landmarker(MODEL_PATHS['FACE'])

        # 3. Hand Landmarker
        self.hand_landmarker = self._build_hand_landmarker(MODEL_PATHS['HAND'])

    def _should_use_gpu_delegate(self):
        backend = str(INFERENCE_BACKEND).strip().lower()

        if backend == 'cpu':
            return False

        if backend == 'mps':
            if platform.system() == 'Darwin':
                return True
            print("[Detector] INFERENCE_BACKEND='mps' requested on non-macOS. Falling back to CPU")
            return False

        if backend == 'gpu':
            return platform.system() == 'Darwin'

        if backend == 'auto':
            return bool(PREFER_GPU_DELEGATE and platform.system() == 'Darwin')

        print(f"[Detector] Unknown INFERENCE_BACKEND='{INFERENCE_BACKEND}'. Using auto mode")
        return bool(PREFER_GPU_DELEGATE and platform.system() == 'Darwin')

    def _create_base_options(self, model_path):
        if self.use_gpu_delegate:
            try:
                return python.BaseOptions(
                    model_asset_path=model_path,
                    delegate=python.BaseOptions.Delegate.GPU
                )
            except Exception as error:
                print(f"[Detector] GPU delegate unavailable ({error}), falling back to CPU")
                self.use_gpu_delegate = False

        return python.BaseOptions(model_asset_path=model_path)

    def _build_pose_landmarker(self, model_path):
        try:
            base_opts = self._create_base_options(model_path)
            pose_opts = vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=NUM_POSES,
                min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            return vision.PoseLandmarker.create_from_options(pose_opts)
        except Exception as error:
            if self.use_gpu_delegate:
                print(f"[Detector] Pose GPU init failed ({error}), retrying on CPU")
                self.use_gpu_delegate = False
                return self._build_pose_landmarker(model_path)
            raise

    def _build_face_landmarker(self, model_path):
        try:
            base_opts = self._create_base_options(model_path)
            face_opts = vision.FaceLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=NUM_FACES,
                min_face_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_face_presence_confidence=MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            return vision.FaceLandmarker.create_from_options(face_opts)
        except Exception as error:
            if self.use_gpu_delegate:
                print(f"[Detector] Face GPU init failed ({error}), retrying on CPU")
                self.use_gpu_delegate = False
                return self._build_face_landmarker(model_path)
            raise

    def _build_hand_landmarker(self, model_path):
        try:
            base_opts = self._create_base_options(model_path)
            hand_opts = vision.HandLandmarkerOptions(
                base_options=base_opts,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=NUM_HANDS,
                min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
                min_hand_presence_confidence=MIN_TRACKING_CONFIDENCE,
                min_tracking_confidence=MIN_TRACKING_CONFIDENCE
            )
            return vision.HandLandmarker.create_from_options(hand_opts)
        except Exception as error:
            if self.use_gpu_delegate:
                print(f"[Detector] Hand GPU init failed ({error}), retrying on CPU")
                self.use_gpu_delegate = False
                return self._build_hand_landmarker(model_path)
            raise
        
    def reload(self, model_type="FULL"):
        """Reload Pose Model with specific complexity (LITE, FULL, HEAVY)."""
        if model_type not in MODEL_PATHS:
            print(f"Unknown model type: {model_type}")
            return

        print(f"Reloading Pose Model: {model_type}")
        if hasattr(self, 'pose_landmarker'):
            self.pose_landmarker.close()
            
        pose_model_path = MODEL_PATHS[model_type]
        self.pose_landmarker = self._build_pose_landmarker(pose_model_path)
        
    def set_imaging_params(self, gamma=1.0, face_exposure=False, enable_face=True, enable_hand=True, enable_roi=None):
        """Update runtime imaging parameters."""
        self.gamma = gamma
        self.face_exposure = face_exposure
        self.enable_face = enable_face
        self.enable_hand = enable_hand
        if enable_roi is not None:
            self.enable_roi_cropping = enable_roi

    def _apply_gamma(self, image, gamma=1.0):
        if gamma == 1.0: return image
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _apply_face_exposure(self, image):
        """
        Normalize brightness based on previous face detection.
        Target brightness from config.
        """
        if self.last_face_rect is None: return image
        
        x, y, w, h = self.last_face_rect
        # Valid bounds check
        H, W = image.shape[:2]
        x = max(0, min(x, W-1))
        y = max(0, min(y, H-1))
        w = max(1, min(w, W-x))
        h = max(1, min(h, H-y))
        
        roi = image[y:y+h, x:x+w]
        if roi.size == 0: return image
        
        # Calculate mean V (Brightness)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_mean = np.mean(hsv_roi[:,:,2])
        
        if v_mean < FACE_EXPOSURE_MIN_BRIGHTNESS:
            v_mean = FACE_EXPOSURE_MIN_BRIGHTNESS
        
        factor = FACE_EXPOSURE_TARGET / v_mean
        factor = min(factor, FACE_EXPOSURE_MAX_GAIN)
        factor = max(factor, FACE_EXPOSURE_MIN_GAIN)
        
        # Apply gain
        return cv2.convertScaleAbs(image, alpha=factor, beta=0)

    def process(self, frame):
        """
        Process a frame and return pose, face, and hand results.
        Expects BGR frame.
        """
        import numpy as np
        
        original_frame = frame.copy()
        H_orig, W_orig = frame.shape[:2]
        
        # --- ROI CROPPING (High Quality) ---
        # We crop from the ORIGINAL frame to preserve maximum detail.
        roi_x, roi_y, roi_w, roi_h = 0, 0, W_orig, H_orig # Defaults
        roi_active = False
        
        # Determine effective frame to process
        processing_frame = frame
        
        if self.enable_roi_cropping and self.last_roi is not None:
            x, y, w, h = self.last_roi
            
            # ROI Coords are in W_orig, H_orig scale
            # Expand ROI
            expand = int(max(w, h) * self.roi_expansion)
            x = max(0, x - expand)
            y = max(0, y - expand)
            w = min(W_orig - x, w + 2 * expand)
            h = min(H_orig - y, h + 2 * expand)
            
            # Ensure minimum size
            if w >= self.roi_min_size and h >= self.roi_min_size:
                # Crop from ORIGINAL frame (Best Resolution)
                roi_frame = original_frame[y:y+h, x:x+w]
                
                # Resize to target size for inference
                # e.g. Crop 500x500 -> Resize 640x640 (Upscale or Downscale)
                max_dim = max(w, h)
                scale_to_target = self.roi_target_size / max_dim
                
                new_w = int(w * scale_to_target)
                new_h = int(h * scale_to_target)
                processing_frame = cv2.resize(roi_frame, (new_w, new_h))
                
                # Store EXACT ROI params for reprojection formula
                roi_x, roi_y = x, y
                roi_w, roi_h = w, h
                roi_active = True
            else:
                # ROI invalid/too small? Fallback to full frame
                processing_frame = frame
                # Standard Resize if full frame is massive
                if W_orig > MAX_FRAME_WIDTH:
                   scale = MAX_FRAME_WIDTH / W_orig
                   processing_frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(H_orig * scale)))
                   # In fallback, logic is normal full frame.
                   # ROI params effectively cover the whole (or resized) frame? 
                   # Actually, if we resize, the implicit "ROI" is the resized frame.
                   # But to keep it simple: If fallback, just use standard path.
                   pass
        else:
            # 1. Resize for consistency (if too large)
            if W_orig > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / W_orig
                processing_frame = cv2.resize(frame, (MAX_FRAME_WIDTH, int(H_orig * scale)))
                
        # Update frame reference for filters
        frame = processing_frame
            
        # 2. Gamma Correction
        if self.gamma != 1.0:
            frame = self._apply_gamma(frame, self.gamma)
            
        # 3. Face-Guided Exposure
        if self.face_exposure:
            frame = self._apply_face_exposure(frame)
            
        # 4. CLAHE
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # ---------------------
        
        # Convert frame for MediaPipe Image
        if self.use_gpu_delegate:
            # GPU path on macOS is more stable with SRGBA input
            frame_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGBA, data=frame_rgba)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Calculate Timestamp in Milliseconds
        timestamp_ms = int(time.time() * 1000)
        
        # Monotonicity Check
        if timestamp_ms <= self.last_timestamp_ms:
            timestamp_ms = self.last_timestamp_ms + 1
        self.last_timestamp_ms = timestamp_ms
        
        # Run Detectors
        pose_result = self.pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # --- REPROJECT LANDMARKS TO ORIGINAL FRAME ---
        if pose_result and pose_result.pose_landmarks:
            plm = pose_result.pose_landmarks[0]
            
            # Reproject landmarks in-place using User's Formula
            if roi_active:
                for lm in plm:
                    # User Formula: full_x = roi_x + lm.x * roi_w
                    px_full_x = roi_x + lm.x * roi_w
                    px_full_y = roi_y + lm.y * roi_h
                    
                    # Normalize back to [0,1] of ORIGINAL frame
                    lm.x = px_full_x / W_orig
                    lm.y = px_full_y / H_orig
            else:
                 # Standard logic (if simple resize happens)
                 # If we resized the input frame, we just need to confirm aspect ratio.
                 # MP outputs normalized [0,1].
                 # If we resized 1920x1080 -> 960x540, normalized x=0.5 is still x=0.5.
                 # So NO REPROJECTION needed for standard resize 
                 # UNLESS aspect ratio changed (which we avoided).
                 pass
            
            # Now lm.x, lm.y are normalized to ORIGINAL frame.
            
            # --- UPDATE NEXT ROI ---
            xs = [lm.x * W_orig for lm in plm]
            ys = [lm.y * H_orig for lm in plm]
            
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            new_roi = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Smooth ROI position
            if self.enable_roi_cropping:
                if self.last_roi is not None:
                    a = self.roi_alpha
                    self.last_roi = (
                        int(a * new_roi[0] + (1-a) * self.last_roi[0]),
                        int(a * new_roi[1] + (1-a) * self.last_roi[1]),
                        int(a * new_roi[2] + (1-a) * self.last_roi[2]),
                        int(a * new_roi[3] + (1-a) * self.last_roi[3])
                    )
                else:
                    self.last_roi = new_roi
            else:
                 self.last_roi = None

        elif self.enable_roi_cropping:
             # Lost tracking? Reset.
             self.last_roi = None 

        
        face_result = None
        if self.enable_face:
            face_result = self.face_landmarker.detect_for_video(mp_image, timestamp_ms)
            # Update Face Rect for Exposure (from first face)
            if face_result and face_result.face_landmarks:
                 # Calculate bounding box
                 flm = face_result.face_landmarks[0]
                 xs = [lm.x for lm in flm]
                 ys = [lm.y for lm in flm]
                 H, W = original_frame.shape[:2]
                 x_min, x_max = min(xs) * W, max(xs) * W
                 y_min, y_max = min(ys) * H, max(ys) * H
                 self.last_face_rect = (int(x_min), int(y_min), int(x_max-x_min), int(y_max-y_min))
        
        hand_result = None
        if self.enable_hand:
            hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        return {
            'pose': pose_result,
            'face': face_result,
            'hand': hand_result,
            'roi': self.last_roi if self.enable_roi_cropping else None
        }


