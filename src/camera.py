import cv2
import threading
import time
from config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS

class Camera:
    def __init__(self, camera_id=CAMERA_ID):
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        # Configure Camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        
        self.grabbed = False
        self.frame = None
        self.running = False
        self.thread = None
        
        if self.cap.isOpened():
            self.grabbed, self.frame = self.cap.read()
            self.running = True
            self.start_thread()
        else:
            raise RuntimeError(f"Could not open camera with ID {camera_id}")
            
    def start_thread(self):
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while self.running:
            if self.cap.isOpened():
                grabbed, frame = self.cap.read()
                if grabbed:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    self.running = False
            else:
                self.running = False
            time.sleep(0.005) # Slight delay to yield CPU

    def read(self):
        """Return the most recent frame."""
        return self.frame if self.grabbed else None
        
    def release(self):
        """Release the camera resource."""
        self.running = False
        if self.thread:
            self.thread.join()
        if self.cap.isOpened():
            self.cap.release()
            
    def is_opened(self):
        return self.cap.isOpened()
