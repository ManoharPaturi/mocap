import cv2
import sys
from src.camera import Camera
from src.detector import MocapDetector
from src.visualizer import Visualizer
from config import DRAW_LANDMARKS

def main():
    print("Starting MoCap Application...")
    print("Press 'q' to quit.")
    
    try:
        # Initialize modules
        cam = Camera()
        detector = MocapDetector()
        viz = Visualizer()
        
        while cam.is_opened():
            frame = cam.read()
            if frame is None:
                print("Failed to read frame.")
                break
                
            # Detection
            results = detector.process(frame)
            
            # Visualization
            if DRAW_LANDMARKS:
                frame = viz.draw_landmarks(frame, results)
                
            frame = viz.draw_fps(frame)
            
            # Display
            cv2.imshow('MoCap - vs3', frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        cam.release()
        cv2.destroyAllWindows()
        print("Application closed.")

if __name__ == "__main__":
    main()
