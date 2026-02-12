import cv2
import asyncio
import json
import uvicorn
import sys
import os

# Allow running as script from vs4 root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from src.camera import Camera
from src.detector import MocapDetector
from src.pose_corrector import PoseCorrector

app = FastAPI()

# Serve Web Client
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, 'web')

app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

@app.get("/")
async def get():
    index_path = os.path.join(WEB_DIR, "index.html")
    with open(index_path, 'r') as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    camera = Camera()
    detector = MocapDetector()
    corrector = PoseCorrector()
    
    print("Client connected. Streaming Mocap Data...")
    
    try:
        while True:
            frame = camera.read()
            if frame is None: break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect
            results = detector.process(frame)
            
            # Correct Physics
            results = corrector.process(results)
            
            # Prepare JSON Payload
            data = {}
            if results.get('pose') and results['pose'].pose_landmarks:
                landmarks = results['pose'].pose_landmarks[0]
                
                # We send the Full Body landmarks (33 points)
                # Client (Three.js) will handle the vector mapping
                msg_landmarks = []
                for lm in landmarks:
                    msg_landmarks.append({'x': lm.x, 'y': lm.y, 'z': lm.z, 'vis': lm.visibility})
                
                data['pose'] = msg_landmarks
            
            await websocket.send_text(json.dumps(data))
            
            # Optional: Sleep to limit FPS if needed, but 0 relies on process speed
            await asyncio.sleep(0.01) 
            
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        camera.release()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
