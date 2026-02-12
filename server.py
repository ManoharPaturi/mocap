from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from src.streamer import VideoStreamer
from src.report_generator import ReportGenerator
from src.visualizer_3d import Visualizer3D
import uvicorn
import threading

app = FastAPI()

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Streamer Instance
streamer = VideoStreamer()

@app.on_event("startup")
async def startup_event():
    print("Starting Video Streamer...")
    streamer.start()

@app.on_event("shutdown")
async def shutdown_event():
    print("Stopping Video Streamer...")
    streamer.stop()

@app.get("/video_feed")
async def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return StreamingResponse(streamer.generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    return {
        "status": "running", 
        "streaming": streamer.running,
        "recording": streamer.db.running
    }

@app.get("/metrics")
async def get_metrics():
    """Return latest calculated body metrics."""
    return streamer.latest_metrics

@app.post("/record/start")
async def start_recording():
    session_id = streamer.db.start_recording()
    return {"status": "started", "session_id": session_id}

@app.post("/record/stop")
async def stop_recording():
    streamer.db.stop_recording()
    return {"status": "stopped"}

@app.get("/record/export")
async def export_data():
    csv_content = streamer.db.export_latest_session_csv()
    if not csv_content:
        return {"error": "No recordings found"}
    
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=mocap_export.csv"}
    )

# --- AI Features ---
reporter = ReportGenerator(streamer.db)
viz_3d = Visualizer3D(streamer.db)

@app.post("/report/generate")
async def generate_report():
    try:
        path = reporter.generate_report()
        if path:
            return {"status": "success", "path": path}
        return {"status": "failed", "error": "No data to analyze"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/viz/3d")
async def start_viz():
    try:
        # Run visualization in a separate thread to avoid blocking the server
        threading.Thread(target=viz_3d.plot_latest_session, daemon=True).start()
        return {"status": "success", "message": "3D Visualization launched on server"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
