import os
import io
import re
import time
import base64
import datetime
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ppe_detector import PPEDetector
from db_logger import (
    get_recent_violations, 
    resolve_violation_in_db, 
    get_analytics_summary,
    log_customer_feedback,
    get_customer_feedbacks,
    SNAPSHOTS_DIR
)
from face_recognizer import DEFAULT_KNOWN_DIR

app = FastAPI(title="SafeWatch AI - Enterprise Industrial Safety Command")

# Security payload limits
MAX_VIDEO_SIZE = 150 * 1024 * 1024   # 150 MB
MAX_IMAGE_SIZE = 15 * 1024 * 1024    # 15 MB
MAX_FRAME_SIZE = 5 * 1024 * 1024     # 5 MB


def sanitize_filename(filename: str) -> str:
    """Strips path traversal artifacts and keeps safe alphanumeric characters."""
    clean = Path(filename).name
    clean = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', clean)
    return clean[:128]


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """
    Enterprise Security Headers Middleware:
    Prevents clickjacking, MIME sniffing, cross-site scripting (XSS),
    and enforces HSTS over HTTPS Cloudflare Tunnels.
    """
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # HSTS when served through secure HTTPS proxy/tunnel
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
    # Content-Security-Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.tailwindcss.com https://cdnjs.cloudflare.com data: blob:; "
        "img-src 'self' data: blob: https:; "
        "media-src 'self' data: blob:; "
        "connect-src 'self' blob: data:;"
    )
    return response

# Directory setup
BASE_DIR = Path(r"C:\ppe detection")
KNOWN_FACES_DIR = Path(DEFAULT_KNOWN_DIR)
FOOTAGE_DIR = BASE_DIR / "custom_footage"
TEMPLATES_DIR = BASE_DIR / "templates"

KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
FOOTAGE_DIR.mkdir(parents=True, exist_ok=True)

# Mount static asset routes
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOTS_DIR)), name="snapshots")
app.mount("/known_faces", StaticFiles(directory=str(KNOWN_FACES_DIR)), name="known_faces")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Single global instance of PPEDetector
detector = PPEDetector()

# Global state for active streaming channel
active_stream_state = {
    "source": "footage",       # "footage" or "cam"
    "active_video": None,      # filename in FOOTAGE_DIR
    "last_inference_ms": 28.5,
    "last_fps": 35.0
}


def get_available_footages():
    """Returns list of video files in custom_footage directory."""
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [f.name for f in FOOTAGE_DIR.iterdir() if f.suffix.lower() in valid_exts]


def make_standby_frame(w=1280, h=720, message="AWAITING SITE FOOTAGE"):
    """
    Renders an authentic, high-tech industrial CCTV standby screen.
    Eliminates stock / foreign face footage completely.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark obsidian slate background with grid pattern
    img[:] = (20, 16, 12)
    grid_spacing = 64
    for x in range(0, w, grid_spacing):
        cv2.line(img, (x, 0), (x, h), (30, 24, 20), 1)
    for y in range(0, h, grid_spacing):
        cv2.line(img, (0, y), (w, y), (30, 24, 20), 1)

    # High-tech corner HUD brackets
    pad = 40
    length = 60
    # Top-left
    cv2.line(img, (pad, pad), (pad + length, pad), (0, 210, 255), 2)
    cv2.line(img, (pad, pad), (pad, pad + length), (0, 210, 255), 2)
    # Top-right
    cv2.line(img, (w - pad, pad), (w - pad - length, pad), (0, 210, 255), 2)
    cv2.line(img, (w - pad, pad), (w - pad, pad + length), (0, 210, 255), 2)
    # Bottom-left
    cv2.line(img, (pad, h - pad), (pad + length, h - pad), (0, 210, 255), 2)
    cv2.line(img, (pad, h - pad), (pad, h - pad - length), (0, 210, 255), 2)
    # Bottom-right
    cv2.line(img, (w - pad, h - pad), (w - pad - length, h - pad), (0, 210, 255), 2)
    cv2.line(img, (w - pad, h - pad), (w - pad, h - pad - length), (0, 210, 255), 2)

    # Top Status HUD
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cv2.circle(img, (pad + 25, pad + 30), 6, (0, 220, 255), -1)
    cv2.putText(img, f"REC [LIVE CCTV] {now_str}", (pad + 42, pad + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)

    cv2.putText(img, "SAFEWATCH AI // SITE CAMERA STANDBY", (w // 2 - 270, h // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(img, message, (w // 2 - 320, h // 2 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 180, 255), 2)
    cv2.putText(img, "Drop industrial site .MP4 videos into custom_footage/ or upload via dashboard", 
                (w // 2 - 370, h // 2 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 165, 180), 1)

    # Bottom hardware telemetry
    gpu_label = "GPU: NVIDIA GeForce RTX 3050 (6GB CUDA) | YOLOX-S FP16 | SFace (Aman & Vishesh)"
    cv2.putText(img, gpu_label, (pad + 20, h - pad - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 140, 160), 1)

    return img


def generate_frames(source="footage", video_file=None):
    """
    Video streaming generator for MJPEG multipart responses.
    Streams either:
      1. Real uploaded industrial site videos ('footage')
      2. Live local laptop webcam ('cam')
      3. Standby test screen when awaiting footage
    """
    if source == "cam":
        # Windows DirectShow backend for zero buffer latency
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            # If camera 0 cannot be opened, stream standby graphic
            while True:
                standby = make_standby_frame(message="CAMERA 0 UNAVAILABLE // SWITCH TO FOOTAGE")
                ret_enc, buffer = cv2.imencode('.jpg', standby, [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                while True:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.01)
                        continue

                    annotated, _ = detector.detect(frame, camera_id="Cam_01_Webcam", save_snapshots=True)
                    infer_ms = (time.time() - t0) * 1000
                    active_stream_state["last_inference_ms"] = round(infer_ms, 1)
                    active_stream_state["last_fps"] = round(1000.0 / max(infer_ms, 1.0), 1)

                    ret_enc, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ret_enc:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    elapsed = time.time() - t0
                    sleep_time = max(0.001, 0.033 - elapsed)
                    time.sleep(sleep_time)
            finally:
                cap.release()

    elif source == "footage":
        footages = get_available_footages()
        if not footages:
            # Standby screen awaiting user footage
            while True:
                standby = make_standby_frame()
                ret_enc, buffer = cv2.imencode('.jpg', standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.1)
        else:
            target_file = video_file if (video_file and video_file in footages) else footages[0]
            video_path = str(FOOTAGE_DIR / target_file)
            cap = cv2.VideoCapture(video_path)

            try:
                while True:
                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        # Loop video back to start seamlessly
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret:
                            time.sleep(0.1)
                            continue

                    # Downscale 4K / large frames for smooth 30+ FPS playback
                    h, w, _ = frame.shape
                    if max(h, w) > 1280:
                        scale = 1280 / max(h, w)
                        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                    annotated, _ = detector.detect(frame, camera_id=f"Zone_{target_file}", save_snapshots=True)
                    infer_ms = (time.time() - t0) * 1000
                    active_stream_state["last_inference_ms"] = round(infer_ms, 1)
                    active_stream_state["last_fps"] = round(1000.0 / max(infer_ms, 1.0), 1)

                    ret_enc, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
                    if ret_enc:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
                    # Control playback speed to ~25-30 FPS
                    elapsed = time.time() - t0
                    sleep_time = max(0.01, 0.033 - elapsed)
                    time.sleep(sleep_time)
            finally:
                cap.release()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Renders the improvised command center dashboard."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/video_feed")
async def video_feed(source: str = "footage", video: Optional[str] = None):
    """MJPEG video feed endpoint."""
    return StreamingResponse(
        generate_frames(source=source, video_file=video),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/footages")
async def list_footages():
    """Lists available site videos in custom_footage/."""
    footages = get_available_footages()
    return {
        "footages": footages,
        "count": len(footages),
        "active": active_stream_state.get("active_video") or (footages[0] if footages else None)
    }


@app.post("/api/upload_footage")
async def upload_footage(file: UploadFile = File(...)):
    """Uploads industrial site video directly from the dashboard with security guards."""
    valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    safe_name = sanitize_filename(file.filename)
    ext = Path(safe_name).suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(status_code=400, detail="Invalid video format. Supported: .mp4, .avi, .mov, .mkv, .webm")

    content = await file.read()
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size ({MAX_VIDEO_SIZE // (1024*1024)}MB).")

    save_path = FOOTAGE_DIR / safe_name
    with open(save_path, "wb") as f:
        f.write(content)

    active_stream_state["active_video"] = safe_name
    active_stream_state["source"] = "footage"
    print(f"[FOOTAGE UPLOAD] Securely saved industrial video: {safe_name} ({len(content)} bytes)")

    return {
        "success": True,
        "filename": safe_name,
        "message": f"Successfully loaded industrial site footage: {safe_name}"
    }


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns live hardware, model, and inference diagnostics for HUD."""
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    enrolled_names = [w["name"] for w in detector.face_system.list_enrolled()]
    return {
        "gpu": device_name,
        "cuda_active": torch.cuda.is_available(),
        "model": "YOLOX-Small (5-Class PPE)",
        "face_engine": "OpenCV YuNet + SFace (128-d)",
        "enrolled_workers": enrolled_names,
        "inference_ms": active_stream_state["last_inference_ms"],
        "fps": active_stream_state["last_fps"]
    }


@app.get("/api/violations")
async def get_violations(limit: int = 50, resolved: Optional[int] = None):
    """Returns recent violations from SQLite / Atlas."""
    filter_val = None if resolved is None else bool(resolved)
    return get_recent_violations(limit=limit, resolved_filter=filter_val)


@app.post("/api/resolve/{violation_id}")
async def resolve_violation(violation_id: int):
    """Marks a violation incident as resolved."""
    ok = resolve_violation_in_db(violation_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Failed to resolve incident")
    return {"status": "success", "resolved_id": violation_id}


@app.get("/api/stats")
async def get_stats():
    """Returns real-time KPI metrics."""
    return get_analytics_summary()


@app.get("/api/workers")
async def get_enrolled_workers():
    """Returns list of currently enrolled workers in the face database."""
    return detector.face_system.list_enrolled()


@app.post("/api/enroll")
async def enroll_worker(name: str = Form(...), worker_id: str = Form(""), file: UploadFile = File(...)):
    """
    Enrolls a worker's face into SFace & YuNet feature database.
    Supports registering ANY worker on site for face verification.
    """
    clean_name = sanitize_filename(name.strip())
    clean_worker_id = sanitize_filename(worker_id.strip()) if worker_id else ""
    if not clean_name:
        raise HTTPException(status_code=400, detail="Worker name contains invalid characters.")

    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail=f"Image exceeds maximum size ({MAX_IMAGE_SIZE // (1024*1024)}MB).")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    ok, msg = detector.face_system.enroll_face(img, name=clean_name, worker_id=clean_worker_id)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)

    return {
        "success": True, 
        "message": msg,
        "workers": detector.face_system.list_enrolled()
    }


@app.post("/api/workers/delete/{filename}")
@app.delete("/api/workers/{filename}")
async def delete_worker(filename: str):
    """Unregisters a worker and removes their face embedding from the database."""
    safe_filename = sanitize_filename(filename)
    ok, msg = detector.face_system.delete_face(safe_filename)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {
        "success": True, 
        "message": msg,
        "workers": detector.face_system.list_enrolled()
    }


@app.post("/api/audit")
async def audit_image(file: UploadFile = File(...)):
    """Runs on-demand PPE detection and face matching on an uploaded photo."""
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail=f"Audit image exceeds maximum size ({MAX_IMAGE_SIZE // (1024*1024)}MB).")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    annotated, summary = detector.detect(frame, camera_id="Audit_Upload", save_snapshots=True)
    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64_str = base64.b64encode(buffer).decode('utf-8')

    return {
        "summary": summary,
        "annotated_image": f"data:image/jpeg;base64,{b64_str}"
    }


class ClientFrameRequest(BaseModel):
    frame: str  # Data URL or base64 JPEG
    camera_id: Optional[str] = "Device_Cam"


@app.post("/api/process_client_frame")
async def process_client_frame(data: ClientFrameRequest):
    """
    Ingests live video frames from client devices (Smartphones, Tablets, External USB cameras).
    Runs real-time RTX 3050 CUDA YOLOX + SFace detection and streams back the annotated frame & telemetry.
    """
    t0 = time.time()
    raw_b64 = data.frame
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(raw_b64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 payload.")

    if len(img_bytes) > MAX_FRAME_SIZE:
        raise HTTPException(status_code=413, detail="Frame exceeds 5MB payload limit.")

    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Unable to decode frame.")

    # Downscale high-res frames to max 1280px for guaranteed 30+ FPS low latency
    h, w = frame.shape[:2]
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    cam_id = sanitize_filename(data.camera_id or "Device_Cam")
    annotated, summary = detector.detect(frame, camera_id=cam_id, save_snapshots=True)

    infer_ms = (time.time() - t0) * 1000
    fps = round(1000.0 / max(infer_ms, 1.0), 1)
    active_stream_state["last_inference_ms"] = round(infer_ms, 1)
    active_stream_state["last_fps"] = fps

    ret_enc, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 78])
    if not ret_enc:
        raise HTTPException(status_code=500, detail="Frame compression failure.")

    out_b64 = base64.b64encode(buffer).decode('utf-8')
    return {
        "summary": summary,
        "annotated_frame": f"data:image/jpeg;base64,{out_b64}",
        "inference_ms": round(infer_ms, 1),
        "fps": fps
    }


class CustomerFeedbackRequest(BaseModel):
    evaluator_name: str
    organization: Optional[str] = "Demo Evaluator"
    rating: int = 5
    category: Optional[str] = "General"
    comments: str


@app.post("/api/feedback")
async def submit_feedback(data: CustomerFeedbackRequest, request: Request):
    """Saves customer demonstration feedback and evaluation scores."""
    client_ip = request.client.host if request.client else "unknown"
    evaluator = sanitize_filename(data.evaluator_name.strip()) or "Anonymous"
    fid = log_customer_feedback(
        evaluator_name=evaluator[:64],
        organization=(data.organization or "Client")[:64],
        rating=max(1, min(5, data.rating)),
        category=(data.category or "General")[:32],
        comments=data.comments[:1000],
        ip_address=client_ip
    )
    return {
        "success": True, 
        "feedback_id": fid, 
        "message": "Thank you! Your evaluation and feedback has been securely stored."
    }


@app.get("/api/feedback")
async def list_feedback():
    """Returns past customer feedback records."""
    return get_customer_feedbacks(limit=50)


if __name__ == "__main__":
    import uvicorn
    print("[SAFEWATCH AI] Starting server at http://localhost:8000 ...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
