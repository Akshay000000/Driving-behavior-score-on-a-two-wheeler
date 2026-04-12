"""
Varroc Eureka 3.0 - PS3
FastAPI Backend — WebSocket video stream + REST API
Run: uvicorn backend:app --host 0.0.0.0 --port 8000
"""

import cv2, math, base64, time, threading, json
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# ── Import detection logic from dashcam.py / fixed_cam.py ────────────────────
# We inline the core classes here so backend.py is self-contained.
# Paste your dashcam.py class definitions below (KalmanVelocityEstimator,
# EgoMotionEstimator, FleetContext, IoUTracker, RiderBehavior, HelmetDetector)
# OR do: from dashcam import *

try:
    from dashcam import (KalmanVelocityEstimator, EgoMotionEstimator,
                         FleetContext, IoUTracker, RiderBehavior,
                         HelmetDetector, ByteTrackWrapper,
                         BYTETRACK_AVAILABLE, FRAME_W, FRAME_H,
                         CONF_THRESH, PERSON_CONF, ACTIVE_CLASSES,
                         EgoMotionEstimator, DIR_ARROWS, draw_hud,
                         draw_fleet_bar, draw_ego, draw_leaderboard,
                         draw_footer, bbox_mpp, max_plausible_speed,
                         MAX_CAMERA_SPEED_MPS)
    print("[INFO] Loaded from dashcam.py")
except ImportError as e:
    print(f"[ERROR] Could not import dashcam.py: {e}")
    raise

# ── Global state ─────────────────────────────────────────────────────────────

class SessionState:
    def __init__(self):
        self.running     = False
        self.mode        = "dashcam"   # "dashcam" | "fixed"
        self.video_path  = r"../public/video.mp4"
        self.model_path  = "yolov8n.pt"
        self.frame_data  = None   # latest encoded frame
        self.riders      = {}     # tid -> rider snapshot dict
        self.all_time    = {}     # full session archive
        self.frame_count = 0
        self.fps         = 30.0
        self.lock        = threading.Lock()
        self._thread     = None

STATE = SessionState()

def rider_snapshot(b, tid):
    """Serialise a RiderBehavior to a JSON-safe dict."""
    return {
        "id":            tid,
        "score":         round(b.score, 1),
        "speed_kmh":     round(b.speed_kmh(), 1),
        "direction":     b.direction,
        "helmet":        b.helmet_status,
        "hard_brake":    b.hard_brake_count,
        "aggr_accel":    b.aggr_accel_count,
        "weave":         b.weave_count,
        "tailgate":      b.tailgate_count,
        "sudden_stop":   b.sudden_stop_count,
        "no_helmet":     b.no_helmet_count,
        "over_speed":    round(b.over_speed_sec, 1),
        "reliability":   round(b.reliability, 0),
        "stunt":         b.is_doing_stunt,
        "stationary":    b.is_stationary,
        "rating":        b.score_label(),
        "accel":         round(b.fast_accel, 2),
    }

def detection_loop():
    """Background thread: run detection, update STATE."""
    model   = YOLO(STATE.model_path)
    cap     = cv2.VideoCapture(STATE.video_path)
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    STATE.fps = fps

    if BYTETRACK_AVAILABLE:
        tracker = ByteTrackWrapper(fps)
    else:
        tracker = IoUTracker()

    ego        = EgoMotionEstimator() if STATE.mode == "dashcam" else None
    fleet      = FleetContext()
    helmet_det = HelmetDetector()
    behaviors  = {}
    all_time   = {}
    n          = 0

    while STATE.running:
        ret, frame = cap.read()
        if not ret:
            # Loop video for demo
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue

        n += 1
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edx, edy = ego.update(gray) if ego else (0.0, 0.0)

        results = model(frame, conf=CONF_THRESH, verbose=False)[0]
        det_bikes = []; person_boxes = []
        for b in results.boxes:
            cls  = int(b.cls[0]); box = tuple(map(int, b.xyxy[0]))
            conf = float(b.conf[0])
            if cls in ACTIVE_CLASSES and conf >= CONF_THRESH:
                det_bikes.append((box, cls))
            elif cls == 0 and conf >= PERSON_CONF:
                person_boxes.append(box)

        tracker.update([b for b, _ in det_bikes])
        def get_cls(box):
            for b, c in det_bikes:
                if b == box: return c
            return 3
        active = tracker.active_tracks
        fleet.update(behaviors)

        if behaviors:
            mv  = [float(np.median(list(bv._mpp_hist)))
                   for bv in behaviors.values() if bv._mpp_hist]
            cm  = float(np.median(mv)) if mv else 0.020
            cs  = min(math.hypot(edx, edy) * cm * fps, MAX_CAMERA_SPEED_MPS)
        else:
            cs = 0.0

        for tid, td in active.items():
            if tid not in behaviors:
                behaviors[tid] = RiderBehavior(tid, fps)
            b  = behaviors[tid]; cx, cy = td['cx'], td['cy']
            df = EgoMotionEstimator.depth_factor(cy) if ego else 1.0
            if b._prev_cx is not None:
                rdx = float(cx) - b._prev_cx; rdy = float(cy) - b._prev_cy
            else:
                rdx = rdy = 0.0
            comp_dx = rdx - edx * df; comp_dy = rdy - edy * df
            b.update(cx, cy, td['box'], comp_dx, comp_dy, fleet,
                     cs, get_cls(td['box']))
            all_time[tid] = b

        statuses = helmet_det.update(gray, active, person_boxes)
        for tid, s in statuses.items():
            if tid in behaviors: behaviors[tid].set_helmet(s)

        alive = set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive]:
            del behaviors[k]

        # Draw overlays
        for tid, td in active.items():
            if tid in behaviors: draw_hud(frame, behaviors[tid], td['box'])
        draw_fleet_bar(frame, fleet, n)
        if ego: draw_ego(frame, edx, edy)
        draw_leaderboard(frame, {t: behaviors[t] for t in active if t in behaviors})
        draw_footer(frame, n, fps)

        # Encode frame to JPEG base64
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        b64 = base64.b64encode(buf).decode('utf-8')

        with STATE.lock:
            STATE.frame_data  = b64
            STATE.frame_count = n
            STATE.riders      = {tid: rider_snapshot(behaviors[tid], tid)
                                  for tid in active if tid in behaviors}
            STATE.all_time    = {tid: rider_snapshot(all_time[tid], tid)
                                  for tid in all_time}

        time.sleep(1.0 / 60.0)  # 60fps

    cap.release()

# ── REST endpoints ────────────────────────────────────────────────────────────

@app.post("/api/start")
async def start(body: dict = None):
    if STATE.running:
        return {"status": "already_running"}
    body = body or {}
    STATE.mode       = body.get("mode", "dashcam")
    STATE.video_path = body.get("video_path", STATE.video_path)
    STATE.model_path = body.get("model_path", STATE.model_path)
    STATE.running    = True
    STATE._thread    = threading.Thread(target=detection_loop, daemon=True)
    STATE._thread.start()
    return {"status": "started", "mode": STATE.mode}

@app.post("/api/stop")
async def stop():
    STATE.running = False
    return {"status": "stopped"}

@app.get("/api/session")
async def session():
    with STATE.lock:
        scores = [v['score'] for v in STATE.all_time.values()]
        return {
            "vehicles":   len(STATE.all_time),
            "frames":     STATE.frame_count,
            "duration":   round(STATE.frame_count / max(STATE.fps, 1), 1),
            "riders":     list(STATE.all_time.values()),
            "avg_score":  round(sum(scores)/len(scores), 1) if scores else 0,
            "best":       max(STATE.all_time.values(), key=lambda x:x['score'],
                              default=None),
            "worst":      min(STATE.all_time.values(), key=lambda x:x['score'],
                              default=None),
        }

@app.get("/api/status")
async def status():
    return {"running": STATE.running, "mode": STATE.mode,
            "frames": STATE.frame_count}

# ── WebSocket — streams frame + riders at ~30fps ──────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with STATE.lock:
                frame = STATE.frame_data
                riders = STATE.riders
                n = STATE.frame_count
            if frame:
                await ws.send_text(json.dumps({
                    "frame":  frame,
                    "riders": list(riders.values()),
                    "n":      n,
                }))
            await asyncio.sleep(1/60)
    except WebSocketDisconnect:
        pass