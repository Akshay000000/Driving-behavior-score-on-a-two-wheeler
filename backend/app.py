"""
MotoScore AI — Flask API Backend
Wraps dashcam.py and fixedcam.py for video analysis via HTTP.
Designed for deployment on Render.
"""

import os
import sys
import tempfile
import math
import json
import cv2
import numpy as np
from collections import defaultdict, deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# No file size limit
app.config["MAX_CONTENT_LENGTH"] = None

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
model = None

def get_model():
    global model
    if model is None:
        model = YOLO(MODEL_PATH)
    return model

# ── Shared constants ────────────────────────────────────────────────────────
FRAME_W, FRAME_H = 640, 360
FPS_FALLBACK = 30
ACTIVE_CLASSES = {1, 3}
REAL_WORLD_WIDTHS = {0: 0.50, 1: 0.60, 3: 0.80}
REAL_WORLD_WIDTH_DEFAULT = 0.75
MPP_HISTORY_LEN = 25
MAX_VEHICLE_SPEED_MPS = 35.0
MAX_PLAUSIBLE_CLOSE_MPS = 15.0
CLOSE_BBOX_FRACTION = 0.25
DIR_WINDOW = 10
DIR_ONCOMING_RATE = 0.004
DIR_APPROACH_RATE = 0.001
DIR_RECEDING_RATE = -0.001
HELMET_OVERLAP_IOU = 0.20
HELMET_HEAD_FRACTION = 0.32
HELMET_MIN_AREA_FRAC = 0.08
TAILGATE_AREA_THRESH = 0.09
TAILGATE_GROWTH_RATE = 0.002
SUDDEN_STOP_SPEED_MPS = 1.2
SUDDEN_STOP_FRAMES = 6
SPEED_LIMIT_MPS = 13.9
FLEET_K = 0.6
W_HARD_BRAKE = 5; W_AGGR_ACCEL = 3; W_WEAVE = 2
W_TAILGATE = 4; W_SUDDEN_STOP = 5; W_OVER_SPEED = 0.15
SCORE_EMA_ALPHA = 0.04; COLOR_EMA_ALPHA = 0.02
IOU_MATCH_THRESH_BASE = 0.15
MAX_DISAPPEARED = 45
EVENT_COOLDOWN = 18
DIR_ONCOMING = "ONCOMING"; DIR_APPROACH = "APPROACH"
DIR_RECEDING = "RECEDING"; DIR_ALONGSIDE = "ALONGSIDE"; DIR_UNKNOWN = "?"

# ── Mode-specific configs ───────────────────────────────────────────────────
MODES = {
    "dashcam": {
        "conf_thresh": 0.28,
        "person_conf": 0.32,
        "iou_match_thresh": 0.15,
        "warmup_frames": 15,
        "stationary_disp_px": 3.0,
        "stationary_area_var": 0.0008,
        "stationary_window": 12,
        "helmet_confirm_frames": 30,
        "w_no_helmet": 8,
        "base_hard_brake_mps2": 2.8,
        "base_aggr_accel_mps2": 2.2,
        "lane_weave_px": 18,
        "max_camera_speed_mps": 25.0,
        "ego_compensation": True,
    },
    "fixed": {
        "conf_thresh": 0.35,
        "person_conf": 0.35,
        "iou_match_thresh": 0.20,
        "warmup_frames": 8,
        "stationary_disp_px": 2.0,
        "stationary_area_var": 0.0006,
        "stationary_window": 10,
        "helmet_confirm_frames": 15,
        "w_no_helmet": 10,
        "base_hard_brake_mps2": 2.5,
        "base_aggr_accel_mps2": 2.0,
        "lane_weave_px": 14,
        "max_camera_speed_mps": 0.0,
        "ego_compensation": False,
    },
}

# ── Utility functions ───────────────────────────────────────────────────────
def compute_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    return inter / float((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def bbox_mpp(bw, cls):
    return REAL_WORLD_WIDTHS.get(cls, REAL_WORLD_WIDTH_DEFAULT) / max(bw, 5.0)

def max_plausible_speed(bw):
    frac = bw / FRAME_W
    if frac >= CLOSE_BBOX_FRACTION:
        t = min((frac - CLOSE_BBOX_FRACTION) / (1.0 - CLOSE_BBOX_FRACTION), 1.0)
        return MAX_PLAUSIBLE_CLOSE_MPS * (1.0 - 0.85 * t)
    return MAX_VEHICLE_SPEED_MPS

# ── Kalman (CA) ─────────────────────────────────────────────────────────────
class KalmanVelocityEstimator:
    def __init__(self, process_noise=0.8, measure_noise=28.0):
        self.x = None; self.P = None; self.initialized = False
        dt = 1.0
        self.F = np.array([
            [1,0,dt,0,0.5*dt**2,0],[0,1,0,dt,0,0.5*dt**2],
            [0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]
        ], dtype=float)
        self.H = np.zeros((2,6)); self.H[0,0] = 1; self.H[1,1] = 1
        self.Q = np.diag([process_noise*0.25]*2 + [process_noise*0.5]*2 + [process_noise]*2)
        self.R = np.diag([measure_noise, measure_noise])

    def update(self, cx, cy):
        z = np.array([[cx],[cy]], dtype=float)
        if not self.initialized:
            self.x = np.array([[cx],[cy],[0],[0],[0],[0]], dtype=float)
            self.P = np.eye(6) * 200.0; self.initialized = True
            return cx, cy, 0.0, 0.0
        x_p = self.F @ self.x; P_p = self.F @ self.P @ self.F.T + self.Q
        S = self.H @ P_p @ self.H.T + self.R
        K = P_p @ self.H.T @ np.linalg.inv(S)
        self.x = x_p + K @ (z - self.H @ x_p)
        self.P = (np.eye(6) - K @ self.H) @ P_p
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

# ── Ego-Motion (dashcam only) ──────────────────────────────────────────────
OF_MAX_CORNERS = 200; OF_QUALITY = 0.01; OF_MIN_DIST = 6; OF_WIN_SIZE = (21, 21)

class EgoMotionEstimator:
    def __init__(self):
        self._prev_gray = None; self._prev_pts = None
        self.ego_dx = 0.0; self.ego_dy = 0.0

    @staticmethod
    def depth_factor(cy): return 0.25 + 0.75 * (cy / FRAME_H)

    def _bg_mask(self, h, w):
        mask = np.ones((h, w), np.uint8) * 255
        cv2.rectangle(mask, (w//5, h//2), (4*w//5, h), 0, -1)
        return mask

    def update(self, gray):
        h, w = gray.shape
        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) < 10:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=OF_MAX_CORNERS, qualityLevel=OF_QUALITY,
                minDistance=OF_MIN_DIST, mask=self._bg_mask(h, w))
            self._prev_gray = gray.copy()
            return 0.0, 0.0
        npts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=OF_WIN_SIZE, maxLevel=3)
        good = st.ravel() == 1
        if good.sum() < 5:
            self._prev_pts = None; self._prev_gray = gray.copy()
            return self.ego_dx, self.ego_dy
        flow = npts[good] - self._prev_pts[good]
        dx_v = flow[:, 0, 0]; dy_v = flow[:, 0, 1]
        mdx = float(np.median(dx_v)); mdy = float(np.median(dy_v))
        keep = np.abs(dx_v - mdx) < 6
        redx = float(np.mean(dx_v[keep])) if keep.any() else mdx
        redy = float(np.mean(dy_v[keep])) if keep.any() else mdy
        if math.hypot(redx, redy) < 30.0:
            self.ego_dx = 0.7 * redx + 0.3 * self.ego_dx
            self.ego_dy = 0.7 * redy + 0.3 * self.ego_dy
        self._prev_pts = npts[good].reshape(-1, 1, 2)
        self._prev_gray = gray.copy()
        if len(self._prev_pts) < 25: self._prev_pts = None
        return self.ego_dx, self.ego_dy

# ── Fleet Context ───────────────────────────────────────────────────────────
class FleetContext:
    def __init__(self, cfg):
        self.cfg = cfg
        self.fleet_speed_mps = 0.0; self.fleet_accel_mps2 = 0.0; self.n_vehicles = 0
        self._fleet_mpp = 0.012

    def update(self, behaviors):
        warmup = self.cfg["warmup_frames"]
        speeds = [b.speed_mps for b in behaviors.values() if b._frame_n >= warmup]
        accels = [b.fast_accel for b in behaviors.values() if b._frame_n >= warmup]
        self.n_vehicles = len(speeds)
        self.fleet_speed_mps = float(np.median(speeds)) if speeds else 0.0
        self.fleet_accel_mps2 = float(np.clip(float(np.median(accels)) if accels else 0.0, -4.0, 4.0))
        all_mpp = [float(np.median(list(bv._mpp_hist))) for bv in behaviors.values()
                   if bv._mpp_hist and 0.006 <= float(np.median(list(bv._mpp_hist))) <= 0.030]
        self._fleet_mpp = float(np.median(all_mpp)) if all_mpp else 0.012

    def effective_brake_thr(self):
        return self.cfg["base_hard_brake_mps2"] + FLEET_K * max(0.0, -self.fleet_accel_mps2)
    def effective_accel_thr(self):
        return self.cfg["base_aggr_accel_mps2"] + FLEET_K * max(0.0, self.fleet_accel_mps2)
    def fleet_mpp(self): return self._fleet_mpp

# ── Helmet Detector ─────────────────────────────────────────────────────────
class HelmetDetector:
    def __init__(self, confirm_frames):
        self._history = defaultdict(lambda: deque(maxlen=confirm_frames))
        self._confirm = confirm_frames
        self.status = {}

    def update(self, gray, tracks, person_boxes):
        for tid, td in tracks.items():
            best_iou, best_pb = 0.0, None
            for pb in person_boxes:
                iou = compute_iou(td['box'], pb)
                if iou > best_iou: best_iou, best_pb = iou, pb
            if best_pb is None or best_iou < HELMET_OVERLAP_IOU:
                self.status[tid] = self.status.get(tid, "?"); continue
            px1, py1, px2, py2 = best_pb; p_h = max(py2 - py1, 1)
            h_y1 = max(py1, 0); h_y2 = int(py1 + p_h * HELMET_HEAD_FRACTION)
            h_x1 = max(px1, 0); h_x2 = min(px2, gray.shape[1])
            if h_y2 <= h_y1 or h_x2 <= h_x1: continue
            crop = gray[h_y1:h_y2, h_x1:h_x2]
            if crop.size < 50: continue
            _, thresh = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            blob = max((cv2.contourArea(c) for c in cnts), default=0)
            self._history[tid].append(blob / max(crop.shape[0] * crop.shape[1], 1) >= HELMET_MIN_AREA_FRAC)
            h = self._history[tid]
            threshold = 0.45 if self._confirm == 30 else 0.55
            if len(h) >= self._confirm // 2:
                self.status[tid] = "HELMET" if sum(h) / len(h) >= threshold else "NO_HELMET"
            else:
                self.status[tid] = "?"
        return self.status

# ── IoU Tracker ─────────────────────────────────────────────────────────────
class IoUTracker:
    def __init__(self, iou_thresh):
        self.next_id = 0; self.tracks = {}; self.disappeared = defaultdict(int)
        self._iou_thresh = iou_thresh

    @staticmethod
    def _cen(box): return (box[0]+box[2])//2, (box[1]+box[3])//2

    def _register(self, box):
        cx, cy = self._cen(box)
        self.tracks[self.next_id] = {'box': box, 'cx': cx, 'cy': cy}
        self.disappeared[self.next_id] = 0; self.next_id += 1

    def _dereg(self, tid): self.tracks.pop(tid, None); self.disappeared.pop(tid, None)

    @property
    def active_tracks(self):
        return {t: d for t, d in self.tracks.items() if self.disappeared[t] == 0}

    def update(self, det_boxes):
        if not det_boxes:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > MAX_DISAPPEARED: self._dereg(tid)
            return self.tracks
        if not self.tracks:
            for b in det_boxes: self._register(b)
            return self.tracks
        tids = list(self.tracks.keys())
        tboxes = [self.tracks[t]['box'] for t in tids]
        iou_mat = np.array([[compute_iou(db, tb) for tb in tboxes] for db in det_boxes])
        md = set(); mt = set()
        for idx in np.argsort(-iou_mat, axis=None):
            di, ti = divmod(int(idx), len(tboxes))
            if di in md or ti in mt: continue
            if iou_mat[di, ti] < self._iou_thresh: break
            tid = tids[ti]; box = det_boxes[di]; cx, cy = self._cen(box)
            self.tracks[tid] = {'box': box, 'cx': cx, 'cy': cy}
            self.disappeared[tid] = 0; md.add(di); mt.add(ti)
        unm_d = [i for i in range(len(det_boxes)) if i not in md]
        unm_t = [i for i in range(len(tboxes)) if i not in mt]
        mxd = math.hypot(FRAME_W, FRAME_H) / 4
        for di in unm_d:
            dcx, dcy = self._cen(det_boxes[di]); bd, bti = float('inf'), None
            for ti in unm_t:
                d = math.hypot(dcx - self._cen(tboxes[ti])[0], dcy - self._cen(tboxes[ti])[1])
                if d < bd: bd, bti = d, ti
            if bti is not None and bd < mxd:
                tid = tids[bti]; box = det_boxes[di]; cx, cy = self._cen(box)
                self.tracks[tid] = {'box': box, 'cx': cx, 'cy': cy}
                self.disappeared[tid] = 0; unm_t.remove(bti)
            else:
                self._register(det_boxes[di])
        for ti in unm_t:
            tid = tids[ti]; self.disappeared[tid] += 1
            if self.disappeared[tid] > MAX_DISAPPEARED: self._dereg(tid)
        return self.tracks

# ── Rider Behavior ──────────────────────────────────────────────────────────
class RiderBehavior:
    def __init__(self, rider_id, fps, cfg):
        self.rider_id = rider_id; self.fps = fps; self.dt = 1.0/fps; self._frame_n = 0
        self.cfg = cfg
        self._prev_cx = None; self._prev_cy = None
        pn = 0.8 if cfg["ego_compensation"] else 0.6
        mn = 28.0 if cfg["ego_compensation"] else 22.0
        self._kalman = KalmanVelocityEstimator(pn, mn)
        self._mpp_hist = deque(maxlen=MPP_HISTORY_LEN)
        self._spd_hist = deque(maxlen=max(SUDDEN_STOP_FRAMES+2, 12))
        sw = cfg["stationary_window"]
        self._area_hist = deque(maxlen=max(sw, DIR_WINDOW)+2)
        self._lat_hist = deque(maxlen=8); self._ar_hist = deque(maxlen=10)
        self.cls = 3; self.speed_mps = 0.0; self.fast_accel = 0.0
        self.accel_mps2 = 0.0; self._prev_speed = 0.0
        self.direction = DIR_UNKNOWN
        self.helmet_status = "?"; self.no_helmet_count = 0; self._no_helmet_cd = 0
        self.is_stationary = False; self.is_doing_stunt = False
        self.reliability = 0.0
        self.hard_brake_count = 0; self.aggr_accel_count = 0
        self.weave_count = 0; self.tailgate_count = 0
        self.sudden_stop_count = 0; self.over_speed_sec = 0.0
        self._cd = defaultdict(int)
        self.score = 100.0; self._raw = 100.0; self._committed = 100.0

    def _tick_cd(self):
        for k in list(self._cd): self._cd[k] = max(0, self._cd[k] - 1)
        self._no_helmet_cd = max(0, self._no_helmet_cd - 1)

    def _fire(self, key, attr):
        if self._cd[key] == 0:
            setattr(self, attr, getattr(self, attr) + 1)
            self._cd[key] = EVENT_COOLDOWN

    def _check_stationary(self, dx, dy):
        sw = self.cfg["stationary_window"]
        area_stable = (len(self._area_hist) >= sw and
                       float(np.var(list(self._area_hist)[-sw:])) < self.cfg["stationary_area_var"])
        return math.hypot(dx, dy) < self.cfg["stationary_disp_px"] and area_stable

    def _classify_direction(self):
        if len(self._area_hist) < DIR_WINDOW: return DIR_UNKNOWN
        r = list(self._area_hist)[-DIR_WINDOW:]
        s = float(np.polyfit(np.arange(len(r)), r, 1)[0])
        if s > DIR_ONCOMING_RATE: return DIR_ONCOMING
        if s > DIR_APPROACH_RATE: return DIR_APPROACH
        if s < DIR_RECEDING_RATE: return DIR_RECEDING
        return DIR_ALONGSIDE

    def set_helmet(self, status):
        self.helmet_status = status
        if status == "NO_HELMET" and self._no_helmet_cd == 0:
            self.no_helmet_count += 1; self._no_helmet_cd = EVENT_COOLDOWN * 2

    def update(self, cx, cy, box, fleet, comp_dx=0.0, comp_dy=0.0, camera_speed_mps=0.0, cls=3):
        self._frame_n += 1; self._tick_cd(); self.cls = cls
        prev_cx = self._prev_cx if self._prev_cx is not None else float(cx)
        prev_cy = self._prev_cy if self._prev_cy is not None else float(cy)
        self._prev_cx = float(cx); self._prev_cy = float(cy)

        bw = max(float(box[2]-box[0]), 5.0); bh = max(float(box[3]-box[1]), 5.0)
        af = (bw*bh) / (FRAME_W*FRAME_H); ar = bh/bw
        self._area_hist.append(af); self._ar_hist.append(ar)

        raw_dx = float(cx) - prev_cx; raw_dy = float(cy) - prev_cy

        if self.cfg["ego_compensation"]:
            check_dx, check_dy = comp_dx, comp_dy
        else:
            check_dx, check_dy = raw_dx, raw_dy

        self.direction = self._classify_direction()
        self.is_stationary = self._check_stationary(check_dx, check_dy)

        self._mpp_hist.append(bbox_mpp(bw, cls))
        fm = fleet.fleet_mpp()
        mpp = fm if fm > 0.005 else float(np.median(list(self._mpp_hist)))

        if self.cfg["ego_compensation"]:
            _, _, vx_k, vy_k = self._kalman.update(float(cx) - comp_dx, float(cy) - comp_dy)
        else:
            _, _, vx_k, vy_k = self._kalman.update(float(cx), float(cy))

        raw_spd = math.hypot(vx_k, vy_k) * mpp * self.fps
        self._spd_hist.append(raw_spd)

        prox_cap = max_plausible_speed(bw)
        cam_c = min(camera_speed_mps, self.cfg.get("max_camera_speed_mps", 0.0))
        raw_spd_abs = cam_c if self.is_stationary else min(raw_spd + cam_c, prox_cap)

        raw_fa = (raw_spd - self._prev_speed) / self.dt
        self.fast_accel = float(np.clip(raw_fa, -8.0, 8.0))
        self.speed_mps = 0.30 * raw_spd_abs + 0.70 * self._prev_speed
        self.accel_mps2 = (self.speed_mps - self._prev_speed) / self.dt
        self._prev_speed = self.speed_mps

        self._lat_hist.append(abs(check_dx))
        lat = float(np.median(list(self._lat_hist)))

        if len(self._ar_hist) >= 5:
            am = float(np.mean(list(self._ar_hist)))
            av = float(np.var(list(self._ar_hist)))
            self.is_doing_stunt = am > 2.8 and av > 0.15
            if self.is_doing_stunt: self.score = 0.0; return

        if self._frame_n < self.cfg["warmup_frames"]: self._recompute(); return

        bt = fleet.effective_brake_thr(); at = fleet.effective_accel_thr()
        if self.fast_accel < -bt: self._fire('brake', 'hard_brake_count')
        if self.fast_accel > at: self._fire('accel', 'aggr_accel_count')
        if lat > self.cfg["lane_weave_px"]: self._fire('weave', 'weave_count')
        if (self.direction in (DIR_APPROACH, DIR_ALONGSIDE) and
                len(self._area_hist) >= 5 and af > TAILGATE_AREA_THRESH and
                self._area_hist[-1] - self._area_hist[-5] > TAILGATE_GROWTH_RATE):
            self._fire('tail', 'tailgate_count')
        if (len(self._spd_hist) >= SUDDEN_STOP_FRAMES and not self.is_stationary and
                self._spd_hist[-SUDDEN_STOP_FRAMES] > SUDDEN_STOP_SPEED_MPS and
                self.speed_mps < SUDDEN_STOP_SPEED_MPS):
            self._fire('stop', 'sudden_stop_count')

        if self.cfg["ego_compensation"]:
            if self.speed_mps > SPEED_LIMIT_MPS and self.direction not in (DIR_RECEDING, DIR_UNKNOWN):
                self.over_speed_sec += self.dt
        else:
            if self.speed_mps > SPEED_LIMIT_MPS:
                self.over_speed_sec += self.dt
        self._recompute()

    def _recompute(self):
        w_nh = self.cfg["w_no_helmet"]
        p = (self.hard_brake_count * W_HARD_BRAKE + self.aggr_accel_count * W_AGGR_ACCEL +
             self.weave_count * W_WEAVE + self.tailgate_count * W_TAILGATE +
             self.sudden_stop_count * W_SUDDEN_STOP + self.no_helmet_count * w_nh +
             self.over_speed_sec * W_OVER_SPEED)
        self._raw = max(0.0, min(100.0, 100.0 - p))
        self.score = SCORE_EMA_ALPHA * self._raw + (1 - SCORE_EMA_ALPHA) * self.score
        self._committed = COLOR_EMA_ALPHA * self.score + (1 - COLOR_EMA_ALPHA) * self._committed
        self.reliability = min(100.0, self._frame_n / max(self.cfg["warmup_frames"], 1) * 30.0)

    def speed_kmh(self): return self.speed_mps * 3.6
    def total_events(self):
        return (self.hard_brake_count + self.aggr_accel_count + self.weave_count +
                self.tailgate_count + self.sudden_stop_count + self.no_helmet_count)
    def score_label(self):
        if self.is_doing_stunt: return "STUNT"
        return "GOOD" if self.score >= 80 else ("FAIR" if self.score >= 50 else "POOR")

    def to_dict(self):
        return {
            "rider_id": self.rider_id,
            "score": round(self.score, 2),
            "speed_kmh": round(self.speed_kmh(), 2),
            "hard_brake_count": self.hard_brake_count,
            "aggr_accel_count": self.aggr_accel_count,
            "weave_count": self.weave_count,
            "tailgate_count": self.tailgate_count,
            "sudden_stop_count": self.sudden_stop_count,
            "no_helmet_count": self.no_helmet_count,
            "helmet_status": self.helmet_status,
            "direction": self.direction,
            "is_doing_stunt": self.is_doing_stunt,
            "rating": self.score_label(),
            "total_events": self.total_events(),
        }


# ── Core Analysis Function ──────────────────────────────────────────────────
def analyze_video(video_path: str, mode: str) -> dict:
    """Run the full analysis pipeline on a video and return JSON-serializable results."""
    cfg = MODES.get(mode, MODES["fixed"])
    yolo = get_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK

    tracker = IoUTracker(cfg["iou_match_thresh"])
    fleet = FleetContext(cfg)
    helmet_det = HelmetDetector(cfg["helmet_confirm_frames"])
    ego = EgoMotionEstimator() if cfg["ego_compensation"] else None

    behaviors = {}; all_time = {}; n = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        n += 1
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        edx, edy = ego.update(gray) if ego else (0.0, 0.0)

        results = yolo(frame, conf=cfg["conf_thresh"], verbose=False)[0]
        det_bikes = []; person_boxes = []
        for b in results.boxes:
            cls_id = int(b.cls[0]); box = tuple(map(int, b.xyxy[0])); conf = float(b.conf[0])
            if cls_id in ACTIVE_CLASSES and conf >= cfg["conf_thresh"]:
                det_bikes.append((box, cls_id))
            elif cls_id == 0 and conf >= cfg["person_conf"]:
                person_boxes.append(box)

        tracker.update([b for b, _ in det_bikes])
        def get_cls(box):
            for b, c in det_bikes:
                if b == box: return c
            return 3

        active = tracker.active_tracks
        fleet.update(behaviors)

        cam_spd = 0.0
        if ego and behaviors:
            mv = [float(np.median(list(bv._mpp_hist))) for bv in behaviors.values() if bv._mpp_hist]
            cam_mpp = float(np.median(mv)) if mv else 0.020
            cam_spd = min(math.hypot(edx, edy) * cam_mpp * fps, cfg["max_camera_speed_mps"])

        for tid, td in active.items():
            if tid not in behaviors:
                behaviors[tid] = RiderBehavior(tid, fps, cfg)
            b = behaviors[tid]; cx, cy = td['cx'], td['cy']

            if ego:
                df = EgoMotionEstimator.depth_factor(cy)
                if b._prev_cx is not None:
                    rdx = float(cx) - b._prev_cx; rdy = float(cy) - b._prev_cy
                else:
                    rdx = rdy = 0.0
                comp_dx = rdx - edx * df; comp_dy = rdy - edy * df
                b.update(cx, cy, td['box'], fleet, comp_dx, comp_dy, cam_spd, get_cls(td['box']))
            else:
                b.update(cx, cy, td['box'], fleet, cls=get_cls(td['box']))
            all_time[tid] = b

        statuses = helmet_det.update(gray, active, person_boxes)
        for tid, s in statuses.items():
            if tid in behaviors: behaviors[tid].set_helmet(s)

        alive = set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive]: del behaviors[k]

    cap.release()

    riders = [b.to_dict() for b in sorted(all_time.values(), key=lambda x: -x.score)]
    scores = [b.score for b in all_time.values()] if all_time else [0]

    summary = {
        "avg_score": round(sum(scores) / len(scores), 2),
        "best_rider": max(all_time, key=lambda k: all_time[k].score) if all_time else 0,
        "best_score": round(max(scores), 2),
        "worst_rider": min(all_time, key=lambda k: all_time[k].score) if all_time else 0,
        "worst_score": round(min(scores), 2),
        "good_count": sum(1 for b in all_time.values() if b.score >= 80),
        "fair_count": sum(1 for b in all_time.values() if 50 <= b.score < 80),
        "poor_count": sum(1 for b in all_time.values() if b.score < 50),
        "no_helmet_count": sum(1 for b in all_time.values() if b.no_helmet_count > 0),
    }

    return {
        "mode": mode,
        "total_vehicles": len(all_time),
        "total_frames": n,
        "duration_sec": round(n / fps, 2),
        "fps": round(fps, 2),
        "riders": riders,
        "summary": summary,
    }


# ── Flask Routes ────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    mode = request.form.get("mode", "fixed")
    if mode not in MODES:
        return jsonify({"error": f"Invalid mode: {mode}. Use 'dashcam' or 'fixed'"}), 400

    # Save to temp file
    suffix = os.path.splitext(video.filename)[1] if video.filename else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        video.save(tmp)
        tmp_path = tmp.name

    try:
        result = analyze_video(tmp_path, mode)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
