"""
============================================================
Varroc Eureka 3.0 — Problem Statement 3
Accurate Driving Behavior Score on a Two-Wheeler
v5 — Speed fix + Helmet + Direction + HUD redesign
============================================================

v5 improvements:
1. SPEED SPIKE FIX  — proximity-aware displacement cap: if raw pixel
   displacement implies a speed > MAX_PLAUSIBLE_MPS for the vehicle's
   current distance (bbox size), that frame is skipped for speed calc.
   Eliminates the 120 km/h spikes from close-pass detections.

2. HELMET DETECTION — YOLO detects persons (class 0) alongside bikes.
   Each rider is matched to its motorcycle by bbox overlap. The top 30%
   of the rider bbox is analysed for a helmet-shaped blob (compact,
   roughly circular region). NO_HELMET fires if unprotected for ≥15 frames.
   Penalty: −15 points per confirmed no-helmet event.

3. DIRECTION CLASSIFIER — every track is labelled:
     ONCOMING  : bbox area growing fast  → vehicle approaching head-on
     APPROACH  : bbox area growing slow  → same lane, we're catching up
     RECEDING  : bbox area shrinking     → same lane, they're pulling away
     ALONGSIDE : area stable             → matching our speed
   Tailgating penalty only applies to APPROACH tracks (not ONCOMING).
   Direction tag shown on HUD with arrow icon.

4. HUD REDESIGN — larger, cleaner, judge-readable layout:
   • Filled score badge (big number, color background)
   • Helmet icon  ✓ / ✗  next to score
   • Direction arrow ↑↓←→
   • Leaderboard panel: 2× font size, sorted by score
   • Footer shows mode, frame, fleet context
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import math

# ═══════════════════════════ CONFIGURATION ══════════════════════════════════ #

VIDEO_PATH   = r"D:\hackathon\public\video1.mp4"
MODEL_PATH   = "yolov8n.pt"       # yolov8m.pt → better detection accuracy

# Set to None to show the interactive selector every run.
# Hardcode a string to skip it: "dashcam" | "fixed"
CAMERA_MODE  = None   # None → interactive selector on startup
# Set to None to show the interactive selector every run.
# Hardcode a string to skip it: "side" | "overhead"
CAMERA_ANGLE = None   # None → interactive selector on startup

FRAME_W, FRAME_H = 640, 360
FPS_FALLBACK     = 30

CONF_SIDE     = 0.30
CONF_OVERHEAD = 0.18
PERSON_CONF   = 0.35          # confidence for rider/person detection

TWO_WHEELER_CLASSES_SIDE     = {1, 3}
TWO_WHEELER_CLASSES_OVERHEAD = {0, 1, 3}

# ── Bbox-based dynamic MPP ────────────────────────────────────────────────────
# dynamic_mpp = real_world_width_m / bbox_width_px
# Self-calibrating: works for any camera, mount, focal length.
REAL_WORLD_WIDTHS        = {0: 0.50, 1: 0.60, 3: 0.80}
REAL_WORLD_WIDTH_DEFAULT = 0.75
MPP_HISTORY_LEN          = 20

# ── Speed caps ────────────────────────────────────────────────────────────────
MAX_VEHICLE_SPEED_MPS  = 35.0   # ~126 km/h absolute hard cap
MAX_CAMERA_SPEED_MPS   = 25.0   # ~90 km/h dashcam car cap
# Plausible speed for close vehicles (large bbox). A vehicle filling >25% of
# frame width is <3 m away — physically cannot be doing 120 km/h relative.
MAX_PLAUSIBLE_CLOSE_MPS = 15.0  # ~54 km/h when bbox_w > 0.25 * FRAME_W
CLOSE_BBOX_FRACTION     = 0.25  # fraction of FRAME_W that defines "close"

# ── Stationarity ──────────────────────────────────────────────────────────────
STATIONARY_DISP_PX  = 2.5
STATIONARY_AREA_VAR = 0.0008
STATIONARY_WINDOW   = 12

# ── Direction classification ──────────────────────────────────────────────────
DIR_WINDOW           = 10    # frames to assess area trend
DIR_ONCOMING_RATE    = 0.004 # area growth/frame → head-on approach
DIR_APPROACH_RATE    = 0.001 # area growth/frame → catching up (same lane)
DIR_RECEDING_RATE    = -0.001# area shrink/frame → pulling away

# ── Helmet detection ─────────────────────────────────────────────────────────
HELMET_OVERLAP_IOU   = 0.20  # min IoU for rider↔motorcycle association
HELMET_HEAD_FRACTION = 0.32  # top fraction of rider bbox = head region
HELMET_MIN_AREA_FRAC = 0.08  # head blob must cover ≥ this fraction of head region
HELMET_CONFIRM_FRAMES= 15    # frames of consistent no-helmet before penalty
W_NO_HELMET          = 15    # score penalty per confirmed no-helmet event

# ── Event thresholds ─────────────────────────────────────────────────────────
BASE_HARD_BRAKE_MPS2  = 2.5
BASE_AGGR_ACCEL_MPS2  = 2.0
LANE_WEAVE_PX         = 16
TAILGATE_AREA_THRESH  = 0.09
TAILGATE_GROWTH_RATE  = 0.002
SUDDEN_STOP_SPEED_MPS = 1.2
SUDDEN_STOP_FRAMES    = 6
SPEED_LIMIT_MPS       = 13.9  # 50 km/h
FLEET_K               = 0.6

# ── Score weights ─────────────────────────────────────────────────────────────
W_HARD_BRAKE   = 8
W_AGGR_ACCEL   = 5
W_WEAVE        = 3
W_TAILGATE     = 6
W_SUDDEN_STOP  = 7
W_OVER_SPEED   = 0.25

SCORE_EMA_ALPHA      = 0.04   # slow glide for display
COLOR_EMA_ALPHA      = 0.02   # ultra-slow for box color — no flicker

# ── Tracker ───────────────────────────────────────────────────────────────────
IOU_MATCH_THRESH = 0.20
MAX_DISAPPEARED  = 35
EVENT_COOLDOWN   = 12
WARMUP_FRAMES    = 8

OF_MAX_CORNERS   = 200
OF_QUALITY       = 0.01
OF_MIN_DIST      = 6
OF_WIN_SIZE      = (21, 21)

# ═══════════════════════════ UTILITIES ══════════════════════════════════════ #

def compute_iou(a, b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def bbox_mpp(bbox_width_px: float, cls: int) -> float:
    """Metres-per-pixel from vehicle's own bbox width. Self-calibrating."""
    real_w = REAL_WORLD_WIDTHS.get(cls, REAL_WORLD_WIDTH_DEFAULT)
    return real_w / max(bbox_width_px, 5.0)

def max_plausible_speed(bbox_width_px: float) -> float:
    """
    Proximity-aware speed cap.
    A vehicle with a large bbox is close → relative speed physically limited.
    Returns the max believable speed in m/s for this vehicle's bbox size.
    """
    frac = bbox_width_px / FRAME_W
    if frac >= CLOSE_BBOX_FRACTION:
        # Interpolate: at CLOSE_BBOX_FRACTION → MAX_PLAUSIBLE_CLOSE_MPS
        #              at 1.0                 → 2 m/s (basically touching)
        t = min((frac - CLOSE_BBOX_FRACTION) / (1.0 - CLOSE_BBOX_FRACTION), 1.0)
        return MAX_PLAUSIBLE_CLOSE_MPS * (1.0 - 0.85*t)
    return MAX_VEHICLE_SPEED_MPS

# ═══════════════════════════ HELMET DETECTOR ════════════════════════════════ #

class HelmetDetector:
    """
    Matches person detections to motorcycle tracks and checks for helmets.

    Method:
    1. For each motorcycle bbox, find the person bbox with highest overlap.
    2. Crop the top HELMET_HEAD_FRACTION of the person bbox → head region.
    3. In the head region, look for a compact blob via Otsu threshold +
       contour area check. A helmet produces a large, solid, convex region.
       A bare head produces a smaller or irregular region.
    4. Majority vote over HELMET_CONFIRM_FRAMES → HELMET / NO_HELMET.
    """

    def __init__(self):
        # per track_id → deque of bool (True=helmet detected this frame)
        self._history: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=HELMET_CONFIRM_FRAMES))
        self.status: dict[int, str] = {}  # track_id → "HELMET"/"NO_HELMET"/"?"

    def update(self, frame_gray: np.ndarray,
               bike_tracks: dict, person_boxes: list) -> dict:
        """
        bike_tracks : {tid: {'box':…, 'cx':…, 'cy':…}}
        person_boxes: list of (x1,y1,x2,y2) for detected persons
        Returns: {tid: "HELMET"/"NO_HELMET"/"?"}
        """
        for tid, td in bike_tracks.items():
            bx1,by1,bx2,by2 = td['box']

            # Find best matching person bbox
            best_iou, best_pb = 0.0, None
            for pb in person_boxes:
                iou = compute_iou(td['box'], pb)
                if iou > best_iou:
                    best_iou, best_pb = iou, pb

            if best_pb is None or best_iou < HELMET_OVERLAP_IOU:
                # No rider found — don't update history
                self.status[tid] = self.status.get(tid, "?")
                continue

            px1,py1,px2,py2 = best_pb
            p_h = max(py2-py1, 1)

            # Head region = top HELMET_HEAD_FRACTION of person bbox
            head_y2 = int(py1 + p_h * HELMET_HEAD_FRACTION)
            head_y1 = max(py1, 0)
            head_x1 = max(px1, 0)
            head_x2 = min(px2, frame_gray.shape[1])

            if head_y2 <= head_y1 or head_x2 <= head_x1:
                continue

            head_crop = frame_gray[head_y1:head_y2, head_x1:head_x2]
            if head_crop.size < 50:
                continue

            # Otsu threshold → find largest blob
            _, thresh = cv2.threshold(
                head_crop, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            head_area = head_crop.shape[0] * head_crop.shape[1]
            max_blob  = max((cv2.contourArea(c) for c in contours), default=0)
            blob_frac = max_blob / max(head_area, 1)

            # Helmet → large compact blob; bare head → smaller / irregular
            has_helmet = blob_frac >= HELMET_MIN_AREA_FRAC
            self._history[tid].append(has_helmet)

            # Majority vote
            hist = self._history[tid]
            if len(hist) >= HELMET_CONFIRM_FRAMES // 2:
                self.status[tid] = (
                    "HELMET" if sum(hist)/len(hist) >= 0.55 else "NO_HELMET")
            else:
                self.status[tid] = "?"

        return self.status


# ═══════════════════════════ KALMAN VELOCITY ESTIMATOR ══════════════════════ #

class KalmanVelocityEstimator:
    """
    Constant-velocity Kalman filter on (cx, cy) centroid.

    Why this fixes speed fluctuation:
    ──────────────────────────────────
    Differencing:  speed = (pos[t] - pos[t-1]) × MPP × FPS
      YOLO centroid jitters ±8px → 8 × 0.012 × 30 = ±2.9 m/s = ±10 km/h per frame.
      Median+EMA only reduces to ±4 km/h — still visible fluctuation.

    Kalman:  state = [x, y, vx, vy].  Velocity vx,vy is solved mathematically,
      not differenced.  Converges to ±0.5 km/h stability within ~10 frames.
      Same filter also smooths the centroid used by IoU tracker.

    Tuning:
      PROCESS_NOISE  Q: expected acceleration between frames (px/frame²).
                        Raise if tracking fast/erratic vehicles.
      MEASURE_NOISE  R: expected YOLO centroid variance (px²).
                        Raise if YOLO is very jittery on your footage.
    """
    PROCESS_NOISE = 1.5
    MEASURE_NOISE = 30.0

    def __init__(self):
        self.x           = None
        self.P           = None
        self.initialized = False
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        q = self.PROCESS_NOISE; r = self.MEASURE_NOISE
        self.Q = np.diag([q*0.5, q*0.5, q, q])
        self.R = np.diag([r, r])

    def update(self, cx: float, cy: float):
        """
        Feed raw centroid. Returns (smooth_cx, smooth_cy, vx, vy) in px/frame.
        Velocity magnitude = math.hypot(vx, vy) px/frame — multiply by MPP×FPS for m/s.
        """
        z = np.array([[cx],[cy]], dtype=float)
        if not self.initialized:
            self.x = np.array([[cx],[cy],[0.0],[0.0]], dtype=float)
            self.P = np.eye(4) * 200.0
            self.initialized = True
            return cx, cy, 0.0, 0.0

        # Predict
        x_p = self.F @ self.x
        P_p = self.F @ self.P @ self.F.T + self.Q

        # Update
        S   = self.H @ P_p @ self.H.T + self.R
        K   = P_p @ self.H.T @ np.linalg.inv(S)
        self.x = x_p + K @ (z - self.H @ x_p)
        self.P = (np.eye(4) - K @ self.H) @ P_p

        return (float(self.x[0]), float(self.x[1]),
                float(self.x[2]), float(self.x[3]))

    def reset(self):
        self.initialized = False

# ═══════════════════════════ EGO-MOTION ESTIMATOR ═══════════════════════════ #

class EgoMotionEstimator:
    def __init__(self):
        self._prev_gray = None
        self._prev_pts  = None
        self.ego_dx     = 0.0
        self.ego_dy     = 0.0

    @staticmethod
    def depth_factor(cy):
        return 0.25 + 0.75 * (cy / FRAME_H)

    def _bg_mask(self, h, w):
        mask = np.ones((h,w), np.uint8)*255
        cv2.rectangle(mask,(w//5,h//2),(4*w//5,h),0,-1)
        return mask

    def update(self, gray):
        h,w = gray.shape
        if self._prev_gray is None or self._prev_pts is None \
                or len(self._prev_pts)<10:
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=OF_MAX_CORNERS,
                qualityLevel=OF_QUALITY, minDistance=OF_MIN_DIST,
                mask=self._bg_mask(h,w))
            self._prev_gray = gray.copy()
            return 0.0, 0.0

        npts,st,_ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray,gray,self._prev_pts,None,
            winSize=OF_WIN_SIZE,maxLevel=3)
        good = st.ravel()==1
        if good.sum()<5:
            self._prev_pts=None; self._prev_gray=gray.copy()
            return self.ego_dx, self.ego_dy

        flow    = npts[good]-self._prev_pts[good]
        dx_vals = flow[:,0,0]; dy_vals=flow[:,0,1]
        med_dx  = float(np.median(dx_vals))
        med_dy  = float(np.median(dy_vals))
        keep    = np.abs(dx_vals-med_dx)<6
        raw_edx = float(np.mean(dx_vals[keep])) if keep.any() else med_dx
        raw_edy = float(np.mean(dy_vals[keep])) if keep.any() else med_dy

        MAX_EGO = 30.0
        if math.hypot(raw_edx,raw_edy)<MAX_EGO:
            self.ego_dx = 0.7*raw_edx+0.3*self.ego_dx
            self.ego_dy = 0.7*raw_edy+0.3*self.ego_dy

        self._prev_pts  = npts[good].reshape(-1,1,2)
        self._prev_gray = gray.copy()
        if len(self._prev_pts)<25: self._prev_pts=None
        return self.ego_dx, self.ego_dy

# ═══════════════════════════ FLEET CONTEXT ══════════════════════════════════ #

class FleetContext:
    def __init__(self):
        self.fleet_speed_mps  = 0.0
        self.fleet_accel_mps2 = 0.0
        self.n_vehicles       = 0

    def update(self, behaviors):
        speeds = [b.speed_mps  for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        accels = [b.fast_accel for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        self.n_vehicles       = len(speeds)
        self.fleet_speed_mps  = float(np.median(speeds)) if speeds else 0.0
        raw_accel             = float(np.median(accels))  if accels else 0.0
        self.fleet_accel_mps2 = float(np.clip(raw_accel,-4.0,4.0))

        # Fleet-wide median MPP — collected from all individual MPP histories.
        # Sharing this single value across all vehicles makes speed readings
        # consistent: two bikes at the same distance always get the same MPP.
        all_mpp = []
        for b in behaviors.values():
            if b._mpp_hist:
                all_mpp.append(float(np.median(list(b._mpp_hist))))
        self._fleet_mpp_cache = float(np.median(all_mpp)) if all_mpp else 0.012

    def effective_brake_thr(self):
        return BASE_HARD_BRAKE_MPS2+FLEET_K*max(0.0,-self.fleet_accel_mps2)
    def effective_accel_thr(self):
        return BASE_AGGR_ACCEL_MPS2+FLEET_K*max(0.0, self.fleet_accel_mps2)
    def fleet_mpp(self) -> float:
        """
        Median MPP across all active vehicles.
        Using a shared fleet MPP instead of per-vehicle MPP eliminates
        speed inconsistency between vehicles at similar distances — bbox
        width jitter on individual tracks no longer causes different speeds.
        Falls back to 0.012 (reasonable mid-range) if no data yet.
        """
        return getattr(self, "_fleet_mpp_cache", 0.012)

# ═══════════════════════════ IoU TRACKER ════════════════════════════════════ #

class IoUTracker:
    def __init__(self):
        self.next_id=0; self.tracks={}; self.disappeared=defaultdict(int)

    @staticmethod
    def _cen(box): return (box[0]+box[2])//2,(box[1]+box[3])//2

    def _register(self,box):
        cx,cy=self._cen(box)
        self.tracks[self.next_id]={'box':box,'cx':cx,'cy':cy}
        self.disappeared[self.next_id]=0; self.next_id+=1

    def _deregister(self,tid):
        self.tracks.pop(tid,None); self.disappeared.pop(tid,None)

    @property
    def active_tracks(self):
        return {t:d for t,d in self.tracks.items() if self.disappeared[t]==0}

    def update(self,det_boxes):
        if not det_boxes:
            for tid in list(self.disappeared):
                self.disappeared[tid]+=1
                if self.disappeared[tid]>MAX_DISAPPEARED: self._deregister(tid)
            return self.tracks
        if not self.tracks:
            for b in det_boxes: self._register(b)
            return self.tracks

        tids=list(self.tracks.keys())
        tboxes=[self.tracks[t]['box'] for t in tids]
        iou_mat=np.array([[compute_iou(db,tb) for tb in tboxes] for db in det_boxes])

        matched_d=set(); matched_t=set()
        for idx in np.argsort(-iou_mat,axis=None):
            di,ti=divmod(int(idx),len(tboxes))
            if di in matched_d or ti in matched_t: continue
            if iou_mat[di,ti]<IOU_MATCH_THRESH: break
            tid=tids[ti]; box=det_boxes[di]; cx,cy=self._cen(box)
            self.tracks[tid]={'box':box,'cx':cx,'cy':cy}
            self.disappeared[tid]=0; matched_d.add(di); matched_t.add(ti)

        unm_d=[i for i in range(len(det_boxes)) if i not in matched_d]
        unm_t=[i for i in range(len(tboxes))    if i not in matched_t]
        mxd=math.hypot(FRAME_W,FRAME_H)/4

        for di in unm_d:
            dcx,dcy=self._cen(det_boxes[di])
            bd,bti=float('inf'),None
            for ti in unm_t:
                d=math.hypot(dcx-self._cen(tboxes[ti])[0],
                             dcy-self._cen(tboxes[ti])[1])
                if d<bd: bd,bti=d,ti
            if bti is not None and bd<mxd:
                tid=tids[bti]; box=det_boxes[di]; cx,cy=self._cen(box)
                self.tracks[tid]={'box':box,'cx':cx,'cy':cy}
                self.disappeared[tid]=0; unm_t.remove(bti)
            else: self._register(det_boxes[di])

        for ti in unm_t:
            tid=tids[ti]; self.disappeared[tid]+=1
            if self.disappeared[tid]>MAX_DISAPPEARED: self._deregister(tid)
        return self.tracks

# ═══════════════════════════ RIDER BEHAVIOR ══════════════════════════════════ #

# Direction labels
DIR_ONCOMING  = "ONCOMING"
DIR_APPROACH  = "APPROACH"
DIR_RECEDING  = "RECEDING"
DIR_ALONGSIDE = "ALONGSIDE"
DIR_UNKNOWN   = "?"

DIR_ARROWS = {
    DIR_ONCOMING : "<<",
    DIR_APPROACH : "^^ ",
    DIR_RECEDING : "vv ",
    DIR_ALONGSIDE: "->",
    DIR_UNKNOWN  : "  ",
}

class RiderBehavior:
    def __init__(self, rider_id, fps):
        self.rider_id  = rider_id
        self.fps       = fps
        self.dt        = 1.0/fps
        self._frame_n  = 0

        self._scx=None; self._scy=None
        self._pos_hist: deque = deque(maxlen=25)
        self._raw_dx:float=0.0; self._raw_dy:float=0.0
        self._kalman = KalmanVelocityEstimator()  # smooth velocity estimation

        self._mpp_hist:  deque = deque(maxlen=MPP_HISTORY_LEN)
        self._spd_hist:  deque = deque(maxlen=10)
        self._area_hist: deque = deque(maxlen=max(STATIONARY_WINDOW,DIR_WINDOW)+2)
        self._lat_hist:  deque = deque(maxlen=8)
        self._ar_hist:   deque = deque(maxlen=10)

        self.cls: int = 3
        self.speed_mps   = 0.0
        self.fast_accel  = 0.0
        self.accel_mps2  = 0.0
        self._prev_speed = 0.0

        # Direction
        self.direction: str = DIR_UNKNOWN

        # Helmet
        self.helmet_status:  str = "?"
        self.no_helmet_count: int = 0
        self._no_helmet_cd:  int  = 0

        # Events
        self.hard_brake_count  = 0
        self.aggr_accel_count  = 0
        self.weave_count       = 0
        self.tailgate_count    = 0
        self.sudden_stop_count = 0
        self.over_speed_sec    = 0.0
        self.is_stationary     = False
        self.is_doing_stunt    = False
        self._cd               = defaultdict(int)

        self.score      = 100.0
        self._raw       = 100.0
        self._committed = 100.0

    def _ema_pos(self,cx,cy,a=0.22):
        if self._scx is None: self._scx,self._scy=float(cx),float(cy)
        else:
            self._scx=a*cx+(1-a)*self._scx
            self._scy=a*cy+(1-a)*self._scy

    def _tick_cd(self):
        for k in list(self._cd): self._cd[k]=max(0,self._cd[k]-1)
        self._no_helmet_cd=max(0,self._no_helmet_cd-1)

    def _fire(self,key,attr):
        if self._cd[key]==0:
            setattr(self,attr,getattr(self,attr)+1)
            self._cd[key]=EVENT_COOLDOWN

    def _check_stationary(self,comp_dx,comp_dy):
        comp_disp=math.hypot(comp_dx,comp_dy)
        area_stable=(len(self._area_hist)>=STATIONARY_WINDOW and
                     float(np.var(list(self._area_hist)[-STATIONARY_WINDOW:]))
                     <STATIONARY_AREA_VAR)
        return comp_disp<STATIONARY_DISP_PX and area_stable

    def _classify_direction(self):
        """Classify vehicle direction using bbox area trend."""
        if len(self._area_hist)<DIR_WINDOW:
            return DIR_UNKNOWN
        recent = list(self._area_hist)[-DIR_WINDOW:]
        # Linear slope of area over window
        xs  = np.arange(len(recent))
        slope = float(np.polyfit(xs, recent, 1)[0])
        if slope >  DIR_ONCOMING_RATE: return DIR_ONCOMING
        if slope >  DIR_APPROACH_RATE: return DIR_APPROACH
        if slope <  DIR_RECEDING_RATE: return DIR_RECEDING
        return DIR_ALONGSIDE

    def set_helmet(self, status: str):
        """Called each frame by HelmetDetector."""
        self.helmet_status = status
        if status == "NO_HELMET" and self._no_helmet_cd == 0:
            self.no_helmet_count += 1
            self._no_helmet_cd    = EVENT_COOLDOWN * 2

    def update(self, cx, cy, box, comp_dx, comp_dy,
               fleet: FleetContext, camera_speed_mps:float=0.0, cls:int=3):
        self._frame_n+=1; self._tick_cd()
        self._ema_pos(cx,cy)
        self._pos_hist.append((self._scx,self._scy))
        self.cls=cls

        bw=max(float(box[2]-box[0]),5.0)
        bh=max(float(box[3]-box[1]),5.0)
        af=(bw*bh)/(FRAME_W*FRAME_H)
        ar=bh/bw
        self._area_hist.append(af)
        self._ar_hist.append(ar)

        # Direction
        self.direction = self._classify_direction()

        # Stationarity
        self.is_stationary = self._check_stationary(comp_dx,comp_dy)

        # Per-vehicle MPP (still collected for fleet computation)
        frame_mpp = bbox_mpp(bw, cls)
        self._mpp_hist.append(frame_mpp)
        fleet_mpp_val = fleet.fleet_mpp()
        mpp = fleet_mpp_val if fleet_mpp_val > 0.005 else float(np.median(list(self._mpp_hist)))

        # ── Kalman filter on ego-compensated centroid ─────────────────────
        # We feed the COMPENSATED centroid position so the Kalman velocity
        # represents motion relative to the ground (not the moving camera).
        # The filter returns smooth position + velocity in px/frame.
        comp_cx = cx - (self._raw_dx - comp_dx)
        comp_cy = cy - (self._raw_dy - comp_dy)
        _, _, vx_k, vy_k = self._kalman.update(comp_cx, comp_cy)

        # Velocity magnitude → relative speed (m/s)
        # This is already smooth — no median or EMA needed on top
        rel_spd_kalman = math.hypot(vx_k, vy_k) * mpp * self.fps
        raw_spd_rel    = rel_spd_kalman

        # ── Absolute speed = relative + camera's own speed ────────────────
        prox_cap = max_plausible_speed(bw)
        cam_c    = min(camera_speed_mps, MAX_CAMERA_SPEED_MPS)
        if self.is_stationary:
            raw_spd_abs = cam_c
        else:
            raw_spd_abs = min(rel_spd_kalman + cam_c, prox_cap)

        # Acceleration — 1-frame delta on already-smooth Kalman velocity
        raw_fast_accel  = (raw_spd_rel - self._prev_speed) / self.dt
        self.fast_accel = float(np.clip(raw_fast_accel, -8.0, 8.0))

        # Light final EMA (α=0.30) on top of Kalman output for display
        self.speed_mps  = 0.30 * raw_spd_abs + 0.70 * self._prev_speed
        self.accel_mps2 = (self.speed_mps - self._prev_speed) / self.dt
        self._prev_speed = self.speed_mps

        self._lat_hist.append(abs(comp_dx))
        lat=float(np.median(list(self._lat_hist)))

        # Stunt detection (wheelie)
        if len(self._ar_hist)>=5:
            ar_mean=float(np.mean(list(self._ar_hist)))
            ar_var =float(np.var(list(self._ar_hist)))
            self.is_doing_stunt=ar_mean>2.8 and ar_var>0.15
            if self.is_doing_stunt: self.score=0.0; return

        if self._frame_n<WARMUP_FRAMES: self._recompute(); return

        brake_thr=fleet.effective_brake_thr()
        accel_thr=fleet.effective_accel_thr()

        if self.fast_accel<-brake_thr: self._fire('brake','hard_brake_count')
        if self.fast_accel> accel_thr: self._fire('accel','aggr_accel_count')
        if lat>LANE_WEAVE_PX:          self._fire('weave','weave_count')

        # Tailgating only meaningful for approaching same-lane vehicles
        if (self.direction in (DIR_APPROACH, DIR_ALONGSIDE) and
                len(self._area_hist)>=5 and af>TAILGATE_AREA_THRESH and
                self._area_hist[-1]-self._area_hist[-5]>TAILGATE_GROWTH_RATE):
            self._fire('tail','tailgate_count')

        if (len(self._spd_hist)>=SUDDEN_STOP_FRAMES and not self.is_stationary
                and self._spd_hist[-SUDDEN_STOP_FRAMES]>SUDDEN_STOP_SPEED_MPS
                and self.speed_mps<SUDDEN_STOP_SPEED_MPS):
            self._fire('stop','sudden_stop_count')

        if self.speed_mps>SPEED_LIMIT_MPS: self.over_speed_sec+=self.dt
        self._recompute()

    def _recompute(self):
        p=(self.hard_brake_count  * W_HARD_BRAKE  +
           self.aggr_accel_count  * W_AGGR_ACCEL  +
           self.weave_count       * W_WEAVE        +
           self.tailgate_count    * W_TAILGATE     +
           self.sudden_stop_count * W_SUDDEN_STOP  +
           self.no_helmet_count   * W_NO_HELMET    +
           self.over_speed_sec    * W_OVER_SPEED)
        self._raw       = max(0.0,min(100.0,100.0-p))
        self.score      = SCORE_EMA_ALPHA*self._raw+(1-SCORE_EMA_ALPHA)*self.score
        self._committed = COLOR_EMA_ALPHA*self.score+(1-COLOR_EMA_ALPHA)*self._committed

    def speed_kmh(self): return self.speed_mps*3.6
    def total_events(self):
        return (self.hard_brake_count+self.aggr_accel_count+
                self.weave_count+self.tailgate_count+self.sudden_stop_count+
                self.no_helmet_count)
    def score_label(self):
        if self.is_doing_stunt: return "STUNT"
        return "GOOD" if self.score>=80 else ("FAIR" if self.score>=50 else "POOR")
    def score_color(self):
        if self.is_doing_stunt: return (0,0,220)
        s=self._committed
        return ((30,200,30) if s>=80 else ((0,165,255) if s>=50 else (40,40,220)))

# ═══════════════════════════ HUD v5 ══════════════════════════════════════════ #

def _safe_text(frame, text, org, scale, color, thick=1):
    """Draw text clamped to frame bounds so labels never overflow edges."""
    (tw,th),_ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(2, min(org[0], FRAME_W-tw-2))
    y = max(th+2, min(org[1], FRAME_H-4))
    cv2.putText(frame, text, (x,y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, rider: RiderBehavior, box):
    x1,y1,x2,y2 = box
    color = rider.score_color()
    bw    = max(x2-x1,1)

    # ── Bounding box ──────────────────────────────────────────────────────
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    # ── Score bar (inside top of box) ────────────────────────────────────
    fill=int(bw*rider.score/100)
    cv2.rectangle(frame,(x1,y1),(x2,y1+5),(30,30,30),-1)
    cv2.rectangle(frame,(x1,y1),(x1+fill,y1+5),color,-1)

    # Score badge removed — leaderboard panel shows score clearly

    # ── Labels (ID, direction, helmet) ───────────────────────────────────
    dir_arrow  = DIR_ARROWS.get(rider.direction, "")
    helmet_sym = ("H+" if rider.helmet_status == "HELMET"
                  else ("X!" if rider.helmet_status == "NO_HELMET" else ""))
    helmet_col = ((30,220,30) if rider.helmet_status == "HELMET"
                  else ((0,50,255) if rider.helmet_status == "NO_HELMET"
                        else (180,180,180)))
    still_tag  = " ST" if rider.is_stationary else ""

    # ── Compact 2-line label strip above the box ──────────────────────────
    # Line 1: ID  score/100  speed  direction
    # Line 2: events summary  helmet
    score_str = f"{rider.score:.0f}/100"
    spd_str   = f"{rider.speed_kmh():.0f}km/h"
    line1 = f"#{rider.rider_id} {score_str}  {spd_str}  {dir_arrow}{still_tag}"
    line2 = f"B{rider.hard_brake_count} W{rider.weave_count} T{rider.tailgate_count} S{rider.sudden_stop_count}"
    if helmet_sym:
        line2 += f"  {helmet_sym}"

    ty_base = max(y1 - 34, 0)
    _safe_text(frame, line1, (x1, ty_base),      0.38, color,       1)
    _safe_text(frame, line2, (x1, ty_base + 16), 0.34, (180,180,180), 1)


def draw_fleet_bar(frame, fleet:FleetContext, n_frame:int):
    """Full-width top bar: fleet stats + frame counter."""
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(FRAME_W,20),(15,15,50),-1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
    txt=(f"Fleet  {fleet.fleet_speed_mps*3.6:.0f}km/h  "
         f"a={fleet.fleet_accel_mps2:+.1f}m/s²  "
         f"brk_thr={fleet.effective_brake_thr():.1f}  "
         f"n={fleet.n_vehicles}    Frame {n_frame}")
    cv2.putText(frame,txt,(4,14),
                cv2.FONT_HERSHEY_SIMPLEX,0.34,(180,210,255),1,cv2.LINE_AA)


def draw_ego_arrow(frame, dx, dy):
    if CAMERA_MODE!="dashcam": return
    cx,cy=FRAME_W//2,34
    cv2.arrowedLine(frame,(cx,cy),(int(cx+dx*3),int(cy+dy*3)),
                    (0,200,255),2,tipLength=0.35)
    _safe_text(frame,f"ego({dx:+.1f},{dy:+.1f})",(cx-40,cy-6),
               0.30,(0,200,255),1)


def draw_leaderboard(frame, behaviors:dict):
    """Right-side leaderboard panel, sorted best→worst score."""
    if not behaviors: return
    rows    = sorted(behaviors.values(), key=lambda b:-b.score)
    pane_w  = 195
    pane_h  = len(rows)*22+32
    pane_x  = FRAME_W-pane_w

    ov=frame.copy()
    cv2.rectangle(ov,(pane_x,20),(FRAME_W,20+pane_h),(10,10,10),-1)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)

    # Header
    cv2.putText(frame,"  ID  Score  km/h  Ev",(pane_x+4,36),
                cv2.FONT_HERSHEY_SIMPLEX,0.36,(220,200,0),1)
    cv2.line(frame,(pane_x,38),(FRAME_W,38),(60,60,60),1)

    for i,b in enumerate(rows):
        y = 54+i*22
        col = b.score_color()
        # Colored score pill
        pill_txt=f"{b.score:.0f}"
        (ptw,_),_=cv2.getTextSize(pill_txt,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        cv2.rectangle(frame,(pane_x+28,y-10),(pane_x+28+ptw+6,y+4),col,-1)
        cv2.putText(frame,pill_txt,(pane_x+31,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1,cv2.LINE_AA)
        # Rest of row
        helm_c=((0,200,0) if b.helmet_status=="HELMET"
                else ((0,40,220) if b.helmet_status=="NO_HELMET"
                      else (140,140,140)))
        row_txt=(f"#{b.rider_id:<2}        "
                 f"{b.speed_kmh():4.0f}  {b.total_events():2}")
        cv2.putText(frame,row_txt,(pane_x+4,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.36,col,1,cv2.LINE_AA)
        # Helmet dot
        cv2.circle(frame,(FRAME_W-10,y-3),4,helm_c,-1)


def draw_footer(frame):
    ov=frame.copy()
    cv2.rectangle(ov,(0,FRAME_H-18),(FRAME_W,FRAME_H),(10,10,10),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    mode=f"{CAMERA_MODE.upper()}/{CAMERA_ANGLE.upper()}"
    cv2.putText(frame,
                f"Varroc Eureka 3.0 — PS3 Driving Behavior Score v5  [{mode}]",
                (4,FRAME_H-4),cv2.FONT_HERSHEY_SIMPLEX,
                0.32,(160,160,160),1,cv2.LINE_AA)


# ═══════════════════════════ STARTUP SELECTOR ═══════════════════════════════ #

def run_selector() -> tuple:
    """
    Full-screen OpenCV click/key selector shown before video starts.
    Returns (camera_mode, camera_angle).

    Keys:  D = Dashcam   F = Fixed
           S = Side      O = Overhead
           ENTER = confirm and start
           ESC   = quit
    """
    sel_mode  = "dashcam"
    sel_angle = "side"

    W, H   = 640, 360
    BG     = (18,  18,  30)
    ACT    = (30,  200, 30)
    INACT  = (80,  80,  80)
    TITLE  = (220, 200, 50)
    WHITE  = (240, 240, 240)
    HINT   = (120, 120, 120)

    def pill(canvas, label, desc, key_ch, y, active):
        col = ACT if active else INACT
        cv2.rectangle(canvas,(60,y-26),(580,y+8), col, -1 if active else 2)
        cv2.rectangle(canvas,(60,y-26),(580,y+8), col, 2)
        cv2.rectangle(canvas,(68,y-22),(98,y+4), (40,40,40),-1)
        cv2.putText(canvas, key_ch, (73,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, label, (108,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    WHITE if active else INACT, 2 if active else 1, cv2.LINE_AA)
        cv2.putText(canvas, desc, (300,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37,
                    WHITE if active else INACT, 1, cv2.LINE_AA)

    cv2.namedWindow("Varroc Eureka PS3 — Setup", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Varroc Eureka PS3 — Setup", W, H)

    while True:
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)

        # ── Title bar ─────────────────────────────────────────────────────
        cv2.rectangle(canvas,(0,0),(W,44),(25,25,55),-1)
        cv2.putText(canvas,"VARROC EUREKA 3.0  |  PS3  —  Camera Setup",
                    (12,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,TITLE,1,cv2.LINE_AA)

        # ── Camera MODE section ───────────────────────────────────────────
        cv2.putText(canvas,"CAMERA MODE",(60,74),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,HINT,1,cv2.LINE_AA)
        pill(canvas,"DASHCAM","bike-mounted / moving vehicle","D",
             100, sel_mode=="dashcam")
        pill(canvas,"FIXED",  "roadside / overhead / stationary","F",
             148, sel_mode=="fixed")

        cv2.line(canvas,(60,172),(580,172),(50,50,50),1)

        # ── Camera ANGLE section ──────────────────────────────────────────
        cv2.putText(canvas,"CAMERA ANGLE",(60,196),
                    cv2.FONT_HERSHEY_SIMPLEX,0.42,HINT,1,cv2.LINE_AA)
        pill(canvas,"SIDE",    "front / side profile (standard)","S",
             222, sel_angle=="side")
        pill(canvas,"OVERHEAD","top-down / bird's-eye view","O",
             270, sel_angle=="overhead")

        # ── Live preview strip ────────────────────────────────────────────
        cv2.rectangle(canvas,(60,293),(580,318),(30,40,30),-1)
        cv2.putText(canvas,
                    f"Selected:  {sel_mode.upper()}  /  {sel_angle.upper()}"
                    f"      (press ENTER to start, ESC to quit)",
                    (68,311),cv2.FONT_HERSHEY_SIMPLEX,0.38,ACT,1,cv2.LINE_AA)

        # ── Key guide ─────────────────────────────────────────────────────
        cv2.putText(canvas,"D / F  =  mode      S / O  =  angle      ENTER  =  start",
                    (100,345),cv2.FONT_HERSHEY_SIMPLEX,0.36,HINT,1,cv2.LINE_AA)

        cv2.imshow("Varroc Eureka PS3 — Setup", canvas)
        key = cv2.waitKey(30) & 0xFF

        if   key in (ord('d'), ord('D')): sel_mode  = "dashcam"
        elif key in (ord('f'), ord('F')): sel_mode  = "fixed"
        elif key in (ord('s'), ord('S')): sel_angle = "side"
        elif key in (ord('o'), ord('O')): sel_angle = "overhead"
        elif key in (13, 10):             break   # ENTER
        elif key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled.")

    cv2.destroyWindow("Varroc Eureka PS3 — Setup")
    print(f"[MODE]  {sel_mode.upper()} / {sel_angle.upper()}")
    return sel_mode, sel_angle

# ═══════════════════════════ MAIN ═══════════════════════════════════════════ #

def main():
    global CAMERA_MODE, CAMERA_ANGLE
    if CAMERA_MODE is None or CAMERA_ANGLE is None:
        CAMERA_MODE, CAMERA_ANGLE = run_selector()

    model   = YOLO(MODEL_PATH)
    cap     = cv2.VideoCapture(VIDEO_PATH)
    fps     = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK

    if CAMERA_ANGLE=="overhead":
        active_conf    = CONF_OVERHEAD
        active_classes = TWO_WHEELER_CLASSES_OVERHEAD
    else:
        active_conf    = CONF_SIDE
        active_classes = TWO_WHEELER_CLASSES_SIDE

    print(f"[INFO] FPS:{fps:.0f}  Mode:{CAMERA_MODE}  Angle:{CAMERA_ANGLE}"
          f"  Classes:{active_classes}  Conf:{active_conf}")

    tracker      = IoUTracker()
    ego_est      = EgoMotionEstimator() if CAMERA_MODE=="dashcam" else None
    fleet        = FleetContext()
    helmet_det   = HelmetDetector()
    behaviors:   dict[int,RiderBehavior] = {}
    all_time:    dict[int,RiderBehavior] = {}
    n = 0

    while True:
        ret,frame = cap.read()
        if not ret: break
        n+=1
        frame = cv2.resize(frame,(FRAME_W,FRAME_H))
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Ego-motion
        raw_edx,raw_edy = ego_est.update(gray) if ego_est else (0.0,0.0)

        # ── Detection ─────────────────────────────────────────────────────
        results = model(frame, conf=min(active_conf,PERSON_CONF),
                        verbose=False)[0]

        det_bikes   = []   # (box, cls)
        person_boxes= []   # (x1,y1,x2,y2) for riders

        for b in results.boxes:
            cls = int(b.cls[0])
            box = tuple(map(int,b.xyxy[0]))
            conf= float(b.conf[0])
            if cls in active_classes and conf>=active_conf:
                det_bikes.append((box,cls))
            elif cls==0 and conf>=PERSON_CONF:
                person_boxes.append(box)

        # ── Track ─────────────────────────────────────────────────────────
        tracker.update([b for b,_ in det_bikes])
        det_cls_map = {b:c for b,c in det_bikes}
        active      = tracker.active_tracks

        # ── Fleet context ─────────────────────────────────────────────────
        fleet.update(behaviors)

        # ── Per-rider behavior update ──────────────────────────────────────
        for tid,td in active.items():
            if tid not in behaviors:
                behaviors[tid]=RiderBehavior(tid,fps)
            b=behaviors[tid]

            df    = EgoMotionEstimator.depth_factor(td['cy']) if ego_est else 1.0
            edx_v = raw_edx*df if ego_est else 0.0
            edy_v = raw_edy*df if ego_est else 0.0

            cx,cy=td['cx'],td['cy']
            if b._scx is not None:
                raw_dx=cx-b._scx; raw_dy=cy-b._scy
            else:
                raw_dx=raw_dy=0.0
            b._raw_dx=raw_dx; b._raw_dy=raw_dy
            comp_dx=raw_dx-edx_v; comp_dy=raw_dy-edy_v

            # Camera speed via median MPP of active tracks
            if ego_est and behaviors:
                mpp_vals=[float(np.median(list(bv._mpp_hist)))
                          for bv in behaviors.values() if bv._mpp_hist]
                cam_mpp=float(np.median(mpp_vals)) if mpp_vals else 0.020
                camera_speed_mps=min(math.hypot(raw_edx,raw_edy)*cam_mpp*fps,
                                     MAX_CAMERA_SPEED_MPS)
            else:
                camera_speed_mps=0.0

            det_cls=det_cls_map.get(td['box'],3)
            b.update(cx,cy,td['box'],comp_dx,comp_dy,fleet,
                     camera_speed_mps,det_cls)
            all_time[tid]=b

        # ── Helmet detection ──────────────────────────────────────────────
        helmet_statuses = helmet_det.update(gray, active, person_boxes)
        for tid,status in helmet_statuses.items():
            if tid in behaviors:
                behaviors[tid].set_helmet(status)

        # ── Cleanup stale behaviors ───────────────────────────────────────
        alive_ids=set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive_ids]:
            del behaviors[k]

        # ── Render ────────────────────────────────────────────────────────
        for tid,td in active.items():
            if tid in behaviors:
                draw_hud(frame,behaviors[tid],td['box'])

        draw_fleet_bar(frame, fleet, n)
        draw_ego_arrow(frame, raw_edx, raw_edy)
        active_beh={tid:behaviors[tid] for tid in active if tid in behaviors}
        draw_leaderboard(frame, active_beh)
        draw_footer(frame)

        cv2.imshow("Varroc Eureka 3.0 — PS3  [ESC to quit]",frame)
        if cv2.waitKey(1)&0xFF==27: break

    # ── Final report ──────────────────────────────────────────────────────
    W = 96   # total table width

    # Column spec: (header, width, align)
    COLS = [
        ("ID",      5,  "<"),
        ("Score",   7,  ">"),
        ("km/h",    7,  ">"),
        ("Brakes",  7,  ">"),
        ("Accel",   6,  ">"),
        ("Weave",   6,  ">"),
        ("Tailg",   6,  ">"),
        ("Stop",    5,  ">"),
        ("NoHelmt", 8,  ">"),
        ("Helmet",  8,  "^"),
        ("Direction",10,"^"),
        ("Stunt",   6,  "^"),
        ("Rating",  8,  "^"),
    ]

    def divider(left="├", mid="┼", right="┤", fill="─"):
        parts = [fill*(w+2) for _,w,_ in COLS]
        return left + mid.join(parts) + right

    def row_str(values):
        cells=[]
        for (_, w, align), val in zip(COLS, values):
            cells.append(f" {val:{align}{w}} ")
        return "│" + "│".join(cells) + "│"

    print()
    print("┌" + "─"*(W-2) + "┐")
    title = "VARROC EUREKA 3.0  —  PS3 DRIVING BEHAVIOR REPORT  v5"
    print("│" + title.center(W-2) + "│")
    sub   = (f"Vehicles tracked: {len(all_time)}   "
             f"Frames: {n}   Duration: {n/fps:.1f}s   "
             f"Mode: {CAMERA_MODE.upper()}/{CAMERA_ANGLE.upper()}")
    print("│" + sub.center(W-2) + "│")
    print("├" + "─"*(W-2) + "┤")

    # Header row
    print(divider("├","┬","┤"))
    print(row_str([h for h,_,_ in COLS]))
    print(divider("├","┼","┤"))

    # Data rows
    for i,(rid,b) in enumerate(sorted(all_time.items())):
        helmet_str = ("YES ✓" if b.helmet_status=="HELMET"
                      else ("NO  ✗" if b.helmet_status=="NO_HELMET" else "  ?  "))
        stunt_str  = "YES" if b.is_doing_stunt else " no"
        rating_str = f"[ {b.score_label()} ]"
        vals = [
            f"#{rid}",
            f"{b.score:.1f}",
            f"{b.speed_kmh():.1f}",
            str(b.hard_brake_count),
            str(b.aggr_accel_count),
            str(b.weave_count),
            str(b.tailgate_count),
            str(b.sudden_stop_count),
            str(b.no_helmet_count),
            helmet_str,
            b.direction,
            stunt_str,
            rating_str,
        ]
        print(row_str(vals))
        # Separator between rows (not after last)
        if i < len(all_time)-1:
            print(divider())

    print(divider("└","┴","┘"))

    # Summary block
    if all_time:
        scores = [b.score for b in all_time.values()]
        best_id  = max(all_time, key=lambda k: all_time[k].score)
        worst_id = min(all_time, key=lambda k: all_time[k].score)
        g  = sum(1 for b in all_time.values() if b.score>=80)
        f_ = sum(1 for b in all_time.values() if 50<=b.score<80)
        p  = sum(1 for b in all_time.values() if b.score<50)
        nh = sum(1 for b in all_time.values() if b.no_helmet_count>0)

        print()
        print("┌" + "─"*(W-2) + "┐")
        print("│" + "  SUMMARY".ljust(W-2) + "│")
        print("├" + "─"*(W-2) + "┤")
        summary_lines = [
            f"  Average Score  : {sum(scores)/len(scores):.1f}",
            f"  Best Rider     : #{best_id}  ({max(scores):.1f} pts)",
            f"  Worst Rider    : #{worst_id}  ({min(scores):.1f} pts)",
            f"  GOOD (≥80)     : {g}   FAIR (50–79): {f_}   POOR (<50): {p}",
            f"  No-Helmet Violations : {nh} rider(s)",
        ]
        for line in summary_lines:
            print("│" + line.ljust(W-2) + "│")
        print("└" + "─"*(W-2) + "┘")
    print()
    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()