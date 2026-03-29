"""
============================================================
Varroc Eureka 3.0 — Problem Statement 3
Accurate Driving Behavior Score on a Two-Wheeler
v4 — Context-aware, Dashcam + Fixed Camera
============================================================

Changes vs v3:
──────────────
BUG 1 ► Still vehicles show speed ~80 km/h
  FIX: Per-vehicle expected ego-flow is computed using the vehicle's
       own depth proxy (y-position). If the vehicle's actual pixel
       displacement ≈ expected ego displacement → it is STATIONARY
       relative to the ground → speed clamped to 0.
       Also: if bbox area is stable over 10 frames (vehicle neither
       approaching nor receding), speed is forced to 0.

BUG 2 ► POV dashcam performs worse than fixed camera
  FIX: CAMERA_MODE toggle ("dashcam" | "fixed").
       Fixed mode: skip ego-flow entirely; raw pixel displacement
       scaled by perspective MPP is the ground truth.
       Dashcam mode: full ego-compensation + stationarity check.

BUG 3 ► Slow to detect sudden braking / acceleration
  FIX: Two separate acceleration signals:
       • fast_accel  – 2-frame raw delta, no smoothing → used ONLY
                       for event triggering (no lag).
       • disp_accel  – smoothed via EMA → shown on HUD only.
       Events fire on fast_accel so a 1-frame brake spike is caught
       immediately instead of after 8–10 frames of EMA lag.

BUG 4 ► Traffic slowdowns penalise individual riders unfairly
  FIX: FleetContext collects median speed + median accel across ALL
       tracked vehicles every frame. Dynamic threshold scaling:
         effective_brake_thr = BASE_thr + k * |fleet_median_decel|
       When everyone brakes (traffic), the threshold rises so only
       braking HARDER than traffic counts as a hard-brake event.
       Same logic for aggressive acceleration.

BONUS (deferred) ► Wheelie / stunt detection → score = 0
  Stub included, not yet active.
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import math

# ═══════════════════════════ CONFIGURATION ══════════════════════════════════ #

VIDEO_PATH = r"D:\hackathon\public\video1.mp4"
MODEL_PATH = "yolov8n.pt"        # swap to yolov8m.pt for accuracy

# ── Camera mode ───────────────────────────────────────────────────────────────
# "fixed"   → roadside / overhead camera (no ego-motion compensation)
# "dashcam" → camera mounted in a moving vehicle
CAMERA_MODE = "dashcam"

FRAME_W, FRAME_H = 640, 360
CONF_THRESHOLD   = 0.30
FPS_FALLBACK     = 30

TWO_WHEELER_CLASSES = {1, 3}   # 1=bicycle, 3=motorcycle

# ── Speed calibration (perspective-aware) ────────────────────────────────────
# METRES_PER_PIXEL_BASE: scale at the bottom of frame (nearest objects)
# METRES_PER_PIXEL_TOP : scale at the top of frame  (farthest objects)
# Calibrate: find a lane/object of known real width in your video,
# measure pixel width at bottom → MPP_BASE = real_m / px_width
METRES_PER_PIXEL_BASE = 0.05
METRES_PER_PIXEL_TOP  = 0.015

# ── Stationarity detection ───────────────────────────────────────────────────
# Vehicle is considered stationary if compensated displacement is below this
STATIONARY_DISP_PX   = 2.5    # compensated px/frame → treat as stopped
STATIONARY_AREA_VAR  = 0.0008 # bbox area variance threshold (fraction²)
STATIONARY_WINDOW    = 12     # frames to assess area stability

# ── Base event thresholds ────────────────────────────────────────────────────
# (These are dynamically adjusted by FleetContext at runtime)
BASE_HARD_BRAKE_MPS2    = 2.5
BASE_AGGR_ACCEL_MPS2    = 2.0
LANE_WEAVE_PX           = 16
TAILGATE_AREA_THRESH    = 0.09
TAILGATE_GROWTH_RATE    = 0.002
SUDDEN_STOP_SPEED_MPS   = 1.2
SUDDEN_STOP_FRAMES      = 6
SPEED_LIMIT_MPS         = 13.9   # 50 km/h

# ── Fleet-context scaling factor ─────────────────────────────────────────────
# When fleet median decel = X m/s², effective brake threshold += FLEET_K * X
# Set to 0.0 to disable traffic-awareness
FLEET_K = 0.6

# ── Score weights ─────────────────────────────────────────────────────────────
W_HARD_BRAKE   = 8
W_AGGR_ACCEL   = 5
W_WEAVE        = 3
W_TAILGATE     = 6
W_SUDDEN_STOP  = 7
W_OVER_SPEED   = 0.25

SCORE_EMA_ALPHA = 0.10    # displayed score smoothing (lower = smoother)

# ── Tracker / flow ────────────────────────────────────────────────────────────
IOU_MATCH_THRESH = 0.20
MAX_DISAPPEARED  = 35
EVENT_COOLDOWN   = 12
WARMUP_FRAMES    = 8

OF_MAX_CORNERS = 200
OF_QUALITY     = 0.01
OF_MIN_DIST    = 6
OF_WIN_SIZE    = (21, 21)

# ═══════════════════════════ UTILITIES ══════════════════════════════════════ #

def compute_iou(a, b):
    xA = max(a[0],b[0]); yA = max(a[1],b[1])
    xB = min(a[2],b[2]); yB = min(a[3],b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / float(ua + ub - inter)

def perspective_mpp(cy):
    """Metres per pixel at frame row cy (linear interpolation)."""
    t = 1.0 - cy / FRAME_H
    return METRES_PER_PIXEL_BASE + t*(METRES_PER_PIXEL_TOP - METRES_PER_PIXEL_BASE)

# ═══════════════════════════ EGO-MOTION ESTIMATOR ═══════════════════════════ #

class EgoMotionEstimator:
    """
    Sparse Lucas-Kanade optical flow on background regions.
    Returns (ego_dx, ego_dy): the camera's own pixel translation this frame.

    Background mask excludes the road centre to avoid vehicles polluting
    the ego estimate. Only sky-edge and road-edge features are used.

    Per-vehicle depth correction:
        expected_shift_x(vehicle) = ego_dx * depth_factor(vehicle.cy)
    where depth_factor scales from 1.0 at the bottom to ~0.3 at the top,
    because far objects shift less in pixels for the same camera motion.
    """
    def __init__(self):
        self._prev_gray = None
        self._prev_pts  = None
        self.ego_dx     = 0.0
        self.ego_dy     = 0.0

    @staticmethod
    def depth_factor(cy):
        """
        Objects at cy=FRAME_H are close (depth_factor=1.0).
        Objects at cy=0 are far (depth_factor~0.25).
        Approximates 1/depth ∝ perspective scale.
        """
        norm = cy / FRAME_H          # 0 (top) → 1 (bottom)
        return 0.25 + 0.75 * norm    # 0.25 … 1.0

    def _bg_mask(self, h, w):
        mask = np.ones((h, w), np.uint8) * 255
        # exclude central-bottom road area (where vehicles are)
        cv2.rectangle(mask, (w//5, h//2), (4*w//5, h), 0, -1)
        return mask

    def update(self, gray):
        h, w = gray.shape
        if self._prev_gray is None or self._prev_pts is None \
                or len(self._prev_pts) < 10:
            mask = self._bg_mask(h, w)
            self._prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=OF_MAX_CORNERS,
                qualityLevel=OF_QUALITY, minDistance=OF_MIN_DIST, mask=mask)
            self._prev_gray = gray.copy()
            return 0.0, 0.0

        npts, st, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=OF_WIN_SIZE, maxLevel=3)

        good = st.ravel() == 1
        if good.sum() < 5:
            self._prev_pts  = None
            self._prev_gray = gray.copy()
            return self.ego_dx, self.ego_dy

        flow = npts[good] - self._prev_pts[good]
        # Reject outliers (moving objects leaked through mask)
        dx_vals = flow[:, 0, 0]; dy_vals = flow[:, 0, 1]
        med_dx  = float(np.median(dx_vals))
        med_dy  = float(np.median(dy_vals))
        keep    = np.abs(dx_vals - med_dx) < 8
        self.ego_dx = float(np.mean(dx_vals[keep])) if keep.any() else med_dx
        self.ego_dy = float(np.mean(dy_vals[keep])) if keep.any() else med_dy

        self._prev_pts  = npts[good].reshape(-1,1,2)
        self._prev_gray = gray.copy()
        if len(self._prev_pts) < 25:
            self._prev_pts = None
        return self.ego_dx, self.ego_dy

# ═══════════════════════════ FLEET CONTEXT ══════════════════════════════════ #

class FleetContext:
    """
    Aggregates all active riders' kinematics each frame to compute
    fleet-wide median speed and acceleration.

    Usage: call update(behaviors) each frame BEFORE updating individual
    riders. Then pass the context to each rider's update() so it can
    dynamically adjust its event thresholds.

    Effect on scoring:
    ─────────────────
    If fleet_median_accel = -2 m/s² (everyone braking in traffic),
    the effective hard-brake threshold becomes:
        BASE_HARD_BRAKE_MPS2 + FLEET_K * 2.0
    so only riders braking MUCH harder than traffic get penalised.
    Same logic for acceleration (congestion release).
    """
    def __init__(self):
        self.fleet_speed_mps  = 0.0
        self.fleet_accel_mps2 = 0.0
        self.n_vehicles       = 0

    def update(self, behaviors: dict):
        speeds = [b.speed_mps for b in behaviors.values() if b._frame_n >= WARMUP_FRAMES]
        accels = [b.fast_accel for b in behaviors.values() if b._frame_n >= WARMUP_FRAMES]
        self.n_vehicles = len(speeds)
        self.fleet_speed_mps  = float(np.median(speeds)) if speeds else 0.0
        self.fleet_accel_mps2 = float(np.median(accels)) if accels else 0.0

    def effective_brake_thr(self):
        """Hard-brake threshold raised when whole fleet is decelerating."""
        fleet_decel = max(0.0, -self.fleet_accel_mps2)
        return BASE_HARD_BRAKE_MPS2 + FLEET_K * fleet_decel

    def effective_accel_thr(self):
        """Accel threshold raised when whole fleet is accelerating."""
        fleet_acc = max(0.0, self.fleet_accel_mps2)
        return BASE_AGGR_ACCEL_MPS2 + FLEET_K * fleet_acc

# ═══════════════════════════ IoU TRACKER ════════════════════════════════════ #

class IoUTracker:
    def __init__(self):
        self.next_id     = 0
        self.tracks      = {}
        self.disappeared = defaultdict(int)

    @staticmethod
    def _cen(box):
        return (box[0]+box[2])//2, (box[1]+box[3])//2

    def _register(self, box):
        cx,cy = self._cen(box)
        self.tracks[self.next_id] = {'box':box,'cx':cx,'cy':cy}
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def _deregister(self, tid):
        self.tracks.pop(tid,None); self.disappeared.pop(tid,None)

    @property
    def active_tracks(self):
        """Only tracks matched to a detection THIS frame (disappeared == 0).
        Coasting tracks (disappeared > 0) are kept alive internally for
        re-identification but must NEVER be rendered — that causes ghost boxes."""
        return {tid: td for tid, td in self.tracks.items()
                if self.disappeared[tid] == 0}

    def update(self, det_boxes):
        if not det_boxes:
            for tid in list(self.disappeared):
                self.disappeared[tid] += 1
                if self.disappeared[tid] > MAX_DISAPPEARED:
                    self._deregister(tid)
            return self.tracks

        if not self.tracks:
            for b in det_boxes: self._register(b)
            return self.tracks

        tids   = list(self.tracks.keys())
        tboxes = [self.tracks[t]['box'] for t in tids]

        iou_mat = np.array([[compute_iou(db,tb) for tb in tboxes]
                             for db in det_boxes])

        matched_d = set(); matched_t = set()
        for idx in np.argsort(-iou_mat, axis=None):
            di,ti = divmod(int(idx), len(tboxes))
            if di in matched_d or ti in matched_t: continue
            if iou_mat[di,ti] < IOU_MATCH_THRESH: break
            tid = tids[ti]; box = det_boxes[di]; cx,cy = self._cen(box)
            self.tracks[tid] = {'box':box,'cx':cx,'cy':cy}
            self.disappeared[tid] = 0
            matched_d.add(di); matched_t.add(ti)

        unm_d = [i for i in range(len(det_boxes)) if i not in matched_d]
        unm_t = [i for i in range(len(tboxes))    if i not in matched_t]
        mxd   = math.hypot(FRAME_W, FRAME_H)/4

        for di in unm_d:
            dcx,dcy = self._cen(det_boxes[di])
            bd,bti  = float('inf'), None
            for ti in unm_t:
                d = math.hypot(dcx-self._cen(tboxes[ti])[0],
                               dcy-self._cen(tboxes[ti])[1])
                if d < bd: bd,bti = d,ti
            if bti is not None and bd < mxd:
                tid=tids[bti]; box=det_boxes[di]; cx,cy=self._cen(box)
                self.tracks[tid]={'box':box,'cx':cx,'cy':cy}
                self.disappeared[tid]=0; unm_t.remove(bti)
            else:
                self._register(det_boxes[di])

        for ti in unm_t:
            tid=tids[ti]; self.disappeared[tid]+=1
            if self.disappeared[tid]>MAX_DISAPPEARED: self._deregister(tid)

        return self.tracks

# ═══════════════════════════ RIDER BEHAVIOR ══════════════════════════════════ #

class RiderBehavior:
    """
    Per-rider kinematic tracker and score engine.

    Key design points
    ─────────────────
    • Stationarity check: if compensated displacement < STATIONARY_DISP_PX
      OR bbox area is stable over STATIONARY_WINDOW frames → speed = 0.
    • Fast-path accel (self.fast_accel): raw 2-frame delta, no smoothing.
      Used ONLY for event detection so brakes/accels are caught in 1 frame.
    • Smooth accel (self.accel_mps2): EMA, shown on HUD.
    • Fleet context: event thresholds are passed in each frame from
      FleetContext so traffic conditions raise the bar before penalising.
    • Camera mode handled outside; behavior engine receives already-
      compensated (comp_dx, comp_dy) so it is camera-agnostic.
    """

    def __init__(self, rider_id, fps):
        self.rider_id  = rider_id
        self.fps       = fps
        self.dt        = 1.0 / fps
        self._frame_n  = 0

        # Smoothed position
        self._scx = None; self._scy = None
        self._pos_hist: deque = deque(maxlen=25)

        # Speed / accel
        self.speed_mps      = 0.0
        self.fast_accel     = 0.0    # 2-frame raw → event detection
        self.accel_mps2     = 0.0    # EMA smoothed → display
        self._prev_speed    = 0.0
        self._spd_hist: deque = deque(maxlen=10)

        # Stationarity
        self._area_hist: deque = deque(maxlen=STATIONARY_WINDOW)
        self.is_stationary    = False

        # Lateral
        self._lat_hist: deque = deque(maxlen=8)

        # Events
        self.hard_brake_count  = 0
        self.aggr_accel_count  = 0
        self.weave_count       = 0
        self.tailgate_count    = 0
        self.sudden_stop_count = 0
        self.over_speed_sec    = 0.0

        self._cd = defaultdict(int)

        # Score
        self.score  = 100.0
        self._raw   = 100.0

        # Wheelie stub
        self.is_doing_stunt = False
        self._ar_hist: deque = deque(maxlen=10)  # aspect-ratio history

    # ── helpers ──────────────────────────────────────────────────────────────

    def _ema_pos(self, cx, cy, alpha=0.22):
        if self._scx is None: self._scx,self._scy = float(cx),float(cy)
        else:
            self._scx = alpha*cx + (1-alpha)*self._scx
            self._scy = alpha*cy + (1-alpha)*self._scy

    def _tick_cd(self):
        for k in list(self._cd): self._cd[k] = max(0, self._cd[k]-1)

    def _fire(self, key, attr):
        if self._cd[key] == 0:
            setattr(self, attr, getattr(self,attr)+1)
            self._cd[key] = EVENT_COOLDOWN

    def _check_stationary(self, comp_dx, comp_dy):
        """True if vehicle appears to be standing still."""
        # Method 1: compensated displacement is tiny
        disp = math.hypot(comp_dx, comp_dy)
        if disp < STATIONARY_DISP_PX:
            return True
        # Method 2: bbox area barely changes (not approaching / receding)
        if len(self._area_hist) >= STATIONARY_WINDOW:
            var = float(np.var(list(self._area_hist)))
            if var < STATIONARY_AREA_VAR:
                return True
        return False

    # ── main update ──────────────────────────────────────────────────────────

    def update(self, cx, cy, box, comp_dx, comp_dy, fleet: FleetContext):
        self._frame_n += 1
        self._tick_cd()
        self._ema_pos(cx, cy)
        self._pos_hist.append((self._scx, self._scy))

        # Bbox metrics
        bw = box[2]-box[0]; bh = box[3]-box[1]
        af = (bw*bh) / (FRAME_W*FRAME_H)
        ar = bh / max(bw, 1)           # aspect ratio
        self._area_hist.append(af)
        self._ar_hist.append(ar)

        # ── Stationarity check ────────────────────────────────────────────
        self.is_stationary = self._check_stationary(comp_dx, comp_dy)

        if self.is_stationary:
            raw_spd = 0.0
        else:
            mpp     = perspective_mpp(self._scy)
            raw_spd = math.hypot(comp_dx, comp_dy) * mpp * self.fps

        # ── Speed: median window + gentle EMA ────────────────────────────
        self._spd_hist.append(raw_spd)
        med_spd = float(np.median(list(self._spd_hist)))

        # Fast speed (short EMA, α=0.35) used for fast_accel
        fast_spd       = 0.35*raw_spd + 0.65*self._prev_speed
        self.fast_accel = (fast_spd - self._prev_speed) / self.dt

        # Smooth speed for display
        self.speed_mps  = 0.15*med_spd + 0.85*self._prev_speed
        self.accel_mps2 = (self.speed_mps - self._prev_speed) / self.dt
        self._prev_speed = self.speed_mps

        # Lateral
        self._lat_hist.append(abs(comp_dx))
        lat = float(np.median(list(self._lat_hist)))

        # ── Wheelie stub ──────────────────────────────────────────────────
        if len(self._ar_hist) >= 5:
            ar_mean = float(np.mean(list(self._ar_hist)))
            ar_var  = float(np.var(list(self._ar_hist)))
            # Wheelie: aspect ratio spikes AND high variance
            self.is_doing_stunt = ar_mean > 2.8 and ar_var > 0.15
            if self.is_doing_stunt:
                self.score = 0.0; return

        # ── Skip events during warm-up ────────────────────────────────────
        if self._frame_n < WARMUP_FRAMES:
            self._recompute(); return

        # ── Dynamic thresholds from fleet context ─────────────────────────
        brake_thr = fleet.effective_brake_thr()
        accel_thr = fleet.effective_accel_thr()

        # ── Event detection (using fast_accel for immediacy) ──────────────
        if self.fast_accel < -brake_thr:
            self._fire('brake', 'hard_brake_count')

        if self.fast_accel > accel_thr:
            self._fire('accel', 'aggr_accel_count')

        if lat > LANE_WEAVE_PX:
            self._fire('weave', 'weave_count')

        if (len(self._area_hist) >= 5 and af > TAILGATE_AREA_THRESH
                and self._area_hist[-1] - self._area_hist[-5] > TAILGATE_GROWTH_RATE):
            self._fire('tail', 'tailgate_count')

        if (len(self._spd_hist) >= SUDDEN_STOP_FRAMES and not self.is_stationary
                and self._spd_hist[-SUDDEN_STOP_FRAMES] > SUDDEN_STOP_SPEED_MPS
                and self.speed_mps < SUDDEN_STOP_SPEED_MPS):
            self._fire('stop', 'sudden_stop_count')

        if self.speed_mps > SPEED_LIMIT_MPS:
            self.over_speed_sec += self.dt

        self._recompute()

    def _recompute(self):
        p = (self.hard_brake_count  * W_HARD_BRAKE  +
             self.aggr_accel_count  * W_AGGR_ACCEL  +
             self.weave_count       * W_WEAVE        +
             self.tailgate_count    * W_TAILGATE     +
             self.sudden_stop_count * W_SUDDEN_STOP  +
             self.over_speed_sec    * W_OVER_SPEED)
        self._raw  = max(0.0, min(100.0, 100.0 - p))
        self.score = SCORE_EMA_ALPHA*self._raw + (1-SCORE_EMA_ALPHA)*self.score

    def speed_kmh(self):   return self.speed_mps * 3.6
    def total_events(self):
        return (self.hard_brake_count + self.aggr_accel_count +
                self.weave_count + self.tailgate_count + self.sudden_stop_count)
    def score_label(self):
        if self.is_doing_stunt: return "STUNT"
        return "GOOD" if self.score>=80 else ("FAIR" if self.score>=50 else "POOR")
    def score_color(self):
        if self.is_doing_stunt: return (0,0,255)
        return ((30,200,30) if self.score>=80 else
                ((0,165,255) if self.score>=50 else (40,40,220)))

# ═══════════════════════════ HUD ════════════════════════════════════════════ #

def draw_hud(frame, rider: RiderBehavior, box):
    x1,y1,x2,y2 = box
    color = rider.score_color()
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    # Score bar
    bw   = max(x2-x1, 1)
    fill = int(bw * rider.score/100)
    cv2.rectangle(frame,(x1,y1),(x2,y1+5),(40,40,40),-1)
    cv2.rectangle(frame,(x1,y1),(x1+fill,y1+5),color,-1)

    stat_tag = " [STILL]" if rider.is_stationary else ""
    stunt_tag= " ⚡STUNT!" if rider.is_doing_stunt else ""
    lines = [
        (f"#{rider.rider_id} {rider.score:.0f}/100"
         f" [{rider.score_label()}]{stat_tag}{stunt_tag}", color),
        (f"spd {rider.speed_kmh():.1f}km/h "
         f"fa={rider.fast_accel:+.1f}m/s²", (230,230,230)),
        (f"brk:{rider.hard_brake_count} wv:{rider.weave_count}"
         f" tg:{rider.tailgate_count} stp:{rider.sudden_stop_count}",
         (180,180,180)),
    ]
    ty = max(y1 - len(lines)*14 - 4, 0)
    for i,(txt,clr) in enumerate(lines):
        cv2.putText(frame,txt,(x1,ty+i*14),
                    cv2.FONT_HERSHEY_SIMPLEX,0.37,clr,1,cv2.LINE_AA)


def draw_fleet_bar(frame, fleet: FleetContext):
    """Top-centre: fleet-wide context."""
    txt = (f"Fleet  spd:{fleet.fleet_speed_mps*3.6:.1f}km/h "
           f"accel:{fleet.fleet_accel_mps2:+.1f}m/s²  "
           f"brake_thr:{fleet.effective_brake_thr():.1f}  "
           f"n={fleet.n_vehicles}")
    ov = frame.copy()
    tw,_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0], 0
    cv2.rectangle(ov,(0,0),(FRAME_W,18),(20,20,60),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    cv2.putText(frame,txt,(4,13),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(200,220,255),1,cv2.LINE_AA)


def draw_ego_arrow(frame, dx, dy):
    if CAMERA_MODE != "dashcam": return
    cx,cy = FRAME_W//2, 32
    cv2.arrowedLine(frame,(cx,cy),(int(cx+dx*3),int(cy+dy*3)),
                    (0,210,255),2,tipLength=0.4)
    cv2.putText(frame,f"cam({dx:+.1f},{dy:+.1f})",
                (cx-45,cy-5),cv2.FONT_HERSHEY_SIMPLEX,0.32,(0,210,255),1)


def draw_panel(frame, behaviors):
    if not behaviors: return
    n  = len(behaviors)
    ov = frame.copy()
    cv2.rectangle(ov,(0,20),(260,20+n*20+18),(10,10,10),-1)
    cv2.addWeighted(ov,0.65,frame,0.35,0,frame)
    cv2.putText(frame,"ID   Scr  km/h  Ev  Tg  St",(4,34),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(220,220,0),1)
    for i,(rid,b) in enumerate(sorted(behaviors.items())):
        t = (f"#{rid:<3}{b.score:4.0f} {b.speed_kmh():5.1f}"
             f"{b.total_events():3}"
             f"{'  TG' if b.tailgate_count else '    '}"
             f"{'  ST' if b.is_doing_stunt else '    '}")
        cv2.putText(frame,t,(4,50+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,b.score_color(),1,cv2.LINE_AA)


def draw_footer(frame, n):
    ov = frame.copy()
    cv2.rectangle(ov,(0,FRAME_H-20),(FRAME_W,FRAME_H),(15,15,15),-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)
    mode_str = f"[{CAMERA_MODE.upper()}]"
    cv2.putText(frame,
                f"Varroc Eureka PS3 v4 {mode_str} | Frame {n}",
                (6,FRAME_H-6),cv2.FONT_HERSHEY_SIMPLEX,0.33,(160,160,160),1)

# ═══════════════════════════ MAIN ═══════════════════════════════════════════ #

def main():
    model = YOLO(MODEL_PATH)
    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    print(f"[INFO] FPS: {fps:.1f}  Mode: {CAMERA_MODE}")

    tracker  = IoUTracker()
    ego_est  = EgoMotionEstimator() if CAMERA_MODE=="dashcam" else None
    fleet    = FleetContext()
    behaviors: dict[int, RiderBehavior] = {}
    # Permanent record of every rider ever tracked — never deleted,
    # so the final report covers all vehicles seen during the session.
    all_time: dict[int, RiderBehavior] = {}
    n = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        n += 1
        frame = cv2.resize(frame,(FRAME_W,FRAME_H))
        gray  = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # ── Ego-motion (dashcam only) ──────────────────────────────────────
        if ego_est:
            raw_edx, raw_edy = ego_est.update(gray)
        else:
            raw_edx, raw_edy = 0.0, 0.0

        # ── YOLO detection ─────────────────────────────────────────────────
        results   = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        det_boxes = [tuple(map(int,b.xyxy[0]))
                     for b in results.boxes
                     if int(b.cls[0]) in TWO_WHEELER_CLASSES]

        # ── Track update ───────────────────────────────────────────────────
        tracker.update(det_boxes)

        # active_tracks: only boxes matched to a detection this frame.
        # tracker.tracks still holds coasting tracks for re-identification,
        # but we never render or score those — that is what caused ghost boxes.
        active = tracker.active_tracks

        # ── Fleet context (before individual updates) ──────────────────────
        fleet.update(behaviors)

        # ── Per-rider behavior update (active detections only) ─────────────
        for tid, td in active.items():
            if tid not in behaviors:
                behaviors[tid] = RiderBehavior(tid, fps)
            b = behaviors[tid]

            if ego_est:
                df    = EgoMotionEstimator.depth_factor(td['cy'])
                edx_v = raw_edx * df
                edy_v = raw_edy * df
            else:
                edx_v = edy_v = 0.0

            cx, cy = td['cx'], td['cy']
            if b._scx is not None:
                raw_dx = cx - b._scx
                raw_dy = cy - b._scy
            else:
                raw_dx = raw_dy = 0.0
            comp_dx = raw_dx - edx_v
            comp_dy = raw_dy - edy_v

            b.update(cx, cy, td['box'], comp_dx, comp_dy, fleet)

            # Mirror into permanent archive (same object, so updates are live)
            all_time[tid] = b

        # Remove behavior records for tracks fully deregistered by the tracker
        alive_ids = set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive_ids]:
            del behaviors[k]

        # ── Render (active detections only — no ghost boxes) ───────────────
        for tid, td in active.items():
            if tid in behaviors:
                draw_hud(frame, behaviors[tid], td['box'])

        draw_fleet_bar(frame, fleet)
        draw_ego_arrow(frame, raw_edx, raw_edy)
        # Pass only active behaviors so the panel doesn't show ghost entries
        active_behaviors = {tid: behaviors[tid] for tid in active if tid in behaviors}
        draw_panel(frame, active_behaviors)
        draw_footer(frame, n)

        cv2.imshow("Varroc Eureka PS3 v4  [ESC to quit]", frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    # ── Final report — ALL vehicles seen during the session ───────────────
    print("\n" + "="*75)
    print("  VARROC EUREKA PS3 — FINAL DRIVING BEHAVIOR REPORT  v4")
    print(f"  Total vehicles tracked: {len(all_time)}  |  Frames processed: {n}")
    print("="*75)
    print(f"{'ID':<5}{'Score':>7}{'km/h':>7}{'Brakes':>8}{'Accel':>7}"
          f"{'Weave':>7}{'Tailg':>7}{'Stop':>6}{'Stunt':>6}{'Rating':>8}")
    print("-"*75)
    for rid, b in sorted(all_time.items()):
        print(f"#{rid:<4}{b.score:7.1f}{b.speed_kmh():7.1f}"
              f"{b.hard_brake_count:8}{b.aggr_accel_count:7}"
              f"{b.weave_count:7}{b.tailgate_count:7}"
              f"{b.sudden_stop_count:6}"
              f"{'  YES' if b.is_doing_stunt else '   no':>6}"
              f"  [{b.score_label():>5}]")
    print("-"*75)
    # Summary stats across all vehicles
    if all_time:
        scores = [b.score for b in all_time.values()]
        print(f"\n  Avg score : {sum(scores)/len(scores):.1f}")
        print(f"  Best      : #{min(all_time, key=lambda k: -all_time[k].score)}"
              f"  ({max(scores):.1f})")
        print(f"  Worst     : #{min(all_time, key=lambda k: all_time[k].score)}"
              f"  ({min(scores):.1f})")
        good  = sum(1 for b in all_time.values() if b.score >= 80)
        fair  = sum(1 for b in all_time.values() if 50 <= b.score < 80)
        poor  = sum(1 for b in all_time.values() if b.score < 50)
        print(f"  GOOD: {good}  FAIR: {fair}  POOR: {poor}")
    print("="*75)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()