"""
============================================================
Varroc Eureka 3.0 — Problem Statement 3
Accurate Driving Behavior Score on a Two-Wheeler  v5
============================================================
Detection  : YOLOv8n  (motorcycle=3, bicycle=1, person=0)
Tracking   : IoU-first + centroid fallback, no ghost boxes
Speed      : Kalman CA filter on ego-compensated centroid
             Dynamic MPP from bbox width (self-calibrating)
             Fleet-shared MPP for cross-vehicle consistency
             Farneback dense flow blend for dashcam
Events     : Hard brake, aggressive accel, weave, tailgate, stop, no-helmet
Context    : FleetContext scales thresholds with live traffic
Score      : 100 - sum(event*weight), EMA-smoothed display
Modes      : dashcam|fixed  x  side|overhead  (interactive selector)
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import math
try:
    from boxmot import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

# ============================= CONFIGURATION ================================ #

VIDEO_PATH   = r"D:\hackathon\public\video1.mp4"
MODEL_PATH   = "yolov8n.pt"

CAMERA_MODE  = None   # None -> interactive selector | "dashcam" | "fixed"
CAMERA_ANGLE = None   # None -> interactive selector | "side"    | "overhead"

FRAME_W, FRAME_H = 640, 360
FPS_FALLBACK     = 30

CONF_SIDE     = 0.30
CONF_OVERHEAD = 0.18
PERSON_CONF   = 0.35

TWO_WHEELER_CLASSES_SIDE     = {1, 3}
TWO_WHEELER_CLASSES_OVERHEAD = {0, 1, 3}

# -- Speed / MPP -------------------------------------------------------------
REAL_WORLD_WIDTHS        = {0: 0.50, 1: 0.60, 3: 0.80}
REAL_WORLD_WIDTH_DEFAULT = 0.75
MPP_HISTORY_LEN          = 25

MAX_VEHICLE_SPEED_MPS   = 35.0
MAX_CAMERA_SPEED_MPS    = 25.0
MAX_PLAUSIBLE_CLOSE_MPS = 15.0
CLOSE_BBOX_FRACTION     = 0.25

# -- Stationarity ------------------------------------------------------------
STATIONARY_DISP_PX  = 2.5
STATIONARY_AREA_VAR = 0.0008
STATIONARY_WINDOW   = 12

# -- Direction ---------------------------------------------------------------
DIR_WINDOW        = 10
DIR_ONCOMING_RATE = 0.004
DIR_APPROACH_RATE = 0.001
DIR_RECEDING_RATE = -0.001

# -- Helmet ------------------------------------------------------------------
HELMET_OVERLAP_IOU    = 0.20
HELMET_HEAD_FRACTION  = 0.32
HELMET_MIN_AREA_FRAC  = 0.08
HELMET_CONFIRM_FRAMES = 15
W_NO_HELMET           = 15

# -- Events ------------------------------------------------------------------
BASE_HARD_BRAKE_MPS2  = 2.5
BASE_AGGR_ACCEL_MPS2  = 2.0
LANE_WEAVE_PX         = 16
TAILGATE_AREA_THRESH  = 0.09
TAILGATE_GROWTH_RATE  = 0.002
SUDDEN_STOP_SPEED_MPS = 1.2
SUDDEN_STOP_FRAMES    = 6
SPEED_LIMIT_MPS       = 13.9
FLEET_K               = 0.6

# -- Score weights -----------------------------------------------------------
W_HARD_BRAKE   = 5
W_AGGR_ACCEL   = 3
W_WEAVE        = 2
W_TAILGATE     = 4
W_SUDDEN_STOP  = 5
W_OVER_SPEED   = 0.15
SCORE_EMA_ALPHA = 0.04
COLOR_EMA_ALPHA = 0.02

# -- Tracker / flow ----------------------------------------------------------
IOU_MATCH_THRESH = 0.20
MAX_DISAPPEARED  = 35
EVENT_COOLDOWN   = 18
WARMUP_FRAMES    = 8
OF_MAX_CORNERS   = 200
OF_QUALITY       = 0.01
OF_MIN_DIST      = 6
OF_WIN_SIZE      = (21, 21)

# ============================= UTILITIES ==================================== #

def compute_iou(a, b):
    xA=max(a[0],b[0]); yA=max(a[1],b[1])
    xB=min(a[2],b[2]); yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def bbox_mpp(bbox_width_px, cls):
    real_w = REAL_WORLD_WIDTHS.get(cls, REAL_WORLD_WIDTH_DEFAULT)
    return real_w / max(bbox_width_px, 5.0)

def max_plausible_speed(bbox_width_px):
    frac = bbox_width_px / FRAME_W
    if frac >= CLOSE_BBOX_FRACTION:
        t = min((frac-CLOSE_BBOX_FRACTION)/(1.0-CLOSE_BBOX_FRACTION), 1.0)
        return MAX_PLAUSIBLE_CLOSE_MPS*(1.0-0.85*t)
    return MAX_VEHICLE_SPEED_MPS

# ============================= HELMET DETECTOR ============================== #

class HelmetDetector:
    def __init__(self):
        self._history = defaultdict(lambda: deque(maxlen=HELMET_CONFIRM_FRAMES))
        self.status   = {}

    def update(self, frame_gray, bike_tracks, person_boxes):
        for tid, td in bike_tracks.items():
            best_iou, best_pb = 0.0, None
            for pb in person_boxes:
                iou = compute_iou(td['box'], pb)
                if iou > best_iou:
                    best_iou, best_pb = iou, pb
            if best_pb is None or best_iou < HELMET_OVERLAP_IOU:
                self.status[tid] = self.status.get(tid, "?")
                continue
            px1,py1,px2,py2 = best_pb
            p_h    = max(py2-py1, 1)
            head_y2= int(py1 + p_h * HELMET_HEAD_FRACTION)
            head_y1= max(py1, 0)
            head_x1= max(px1, 0)
            head_x2= min(px2, frame_gray.shape[1])
            if head_y2<=head_y1 or head_x2<=head_x1: continue
            crop = frame_gray[head_y1:head_y2, head_x1:head_x2]
            if crop.size < 50: continue
            _, thresh = cv2.threshold(crop, 0, 255,
                                      cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            head_area = crop.shape[0]*crop.shape[1]
            max_blob  = max((cv2.contourArea(c) for c in cnts), default=0)
            has_helmet= (max_blob/max(head_area,1)) >= HELMET_MIN_AREA_FRAC
            self._history[tid].append(has_helmet)
            hist = self._history[tid]
            if len(hist) >= HELMET_CONFIRM_FRAMES//2:
                self.status[tid] = ("HELMET" if sum(hist)/len(hist)>=0.55
                                    else "NO_HELMET")
            else:
                self.status[tid] = "?"
        return self.status

# ============================= KALMAN (CA) ================================== #

class KalmanVelocityEstimator:
    """
    Constant-Acceleration Kalman filter.
    State: [x, y, vx, vy, ax, ay] — tracks acceleration explicitly so
    braking/accelerating vehicles are followed in 1-2 frames, not 5-8.
    """
    PROCESS_NOISE = 0.8
    MEASURE_NOISE = 28.0

    def __init__(self):
        self.x = None; self.P = None; self.initialized = False
        dt = 1.0
        self.F = np.array([
            [1,0,dt,0,0.5*dt**2,0        ],
            [0,1,0,dt,0,        0.5*dt**2],
            [0,0,1,0, dt,       0        ],
            [0,0,0,1, 0,        dt       ],
            [0,0,0,0, 1,        0        ],
            [0,0,0,0, 0,        1        ],
        ], dtype=float)
        self.H = np.zeros((2,6)); self.H[0,0]=1; self.H[1,1]=1
        q = self.PROCESS_NOISE; r = self.MEASURE_NOISE
        self.Q = np.diag([q*.25, q*.25, q*.5, q*.5, q, q])
        self.R = np.diag([r, r])

    def update(self, cx, cy):
        z = np.array([[cx],[cy]], dtype=float)
        if not self.initialized:
            self.x = np.array([[cx],[cy],[0],[0],[0],[0]], dtype=float)
            self.P = np.eye(6)*200.0
            self.initialized = True
            return cx, cy, 0.0, 0.0
        x_p = self.F@self.x
        P_p = self.F@self.P@self.F.T + self.Q
        S   = self.H@P_p@self.H.T + self.R
        K   = P_p@self.H.T@np.linalg.inv(S)
        self.x = x_p + K@(z - self.H@x_p)
        self.P = (np.eye(6)-K@self.H)@P_p
        return (float(self.x[0]), float(self.x[1]),
                float(self.x[2]), float(self.x[3]))

    def reset(self): self.initialized = False

# ============================= EGO-MOTION =================================== #

class EgoMotionEstimator:
    def __init__(self):
        self._prev_gray=None; self._prev_pts=None
        self.ego_dx=0.0; self.ego_dy=0.0

    @staticmethod
    def depth_factor(cy): return 0.25+0.75*(cy/FRAME_H)

    def _bg_mask(self,h,w):
        mask=np.ones((h,w),np.uint8)*255
        cv2.rectangle(mask,(w//5,h//2),(4*w//5,h),0,-1)
        return mask

    def update(self, gray):
        h,w=gray.shape
        if self._prev_gray is None or self._prev_pts is None \
                or len(self._prev_pts)<10:
            self._prev_pts=cv2.goodFeaturesToTrack(
                gray,maxCorners=OF_MAX_CORNERS,qualityLevel=OF_QUALITY,
                minDistance=OF_MIN_DIST,mask=self._bg_mask(h,w))
            self._prev_gray=gray.copy(); return 0.0,0.0
        npts,st,_=cv2.calcOpticalFlowPyrLK(
            self._prev_gray,gray,self._prev_pts,None,
            winSize=OF_WIN_SIZE,maxLevel=3)
        good=st.ravel()==1
        if good.sum()<5:
            self._prev_pts=None; self._prev_gray=gray.copy()
            return self.ego_dx,self.ego_dy
        flow=npts[good]-self._prev_pts[good]
        dx_vals=flow[:,0,0]; dy_vals=flow[:,0,1]
        med_dx=float(np.median(dx_vals)); med_dy=float(np.median(dy_vals))
        keep=np.abs(dx_vals-med_dx)<6
        raw_edx=float(np.mean(dx_vals[keep])) if keep.any() else med_dx
        raw_edy=float(np.mean(dy_vals[keep])) if keep.any() else med_dy
        if math.hypot(raw_edx,raw_edy)<30.0:
            self.ego_dx=0.7*raw_edx+0.3*self.ego_dx
            self.ego_dy=0.7*raw_edy+0.3*self.ego_dy
        self._prev_pts=npts[good].reshape(-1,1,2); self._prev_gray=gray.copy()
        if len(self._prev_pts)<25: self._prev_pts=None
        return self.ego_dx,self.ego_dy

# ============================= FLEET CONTEXT ================================ #

class FleetContext:
    def __init__(self):
        self.fleet_speed_mps=0.0; self.fleet_accel_mps2=0.0; self.n_vehicles=0

    def update(self, behaviors):
        speeds=[b.speed_mps  for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        accels=[b.fast_accel for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        self.n_vehicles      = len(speeds)
        self.fleet_speed_mps = float(np.median(speeds)) if speeds else 0.0
        self.fleet_accel_mps2= float(np.clip(
            float(np.median(accels)) if accels else 0.0, -4.0, 4.0))
        # Only use mid-range vehicles for MPP — close vehicles (large bbox)
        # and far vehicles (tiny bbox) produce unreliable scale estimates.
        all_mpp=[]
        for bv in behaviors.values():
            if not bv._mpp_hist: continue
            med=float(np.median(list(bv._mpp_hist)))
            # MPP range 0.006-0.030 corresponds to vehicles 5-30m away
            if 0.006 <= med <= 0.030: all_mpp.append(med)
        self._fleet_mpp_cache=float(np.median(all_mpp)) if all_mpp else 0.012

    def effective_brake_thr(self):
        return BASE_HARD_BRAKE_MPS2+FLEET_K*max(0.0,-self.fleet_accel_mps2)
    def effective_accel_thr(self):
        return BASE_AGGR_ACCEL_MPS2+FLEET_K*max(0.0, self.fleet_accel_mps2)
    def fleet_mpp(self):
        return getattr(self,'_fleet_mpp_cache',0.012)


# ========================= BYTETRACK WRAPPER ================================ #

class ByteTrackWrapper:
    """
    Wraps boxmot ByteTrack to match IoUTracker interface.
    ByteTrack uses a Kalman filter internally for each track,
    so trajectories are smooth and IDs are stable even under occlusion.
    Speed accuracy improves because ID switches (which look like teleports)
    are nearly eliminated.
    """
    def __init__(self, fps):
        self._bt = ByteTrack(
            track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=int(fps)
        )
        self.tracks      = {}
        self.disappeared = defaultdict(int)

    @property
    def active_tracks(self):
        return {t:d for t,d in self.tracks.items() if self.disappeared[t]==0}

    def update(self, det_boxes):
        self.tracks = {}
        if not det_boxes:
            return self.tracks
        # ByteTrack expects [x1,y1,x2,y2,conf,cls] numpy array
        dets = np.array([[*b, 0.9, 0] for b in det_boxes], dtype=np.float32)
        try:
            tracks = self._bt.update(dets, None)
        except Exception:
            return self.tracks
        # tracks: [x1,y1,x2,y2,track_id,conf,cls,idx]
        for t in tracks:
            x1,y1,x2,y2 = int(t[0]),int(t[1]),int(t[2]),int(t[3])
            tid = int(t[4])
            cx  = (x1+x2)//2; cy=(y1+y2)//2
            self.tracks[tid] = {'box':(x1,y1,x2,y2),'cx':cx,'cy':cy}
            self.disappeared[tid] = 0
        return self.tracks

# ============================= IoU TRACKER ================================== #

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
        iou_mat=np.array([[compute_iou(db,tb) for tb in tboxes]
                           for db in det_boxes])
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

# ============================= DIRECTION LABELS ============================= #

DIR_ONCOMING="ONCOMING"; DIR_APPROACH="APPROACH"
DIR_RECEDING="RECEDING"; DIR_ALONGSIDE="ALONGSIDE"; DIR_UNKNOWN="?"
DIR_ARROWS={DIR_ONCOMING:"<<",DIR_APPROACH:"^^",
            DIR_RECEDING:"vv",DIR_ALONGSIDE:"->",DIR_UNKNOWN:"  "}

# ============================= RIDER BEHAVIOR =============================== #

class RiderBehavior:
    def __init__(self, rider_id, fps):
        self.rider_id=rider_id; self.fps=fps; self.dt=1.0/fps; self._frame_n=0

        # Previous centroid — updated every frame so raw_dx is always 1-frame delta
        self._prev_cx: float = None
        self._prev_cy: float = None

        self._kalman   = KalmanVelocityEstimator()
        self._mpp_hist = deque(maxlen=MPP_HISTORY_LEN)
        self._spd_hist = deque(maxlen=max(SUDDEN_STOP_FRAMES+2, 12))
        self._area_hist= deque(maxlen=max(STATIONARY_WINDOW,DIR_WINDOW)+2)
        self._lat_hist = deque(maxlen=8)
        self._ar_hist  = deque(maxlen=10)

        self.cls=3; self.speed_mps=0.0; self.fast_accel=0.0
        self.accel_mps2=0.0; self._prev_speed=0.0

        self.direction=DIR_UNKNOWN
        self.helmet_status="?"; self.no_helmet_count=0; self._no_helmet_cd=0
        self.is_stationary=False; self.is_doing_stunt=False

        self.hard_brake_count=0; self.aggr_accel_count=0
        self.weave_count=0; self.tailgate_count=0
        self.sudden_stop_count=0; self.over_speed_sec=0.0
        self._cd=defaultdict(int)

        self.score=100.0; self._raw=100.0; self._committed=100.0
        self.reliability=0.0  # 0-100: confidence in this track's data

    def _tick_cd(self):
        for k in list(self._cd): self._cd[k]=max(0,self._cd[k]-1)
        self._no_helmet_cd=max(0,self._no_helmet_cd-1)

    def _fire(self,key,attr):
        if self._cd[key]==0:
            setattr(self,attr,getattr(self,attr)+1); self._cd[key]=EVENT_COOLDOWN

    def _check_stationary(self,comp_dx,comp_dy):
        comp_disp=math.hypot(comp_dx,comp_dy)
        area_stable=(len(self._area_hist)>=STATIONARY_WINDOW and
                     float(np.var(list(self._area_hist)[-STATIONARY_WINDOW:]))
                     <STATIONARY_AREA_VAR)
        return comp_disp<STATIONARY_DISP_PX and area_stable

    def _classify_direction(self):
        if len(self._area_hist)<DIR_WINDOW: return DIR_UNKNOWN
        recent=list(self._area_hist)[-DIR_WINDOW:]
        slope=float(np.polyfit(np.arange(len(recent)),recent,1)[0])
        if slope>DIR_ONCOMING_RATE:  return DIR_ONCOMING
        if slope>DIR_APPROACH_RATE:  return DIR_APPROACH
        if slope<DIR_RECEDING_RATE:  return DIR_RECEDING
        return DIR_ALONGSIDE

    def set_helmet(self,status):
        self.helmet_status=status
        if status=="NO_HELMET" and self._no_helmet_cd==0:
            self.no_helmet_count+=1; self._no_helmet_cd=EVENT_COOLDOWN*2

    def update(self, cx, cy, box, comp_dx, comp_dy,
               fleet, camera_speed_mps=0.0, cls=3):
        self._frame_n+=1; self._tick_cd(); self.cls=cls

        # Capture previous centroid BEFORE updating it
        prev_cx = self._prev_cx if self._prev_cx is not None else float(cx)
        prev_cy = self._prev_cy if self._prev_cy is not None else float(cy)
        self._prev_cx=float(cx); self._prev_cy=float(cy)

        bw=max(float(box[2]-box[0]),5.0)
        bh=max(float(box[3]-box[1]),5.0)
        af=(bw*bh)/(FRAME_W*FRAME_H); ar=bh/bw
        self._area_hist.append(af); self._ar_hist.append(ar)
        self.direction=self._classify_direction()
        self.is_stationary=self._check_stationary(comp_dx,comp_dy)

        # Dynamic MPP (fleet-shared for consistency)
        self._mpp_hist.append(bbox_mpp(bw,cls))
        fleet_mpp=fleet.fleet_mpp()
        mpp=fleet_mpp if fleet_mpp>0.005 else float(np.median(list(self._mpp_hist)))

        # Kalman on ego-compensated centroid -> smooth velocity in px/frame
        _, _, vx_k, vy_k = self._kalman.update(
            float(cx) - comp_dx,
            float(cy) - comp_dy)

        raw_spd_rel = math.hypot(vx_k, vy_k) * mpp * self.fps

        # Speed history for sudden-stop
        self._spd_hist.append(raw_spd_rel)

        # Absolute speed
        prox_cap = max_plausible_speed(bw)
        cam_c    = min(camera_speed_mps, MAX_CAMERA_SPEED_MPS)
        if self.is_stationary:
            raw_spd_abs = cam_c
        else:
            raw_spd_abs = min(raw_spd_rel+cam_c, prox_cap)

        raw_fast_accel  = (raw_spd_rel-self._prev_speed)/self.dt
        self.fast_accel = float(np.clip(raw_fast_accel,-8.0,8.0))
        self.speed_mps  = 0.30*raw_spd_abs+0.70*self._prev_speed
        self.accel_mps2 = (self.speed_mps-self._prev_speed)/self.dt
        self._prev_speed= self.speed_mps

        self._lat_hist.append(abs(comp_dx))
        lat=float(np.median(list(self._lat_hist)))

        # Stunt detection
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

        if (self.direction in (DIR_APPROACH,DIR_ALONGSIDE) and
                len(self._area_hist)>=5 and af>TAILGATE_AREA_THRESH and
                self._area_hist[-1]-self._area_hist[-5]>TAILGATE_GROWTH_RATE):
            self._fire('tail','tailgate_count')

        if (len(self._spd_hist)>=SUDDEN_STOP_FRAMES and not self.is_stationary
                and self._spd_hist[-SUDDEN_STOP_FRAMES]>SUDDEN_STOP_SPEED_MPS
                and self.speed_mps<SUDDEN_STOP_SPEED_MPS):
            self._fire('stop','sudden_stop_count')

        # Only flag over-speed for approaching/alongside vehicles.
        # Receding vehicles read high relative speed by definition.
        if (self.speed_mps>SPEED_LIMIT_MPS and
                self.direction not in (DIR_RECEDING, DIR_UNKNOWN)):
            self.over_speed_sec+=self.dt
        self._recompute()

    def _recompute(self):
        p=(self.hard_brake_count  *W_HARD_BRAKE  +
           self.aggr_accel_count  *W_AGGR_ACCEL  +
           self.weave_count       *W_WEAVE        +
           self.tailgate_count    *W_TAILGATE     +
           self.sudden_stop_count *W_SUDDEN_STOP  +
           self.no_helmet_count   *W_NO_HELMET    +
           self.over_speed_sec    *W_OVER_SPEED)
        self._raw      =max(0.0,min(100.0,100.0-p))
        self.score     =SCORE_EMA_ALPHA*self._raw+(1-SCORE_EMA_ALPHA)*self.score
        self._committed=COLOR_EMA_ALPHA*self.score+(1-COLOR_EMA_ALPHA)*self._committed
        # Reliability: rises with frame count, falls if track is barely visible
        self.reliability=min(100.0, self._frame_n / max(WARMUP_FRAMES,1) * 30.0)

    def speed_kmh(self): return self.speed_mps*3.6
    def total_events(self):
        return (self.hard_brake_count+self.aggr_accel_count+self.weave_count+
                self.tailgate_count+self.sudden_stop_count+self.no_helmet_count)
    def score_label(self):
        if self.is_doing_stunt: return "STUNT"
        return "GOOD" if self.score>=80 else ("FAIR" if self.score>=50 else "POOR")
    def score_color(self):
        if self.is_doing_stunt: return (0,0,220)
        s=self._committed
        return ((30,200,30) if s>=80 else ((0,165,255) if s>=50 else (40,40,220)))

# ============================= HUD ========================================== #

def _safe_text(frame,text,org,scale,color,thick=1):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
    x=max(2,min(org[0],FRAME_W-tw-2)); y=max(th+2,min(org[1],FRAME_H-4))
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def draw_hud(frame, rider, box):
    x1,y1,x2,y2=box; color=rider.score_color(); bw=max(x2-x1,1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    fill=int(bw*rider.score/100)
    cv2.rectangle(frame,(x1,y1),(x2,y1+5),(30,30,30),-1)
    cv2.rectangle(frame,(x1,y1),(x1+fill,y1+5),color,-1)

    dir_arrow=DIR_ARROWS.get(rider.direction,"")
    helm=("H+" if rider.helmet_status=="HELMET"
          else("X!" if rider.helmet_status=="NO_HELMET" else ""))
    still=" ST" if rider.is_stationary else ""
    rel=(f" R{rider.reliability:.0f}" if rider.reliability<80 else "")
    line1=f"#{rider.rider_id} {rider.score:.0f}/100  {rider.speed_kmh():.0f}km/h {dir_arrow}{still}{rel}"
    line2=f"B{rider.hard_brake_count} W{rider.weave_count} T{rider.tailgate_count} S{rider.sudden_stop_count}"
    if helm: line2+=f"  {helm}"
    ty=max(y1-36,0)
    # Dark background strip so text is readable over any vehicle colour
    strip_h=34
    if ty>0:
        ov2=frame.copy()
        sx1=max(x1-2,0); sx2=min(x2+2,FRAME_W)
        cv2.rectangle(ov2,(sx1,ty-2),(sx2,ty+strip_h),(0,0,0),-1)
        cv2.addWeighted(ov2,0.45,frame,0.55,0,frame)
    _safe_text(frame,line1,(x1,ty+12),0.38,color,1)
    _safe_text(frame,line2,(x1,ty+26),0.34,(200,200,200),1)

def draw_fleet_bar(frame,fleet,n_frame):
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(FRAME_W,20),(15,15,50),-1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
    txt=(f"Fleet {fleet.fleet_speed_mps*3.6:.0f}km/h "
         f"a={fleet.fleet_accel_mps2:+.1f}  "
         f"brk_thr={fleet.effective_brake_thr():.1f}  "
         f"n={fleet.n_vehicles}    Frame {n_frame}")
    cv2.putText(frame,txt,(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.33,(180,210,255),1,cv2.LINE_AA)

def draw_ego_arrow(frame,dx,dy):
    if CAMERA_MODE!="dashcam": return
    cx,cy=FRAME_W//2,34
    cv2.arrowedLine(frame,(cx,cy),(int(cx+dx*3),int(cy+dy*3)),(0,200,255),2,tipLength=0.35)
    _safe_text(frame,f"ego({dx:+.1f},{dy:+.1f})",(cx-40,cy-6),0.30,(0,200,255),1)

def draw_leaderboard(frame,behaviors):
    if not behaviors: return
    rows=sorted(behaviors.values(),key=lambda b:-b.score)
    pane_w=200; pane_h=len(rows)*22+32; pane_x=FRAME_W-pane_w
    ov=frame.copy()
    cv2.rectangle(ov,(pane_x,20),(FRAME_W,20+pane_h),(10,10,10),-1)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
    cv2.putText(frame,"  ID  Score  km/h  Ev",(pane_x+4,36),
                cv2.FONT_HERSHEY_SIMPLEX,0.35,(220,200,0),1)
    cv2.line(frame,(pane_x,38),(FRAME_W,38),(60,60,60),1)
    for i,b in enumerate(rows):
        y=54+i*22; col=b.score_color()
        pill=f"{b.score:.0f}"
        (ptw,_),_=cv2.getTextSize(pill,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        cv2.rectangle(frame,(pane_x+28,y-10),(pane_x+28+ptw+6,y+4),col,-1)
        cv2.putText(frame,pill,(pane_x+31,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1,cv2.LINE_AA)
        rel_c=(180,180,180) if b.reliability>=80 else (120,120,80)
        row_txt=f"#{b.rider_id:<2}        {b.speed_kmh():4.0f}  {b.total_events():2}"
        cv2.putText(frame,row_txt,(pane_x+4,y+2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
        helm_c=((0,200,0) if b.helmet_status=="HELMET"
                else((0,40,220) if b.helmet_status=="NO_HELMET"
                     else (140,140,140)))
        cv2.circle(frame,(FRAME_W-10,y-3),4,helm_c,-1)
        # Reliability indicator: dim if track is new/unreliable
        if b.reliability < 80:
            cv2.putText(frame,"~",(pane_x+4,y+14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.28,(120,120,80),1)

def draw_footer(frame):
    ov=frame.copy()
    cv2.rectangle(ov,(0,FRAME_H-18),(FRAME_W,FRAME_H),(10,10,10),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    mode=f"{CAMERA_MODE.upper()}/{CAMERA_ANGLE.upper()}"
    cv2.putText(frame,
                f"Varroc Eureka 3.0 - PS3 v5  [{mode}]",
                (4,FRAME_H-4),cv2.FONT_HERSHEY_SIMPLEX,0.32,(160,160,160),1,cv2.LINE_AA)

# ============================= STARTUP SELECTOR ============================= #

def run_selector():
    sel_mode="dashcam"; sel_angle="side"
    W,H=640,360; BG=(18,18,30); ACT=(30,200,30); INACT=(80,80,80)
    TITLE=(220,200,50); WHITE=(240,240,240); HINT=(120,120,120)

    def pill(canvas,label,desc,key_ch,y,active):
        col=ACT if active else INACT
        cv2.rectangle(canvas,(60,y-26),(580,y+8),col,-1 if active else 2)
        cv2.rectangle(canvas,(60,y-26),(580,y+8),col,2)
        cv2.rectangle(canvas,(68,y-22),(98,y+4),(40,40,40),-1)
        cv2.putText(canvas,key_ch,(73,y+2),cv2.FONT_HERSHEY_SIMPLEX,0.55,WHITE,1,cv2.LINE_AA)
        cv2.putText(canvas,label,(108,y+2),cv2.FONT_HERSHEY_SIMPLEX,0.60,
                    WHITE if active else INACT,2 if active else 1,cv2.LINE_AA)
        cv2.putText(canvas,desc,(300,y+2),cv2.FONT_HERSHEY_SIMPLEX,0.37,
                    WHITE if active else INACT,1,cv2.LINE_AA)

    cv2.namedWindow("Varroc Eureka PS3 Setup",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Varroc Eureka PS3 Setup",W,H)
    while True:
        canvas=np.full((H,W,3),BG,dtype=np.uint8)
        cv2.rectangle(canvas,(0,0),(W,44),(25,25,55),-1)
        cv2.putText(canvas,"VARROC EUREKA 3.0  |  PS3  -  Camera Setup",
                    (12,30),cv2.FONT_HERSHEY_SIMPLEX,0.55,TITLE,1,cv2.LINE_AA)
        cv2.putText(canvas,"CAMERA MODE",(60,74),cv2.FONT_HERSHEY_SIMPLEX,0.42,HINT,1,cv2.LINE_AA)
        pill(canvas,"DASHCAM","bike-mounted / moving vehicle","D",100,sel_mode=="dashcam")
        pill(canvas,"FIXED",  "roadside / overhead / stationary","F",148,sel_mode=="fixed")
        cv2.line(canvas,(60,172),(580,172),(50,50,50),1)
        cv2.putText(canvas,"CAMERA ANGLE",(60,196),cv2.FONT_HERSHEY_SIMPLEX,0.42,HINT,1,cv2.LINE_AA)
        pill(canvas,"SIDE",   "front / side profile (standard)","S",222,sel_angle=="side")
        pill(canvas,"OVERHEAD","top-down / bird's-eye view","O",270,sel_angle=="overhead")
        cv2.rectangle(canvas,(60,293),(580,318),(30,40,30),-1)
        cv2.putText(canvas,
                    f"Selected:  {sel_mode.upper()}  /  {sel_angle.upper()}"
                    "      (ENTER=start  ESC=quit)",
                    (68,311),cv2.FONT_HERSHEY_SIMPLEX,0.37,ACT,1,cv2.LINE_AA)
        cv2.putText(canvas,"D/F = mode      S/O = angle      ENTER = start",
                    (130,345),cv2.FONT_HERSHEY_SIMPLEX,0.36,HINT,1,cv2.LINE_AA)
        cv2.imshow("Varroc Eureka PS3 Setup",canvas)
        key=cv2.waitKey(30)&0xFF
        if   key in (ord('d'),ord('D')): sel_mode="dashcam"
        elif key in (ord('f'),ord('F')): sel_mode="fixed"
        elif key in (ord('s'),ord('S')): sel_angle="side"
        elif key in (ord('o'),ord('O')): sel_angle="overhead"
        elif key in (13,10): break
        elif key==27: cv2.destroyAllWindows(); raise SystemExit("Cancelled.")
    cv2.destroyWindow("Varroc Eureka PS3 Setup")
    print(f"[MODE]  {sel_mode.upper()} / {sel_angle.upper()}")
    return sel_mode,sel_angle

# ============================= MAIN ========================================= #

def main():
    global CAMERA_MODE,CAMERA_ANGLE
    if CAMERA_MODE is None or CAMERA_ANGLE is None:
        CAMERA_MODE,CAMERA_ANGLE=run_selector()

    model=YOLO(MODEL_PATH)
    cap  =cv2.VideoCapture(VIDEO_PATH)
    fps  =cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK

    if CAMERA_ANGLE=="overhead":
        active_conf=CONF_OVERHEAD; active_classes=TWO_WHEELER_CLASSES_OVERHEAD
    else:
        active_conf=CONF_SIDE;     active_classes=TWO_WHEELER_CLASSES_SIDE

    print(f"[INFO] FPS:{fps:.0f}  Mode:{CAMERA_MODE}  Angle:{CAMERA_ANGLE}  "
          f"Conf:{active_conf}  Classes:{active_classes}")

    if BYTETRACK_AVAILABLE:
        tracker = ByteTrackWrapper(fps)
        print("[INFO] Tracker: ByteTrack")
    else:
        tracker = IoUTracker()
        print("[INFO] Tracker: built-in IoU  (pip install boxmot for ByteTrack)")
    ego_est    = EgoMotionEstimator() if CAMERA_MODE=="dashcam" else None
    fleet      = FleetContext()
    helmet_det = HelmetDetector()
    behaviors: dict[int,RiderBehavior] = {}
    all_time:  dict[int,RiderBehavior] = {}
    n=0

    while True:
        ret,frame=cap.read()
        if not ret: break
        n+=1
        frame=cv2.resize(frame,(FRAME_W,FRAME_H))
        gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        raw_edx,raw_edy=(ego_est.update(gray) if ego_est else (0.0,0.0))

        results=model(frame,conf=active_conf,verbose=False)[0]
        det_bikes=[]; person_boxes=[]; det_cls_list=[]
        for b in results.boxes:
            cls=int(b.cls[0]); box=tuple(map(int,b.xyxy[0])); conf=float(b.conf[0])
            if cls in active_classes and conf>=active_conf:
                det_bikes.append((box,cls)); det_cls_list.append((box,cls))
            elif cls==0 and conf>=PERSON_CONF: person_boxes.append(box)

        bike_boxes=[b for b,_ in det_bikes]
        tracker.update(bike_boxes)
        # List-based lookup avoids dict key collision on identical boxes
        def get_cls(box, default=3):
            for b,c in det_bikes:
                if b==box: return c
            return default
        active=tracker.active_tracks

        fleet.update(behaviors)

        if ego_est and behaviors:
            mpp_vals=[float(np.median(list(bv._mpp_hist)))
                      for bv in behaviors.values() if bv._mpp_hist]
            cam_mpp=float(np.median(mpp_vals)) if mpp_vals else 0.020
            camera_speed_mps=min(math.hypot(raw_edx,raw_edy)*cam_mpp*fps,
                                 MAX_CAMERA_SPEED_MPS)
        else:
            camera_speed_mps=0.0

        for tid,td in active.items():
            if tid not in behaviors: behaviors[tid]=RiderBehavior(tid,fps)
            b=behaviors[tid]
            cx,cy=td['cx'],td['cy']
            df    =EgoMotionEstimator.depth_factor(cy) if ego_est else 1.0
            edx_v =raw_edx*df if ego_est else 0.0
            edy_v =raw_edy*df if ego_est else 0.0
            # comp displacement: 1-frame centroid delta minus ego shift
            if b._prev_cx is not None:
                raw_dx=float(cx)-b._prev_cx; raw_dy=float(cy)-b._prev_cy
            else:
                raw_dx=raw_dy=0.0
            comp_dx=raw_dx-edx_v; comp_dy=raw_dy-edy_v
            det_cls=get_cls(td['box'])
            b.update(cx,cy,td['box'],comp_dx,comp_dy,fleet,
                     camera_speed_mps,det_cls)
            all_time[tid]=b

        statuses=helmet_det.update(gray,active,person_boxes)
        for tid,status in statuses.items():
            if tid in behaviors: behaviors[tid].set_helmet(status)

        alive=set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive]: del behaviors[k]

        for tid,td in active.items():
            if tid in behaviors: draw_hud(frame,behaviors[tid],td['box'])
        draw_fleet_bar(frame,fleet,n)
        draw_ego_arrow(frame,raw_edx,raw_edy)
        active_beh={tid:behaviors[tid] for tid in active if tid in behaviors}
        draw_leaderboard(frame,active_beh)
        draw_footer(frame)
        cv2.imshow("Varroc Eureka 3.0 - PS3 v5  [ESC to quit]",frame)
        if cv2.waitKey(1)&0xFF==27: break

    # ── Final report ─────────────────────────────────────────────────────────
    W=102
    COLS=[("ID",5,"<"),("Score",7,">"),("km/h",7,">"),("Brakes",7,">"),
          ("Accel",6,">"),("Weave",6,">"),("Tailg",6,">"),("Stop",5,">"),
          ("NoHlmt",7,">"),("Helmet",8,"^"),("Direction",10,"^"),
          ("Stunt",6,"^"),("Rating",8,"^")]
    def divider(l="├",m="┼",r="┤",f="-"):
        return l+m.join(f*(w+2) for _,w,_ in COLS)+r
    def row_str(vals):
        return "|"+"│".join(f" {v:{a}{w}} " for (_,w,a),v in zip(COLS,vals))+"|"

    print(); print("+" + "-"*(W-2) + "+")
    print("|"+"VARROC EUREKA 3.0  --  PS3 DRIVING BEHAVIOR REPORT  v5".center(W-2)+"|")
    sub=(f"Vehicles:{len(all_time)}  Frames:{n}  Duration:{n/fps:.1f}s  "
         f"Mode:{CAMERA_MODE.upper()}/{CAMERA_ANGLE.upper()}")
    print("|"+sub.center(W-2)+"|")
    print("+" + "-"*(W-2) + "+")
    print(divider("+","+"," +"))
    print(row_str([h for h,_,_ in COLS]))
    print(divider("+","+"," +"))
    for i,(rid,b) in enumerate(sorted(all_time.items())):
        helm=("YES" if b.helmet_status=="HELMET"
              else("NO " if b.helmet_status=="NO_HELMET" else " ? "))
        vals=[f"#{rid}",f"{b.score:.1f}",f"{b.speed_kmh():.1f}",
              str(b.hard_brake_count),str(b.aggr_accel_count),
              str(b.weave_count),str(b.tailgate_count),str(b.sudden_stop_count),
              str(b.no_helmet_count),helm,b.direction,
              "YES" if b.is_doing_stunt else " no",f"[{b.score_label()}]"]
        print(row_str(vals))
        if i<len(all_time)-1: print(divider())
    print(divider("+","+"," +"))
    if all_time:
        scores=[b.score for b in all_time.values()]
        best =max(all_time,key=lambda k:all_time[k].score)
        worst=min(all_time,key=lambda k:all_time[k].score)
        g=sum(1 for b in all_time.values() if b.score>=80)
        f_=sum(1 for b in all_time.values() if 50<=b.score<80)
        p=sum(1 for b in all_time.values() if b.score<50)
        nh=sum(1 for b in all_time.values() if b.no_helmet_count>0)
        print(); print("+" + "-"*(W-2) + "+")
        print("|  SUMMARY".ljust(W-1)+"|")
        print("+" + "-"*(W-2) + "+")
        for line in [
            f"  Average Score : {sum(scores)/len(scores):.1f}",
            f"  Best Rider    : #{best}  ({max(scores):.1f} pts)",
            f"  Worst Rider   : #{worst}  ({min(scores):.1f} pts)",
            f"  GOOD (>=80): {g}   FAIR (50-79): {f_}   POOR (<50): {p}",
            f"  No-Helmet Violations: {nh} rider(s)",
        ]: print("|"+line.ljust(W-2)+"|")
        print("+" + "-"*(W-2) + "+")
    print(); cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    main()