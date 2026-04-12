"""
============================================================
Varroc Eureka 3.0 — PS3  |  FIXED CAMERA MODE
Accurate Driving Behavior Score on a Two-Wheeler
============================================================
Camera    : Stationary roadside, overhead, or CCTV camera
Speed     : CA Kalman on raw centroid (no ego compensation)
            Fleet-shared MPP (mid-range vehicles only)
            Higher confidence threshold — stable background
Events    : Hard brake, aggressive accel, weave, tailgate,
            sudden stop, no-helmet, over-speed
            All directions scored (no receding exemption —
            fixed cam sees absolute motion on the road)
Context   : FleetContext — dynamic thresholds in traffic
Score     : 100 - penalties, EMA-smoothed, reliability indicator
"""

import cv2, math
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO

try:
    from boxmot import ByteTrack
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False

# ========================== CONFIGURATION =================================== #

VIDEO_PATH = r"./public/video1.mp4"
MODEL_PATH = "yolov8m.pt"

FRAME_W, FRAME_H = 640, 360
FPS_FALLBACK     = 30

# Higher conf — stable background means cleaner detections
CONF_THRESH  = 0.35
PERSON_CONF  = 0.35
ACTIVE_CLASSES = {1, 3}

# Speed
REAL_WORLD_WIDTHS        = {0: 0.50, 1: 0.60, 3: 0.80}
REAL_WORLD_WIDTH_DEFAULT = 0.75
MPP_HISTORY_LEN          = 25
MAX_VEHICLE_SPEED_MPS    = 35.0
MAX_PLAUSIBLE_CLOSE_MPS  = 15.0
CLOSE_BBOX_FRACTION      = 0.25

# Stationarity — tighter (no ego noise on fixed cam)
STATIONARY_DISP_PX  = 2.0
STATIONARY_AREA_VAR = 0.0006
STATIONARY_WINDOW   = 10

# Direction
DIR_WINDOW        = 10
DIR_ONCOMING_RATE = 0.004
DIR_APPROACH_RATE = 0.001
DIR_RECEDING_RATE = -0.001

# Helmet
HELMET_OVERLAP_IOU    = 0.20
HELMET_HEAD_FRACTION  = 0.32
HELMET_MIN_AREA_FRAC  = 0.08
HELMET_CONFIRM_FRAMES = 15
W_NO_HELMET           = 10

# Events — tighter thresholds (no ego noise contaminating signal)
BASE_HARD_BRAKE_MPS2  = 2.5
BASE_AGGR_ACCEL_MPS2  = 2.0
LANE_WEAVE_PX         = 14
TAILGATE_AREA_THRESH  = 0.09
TAILGATE_GROWTH_RATE  = 0.002
SUDDEN_STOP_SPEED_MPS = 1.2
SUDDEN_STOP_FRAMES    = 6
SPEED_LIMIT_MPS       = 13.9
FLEET_K               = 0.6

# Score weights
W_HARD_BRAKE  = 5;  W_AGGR_ACCEL = 3;  W_WEAVE       = 2
W_TAILGATE    = 4;  W_SUDDEN_STOP= 5;  W_OVER_SPEED  = 0.15
SCORE_EMA_ALPHA = 0.04;  COLOR_EMA_ALPHA = 0.02

# Tracker — longer disappear window (occlusion more common on fixed cam)
IOU_MATCH_THRESH = 0.20
MAX_DISAPPEARED  = 45
EVENT_COOLDOWN   = 18
WARMUP_FRAMES    = 8

# ========================== UTILITIES ======================================= #

def compute_iou(a,b):
    xA=max(a[0],b[0]);yA=max(a[1],b[1]);xB=min(a[2],b[2]);yB=min(a[3],b[3])
    inter=max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    return inter/float((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def bbox_mpp(bw,cls):
    return REAL_WORLD_WIDTHS.get(cls,REAL_WORLD_WIDTH_DEFAULT)/max(bw,5.0)

def max_plausible_speed(bw):
    frac=bw/FRAME_W
    if frac>=CLOSE_BBOX_FRACTION:
        t=min((frac-CLOSE_BBOX_FRACTION)/(1.0-CLOSE_BBOX_FRACTION),1.0)
        return MAX_PLAUSIBLE_CLOSE_MPS*(1.0-0.85*t)
    return MAX_VEHICLE_SPEED_MPS

# ========================== HELMET DETECTOR ================================= #

class HelmetDetector:
    def __init__(self):
        self._history=defaultdict(lambda:deque(maxlen=HELMET_CONFIRM_FRAMES))
        self.status={}

    def update(self,gray,tracks,person_boxes):
        for tid,td in tracks.items():
            best_iou,best_pb=0.0,None
            for pb in person_boxes:
                iou=compute_iou(td['box'],pb)
                if iou>best_iou: best_iou,best_pb=iou,pb
            if best_pb is None or best_iou<HELMET_OVERLAP_IOU:
                self.status[tid]=self.status.get(tid,"?"); continue
            px1,py1,px2,py2=best_pb;p_h=max(py2-py1,1)
            h_y1=max(py1,0);h_y2=int(py1+p_h*HELMET_HEAD_FRACTION)
            h_x1=max(px1,0);h_x2=min(px2,gray.shape[1])
            if h_y2<=h_y1 or h_x2<=h_x1: continue
            crop=gray[h_y1:h_y2,h_x1:h_x2]
            if crop.size<50: continue
            _,thresh=cv2.threshold(crop,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cnts,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            blob=max((cv2.contourArea(c) for c in cnts),default=0)
            self._history[tid].append(blob/max(crop.shape[0]*crop.shape[1],1)>=HELMET_MIN_AREA_FRAC)
            h=self._history[tid]
            if len(h)>=HELMET_CONFIRM_FRAMES//2:
                self.status[tid]="HELMET" if sum(h)/len(h)>=0.55 else "NO_HELMET"
            else: self.status[tid]="?"
        return self.status

# ========================== KALMAN (CA) ===================================== #

class KalmanVelocityEstimator:
    PROCESS_NOISE=0.6; MEASURE_NOISE=22.0  # tighter for stable fixed cam
    def __init__(self):
        self.x=None;self.P=None;self.initialized=False
        dt=1.0
        self.F=np.array([[1,0,dt,0,0.5*dt**2,0],[0,1,0,dt,0,0.5*dt**2],
                          [0,0,1,0,dt,0],[0,0,0,1,0,dt],[0,0,0,0,1,0],[0,0,0,0,0,1]],dtype=float)
        self.H=np.zeros((2,6));self.H[0,0]=1;self.H[1,1]=1
        q=self.PROCESS_NOISE;r=self.MEASURE_NOISE
        self.Q=np.diag([q*.25,q*.25,q*.5,q*.5,q,q])
        self.R=np.diag([r,r])

    def update(self,cx,cy):
        z=np.array([[cx],[cy]],dtype=float)
        if not self.initialized:
            self.x=np.array([[cx],[cy],[0],[0],[0],[0]],dtype=float)
            self.P=np.eye(6)*200.0;self.initialized=True;return cx,cy,0.0,0.0
        x_p=self.F@self.x;P_p=self.F@self.P@self.F.T+self.Q
        S=self.H@P_p@self.H.T+self.R;K=P_p@self.H.T@np.linalg.inv(S)
        self.x=x_p+K@(z-self.H@x_p);self.P=(np.eye(6)-K@self.H)@P_p
        return float(self.x[0]),float(self.x[1]),float(self.x[2]),float(self.x[3])

# ========================== FLEET CONTEXT =================================== #

class FleetContext:
    def __init__(self):
        self.fleet_speed_mps=0.0;self.fleet_accel_mps2=0.0;self.n_vehicles=0

    def update(self,behaviors):
        speeds=[b.speed_mps for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        accels=[b.fast_accel for b in behaviors.values() if b._frame_n>=WARMUP_FRAMES]
        self.n_vehicles=len(speeds)
        self.fleet_speed_mps=float(np.median(speeds)) if speeds else 0.0
        self.fleet_accel_mps2=float(np.clip(float(np.median(accels)) if accels else 0.0,-4.0,4.0))
        all_mpp=[float(np.median(list(bv._mpp_hist))) for bv in behaviors.values()
                 if bv._mpp_hist and 0.006<=float(np.median(list(bv._mpp_hist)))<=0.030]
        self._fleet_mpp=float(np.median(all_mpp)) if all_mpp else 0.012

    def effective_brake_thr(self):
        return BASE_HARD_BRAKE_MPS2+FLEET_K*max(0.0,-self.fleet_accel_mps2)
    def effective_accel_thr(self):
        return BASE_AGGR_ACCEL_MPS2+FLEET_K*max(0.0,self.fleet_accel_mps2)
    def fleet_mpp(self): return getattr(self,'_fleet_mpp',0.012)

# ========================== TRACKER ========================================= #

class IoUTracker:
    def __init__(self):
        self.next_id=0;self.tracks={};self.disappeared=defaultdict(int)

    @staticmethod
    def _cen(box): return (box[0]+box[2])//2,(box[1]+box[3])//2

    def _register(self,box):
        cx,cy=self._cen(box)
        self.tracks[self.next_id]={'box':box,'cx':cx,'cy':cy}
        self.disappeared[self.next_id]=0;self.next_id+=1

    def _dereg(self,tid): self.tracks.pop(tid,None);self.disappeared.pop(tid,None)

    @property
    def active_tracks(self):
        return {t:d for t,d in self.tracks.items() if self.disappeared[t]==0}

    def update(self,det_boxes):
        if not det_boxes:
            for tid in list(self.disappeared):
                self.disappeared[tid]+=1
                if self.disappeared[tid]>MAX_DISAPPEARED: self._dereg(tid)
            return self.tracks
        if not self.tracks:
            for b in det_boxes: self._register(b); return self.tracks
        tids=list(self.tracks.keys())
        tboxes=[self.tracks[t]['box'] for t in tids]
        iou_mat=np.array([[compute_iou(db,tb) for tb in tboxes] for db in det_boxes])
        md=set();mt=set()
        for idx in np.argsort(-iou_mat,axis=None):
            di,ti=divmod(int(idx),len(tboxes))
            if di in md or ti in mt: continue
            if iou_mat[di,ti]<IOU_MATCH_THRESH: break
            tid=tids[ti];box=det_boxes[di];cx,cy=self._cen(box)
            self.tracks[tid]={'box':box,'cx':cx,'cy':cy}
            self.disappeared[tid]=0;md.add(di);mt.add(ti)
        unm_d=[i for i in range(len(det_boxes)) if i not in md]
        unm_t=[i for i in range(len(tboxes)) if i not in mt]
        mxd=math.hypot(FRAME_W,FRAME_H)/4
        for di in unm_d:
            dcx,dcy=self._cen(det_boxes[di]);bd,bti=float('inf'),None
            for ti in unm_t:
                d=math.hypot(dcx-self._cen(tboxes[ti])[0],dcy-self._cen(tboxes[ti])[1])
                if d<bd: bd,bti=d,ti
            if bti is not None and bd<mxd:
                tid=tids[bti];box=det_boxes[di];cx,cy=self._cen(box)
                self.tracks[tid]={'box':box,'cx':cx,'cy':cy}
                self.disappeared[tid]=0;unm_t.remove(bti)
            else: self._register(det_boxes[di])
        for ti in unm_t:
            tid=tids[ti];self.disappeared[tid]+=1
            if self.disappeared[tid]>MAX_DISAPPEARED: self._dereg(tid)
        return self.tracks

class ByteTrackWrapper:
    def __init__(self,fps):
        self._bt=ByteTrack(track_thresh=0.30,track_buffer=45,match_thresh=0.8,frame_rate=int(fps))
        self.tracks={};self.disappeared=defaultdict(int)

    @property
    def active_tracks(self):
        return {t:d for t,d in self.tracks.items() if self.disappeared[t]==0}

    def update(self,det_boxes):
        self.tracks={}
        if not det_boxes: return self.tracks
        dets=np.array([[*b,0.9,0] for b in det_boxes],dtype=np.float32)
        try: tracks=self._bt.update(dets,None)
        except Exception: return self.tracks
        for t in tracks:
            x1,y1,x2,y2=int(t[0]),int(t[1]),int(t[2]),int(t[3]);tid=int(t[4])
            self.tracks[tid]={'box':(x1,y1,x2,y2),'cx':(x1+x2)//2,'cy':(y1+y2)//2}
            self.disappeared[tid]=0
        return self.tracks

# ========================== DIRECTION ======================================= #

DIR_ONCOMING="ONCOMING";DIR_APPROACH="APPROACH"
DIR_RECEDING="RECEDING";DIR_ALONGSIDE="ALONGSIDE";DIR_UNKNOWN="?"
DIR_ARROWS={DIR_ONCOMING:"<<",DIR_APPROACH:"^^",
            DIR_RECEDING:"vv",DIR_ALONGSIDE:"->",DIR_UNKNOWN:"  "}

# ========================== RIDER BEHAVIOR ================================== #

class RiderBehavior:
    def __init__(self,rider_id,fps):
        self.rider_id=rider_id;self.fps=fps;self.dt=1.0/fps;self._frame_n=0
        self._prev_cx=None;self._prev_cy=None
        self._kalman=KalmanVelocityEstimator()
        self._mpp_hist=deque(maxlen=MPP_HISTORY_LEN)
        self._spd_hist=deque(maxlen=max(SUDDEN_STOP_FRAMES+2,12))
        self._area_hist=deque(maxlen=max(STATIONARY_WINDOW,DIR_WINDOW)+2)
        self._lat_hist=deque(maxlen=8);self._ar_hist=deque(maxlen=10)
        self.cls=3;self.speed_mps=0.0;self.fast_accel=0.0
        self.accel_mps2=0.0;self._prev_speed=0.0
        self.direction=DIR_UNKNOWN
        self.helmet_status="?";self.no_helmet_count=0;self._no_helmet_cd=0
        self.is_stationary=False;self.is_doing_stunt=False
        self.reliability=0.0
        self.hard_brake_count=0;self.aggr_accel_count=0
        self.weave_count=0;self.tailgate_count=0
        self.sudden_stop_count=0;self.over_speed_sec=0.0
        self._cd=defaultdict(int)
        self.score=100.0;self._raw=100.0;self._committed=100.0

    def _tick_cd(self):
        for k in list(self._cd): self._cd[k]=max(0,self._cd[k]-1)
        self._no_helmet_cd=max(0,self._no_helmet_cd-1)

    def _fire(self,key,attr):
        if self._cd[key]==0:
            setattr(self,attr,getattr(self,attr)+1);self._cd[key]=EVENT_COOLDOWN

    def _check_stationary(self,dx,dy):
        area_stable=(len(self._area_hist)>=STATIONARY_WINDOW and
                     float(np.var(list(self._area_hist)[-STATIONARY_WINDOW:]))<STATIONARY_AREA_VAR)
        return math.hypot(dx,dy)<STATIONARY_DISP_PX and area_stable

    def _classify_direction(self):
        if len(self._area_hist)<DIR_WINDOW: return DIR_UNKNOWN
        r=list(self._area_hist)[-DIR_WINDOW:]
        s=float(np.polyfit(np.arange(len(r)),r,1)[0])
        if s>DIR_ONCOMING_RATE: return DIR_ONCOMING
        if s>DIR_APPROACH_RATE: return DIR_APPROACH
        if s<DIR_RECEDING_RATE: return DIR_RECEDING
        return DIR_ALONGSIDE

    def set_helmet(self,status):
        self.helmet_status=status
        if status=="NO_HELMET" and self._no_helmet_cd==0:
            self.no_helmet_count+=1;self._no_helmet_cd=EVENT_COOLDOWN*2

    def update(self,cx,cy,box,fleet,cls=3):
        self._frame_n+=1;self._tick_cd();self.cls=cls
        prev_cx=self._prev_cx if self._prev_cx is not None else float(cx)
        prev_cy=self._prev_cy if self._prev_cy is not None else float(cy)
        self._prev_cx=float(cx);self._prev_cy=float(cy)

        bw=max(float(box[2]-box[0]),5.0);bh=max(float(box[3]-box[1]),5.0)
        af=(bw*bh)/(FRAME_W*FRAME_H);ar=bh/bw
        self._area_hist.append(af);self._ar_hist.append(ar)

        # Raw 1-frame displacement (no ego compensation needed)
        raw_dx=float(cx)-prev_cx;raw_dy=float(cy)-prev_cy
        self.direction=self._classify_direction()
        self.is_stationary=self._check_stationary(raw_dx,raw_dy)

        self._mpp_hist.append(bbox_mpp(bw,cls))
        fm=fleet.fleet_mpp()
        mpp=fm if fm>0.005 else float(np.median(list(self._mpp_hist)))

        # Kalman on raw centroid (no ego compensation on fixed cam)
        _,_,vx_k,vy_k=self._kalman.update(float(cx),float(cy))
        raw_spd=math.hypot(vx_k,vy_k)*mpp*self.fps
        self._spd_hist.append(raw_spd)

        raw_spd_abs=0.0 if self.is_stationary else min(raw_spd,max_plausible_speed(bw))

        raw_fa=(raw_spd-self._prev_speed)/self.dt
        self.fast_accel=float(np.clip(raw_fa,-8.0,8.0))
        self.speed_mps=0.30*raw_spd_abs+0.70*self._prev_speed
        self.accel_mps2=(self.speed_mps-self._prev_speed)/self.dt
        self._prev_speed=self.speed_mps

        self._lat_hist.append(abs(raw_dx))
        lat=float(np.median(list(self._lat_hist)))

        if len(self._ar_hist)>=5:
            am=float(np.mean(list(self._ar_hist)));av=float(np.var(list(self._ar_hist)))
            self.is_doing_stunt=am>2.8 and av>0.15
            if self.is_doing_stunt: self.score=0.0; return

        if self._frame_n<WARMUP_FRAMES: self._recompute(); return

        bt=fleet.effective_brake_thr();at=fleet.effective_accel_thr()
        if self.fast_accel<-bt: self._fire('brake','hard_brake_count')
        if self.fast_accel>at:  self._fire('accel','aggr_accel_count')
        if lat>LANE_WEAVE_PX:   self._fire('weave','weave_count')
        if (self.direction in (DIR_APPROACH,DIR_ALONGSIDE) and
                len(self._area_hist)>=5 and af>TAILGATE_AREA_THRESH and
                self._area_hist[-1]-self._area_hist[-5]>TAILGATE_GROWTH_RATE):
            self._fire('tail','tailgate_count')
        if (len(self._spd_hist)>=SUDDEN_STOP_FRAMES and not self.is_stationary and
                self._spd_hist[-SUDDEN_STOP_FRAMES]>SUDDEN_STOP_SPEED_MPS and
                self.speed_mps<SUDDEN_STOP_SPEED_MPS):
            self._fire('stop','sudden_stop_count')
        # Fixed cam sees absolute speed — all directions scored
        if self.speed_mps>SPEED_LIMIT_MPS: self.over_speed_sec+=self.dt
        self._recompute()

    def _recompute(self):
        p=(self.hard_brake_count*W_HARD_BRAKE+self.aggr_accel_count*W_AGGR_ACCEL+
           self.weave_count*W_WEAVE+self.tailgate_count*W_TAILGATE+
           self.sudden_stop_count*W_SUDDEN_STOP+self.no_helmet_count*W_NO_HELMET+
           self.over_speed_sec*W_OVER_SPEED)
        self._raw=max(0.0,min(100.0,100.0-p))
        self.score=SCORE_EMA_ALPHA*self._raw+(1-SCORE_EMA_ALPHA)*self.score
        self._committed=COLOR_EMA_ALPHA*self.score+(1-COLOR_EMA_ALPHA)*self._committed
        self.reliability=min(100.0,self._frame_n/max(WARMUP_FRAMES,1)*30.0)

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
        return (30,200,30) if s>=80 else ((0,165,255) if s>=50 else (40,40,220))

# ========================== HUD ============================================= #

def _safe_text(frame,text,org,scale,color,thick=1):
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,scale,thick)
    x=max(2,min(org[0],FRAME_W-tw-2));y=max(th+2,min(org[1],FRAME_H-4))
    cv2.putText(frame,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)

def draw_hud(frame,rider,box):
    x1,y1,x2,y2=box;color=rider.score_color();bw=max(x2-x1,1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
    cv2.rectangle(frame,(x1,y1),(x2,y1+5),(30,30,30),-1)
    cv2.rectangle(frame,(x1,y1),(x1+int(bw*rider.score/100),y1+5),color,-1)
    dir_a=DIR_ARROWS.get(rider.direction,"")
    helm="H+" if rider.helmet_status=="HELMET" else("X!" if rider.helmet_status=="NO_HELMET" else "")
    still=" ST" if rider.is_stationary else ""
    rel=f" R{rider.reliability:.0f}" if rider.reliability<80 else ""
    l1=f"#{rider.rider_id} {rider.score:.0f}/100  {rider.speed_kmh():.0f}km/h {dir_a}{still}{rel}"
    l2=f"B{rider.hard_brake_count} W{rider.weave_count} T{rider.tailgate_count} S{rider.sudden_stop_count}"
    if helm: l2+=f"  {helm}"
    ty=max(y1-36,0)
    if ty>0:
        ov=frame.copy()
        cv2.rectangle(ov,(max(x1-2,0),ty-2),(min(x2+2,FRAME_W),ty+34),(0,0,0),-1)
        cv2.addWeighted(ov,0.45,frame,0.55,0,frame)
    _safe_text(frame,l1,(x1,ty+12),0.38,color,1)
    _safe_text(frame,l2,(x1,ty+26),0.34,(200,200,200),1)

def draw_fleet_bar(frame,fleet,n):
    ov=frame.copy();cv2.rectangle(ov,(0,0),(FRAME_W,20),(15,50,15),-1)
    cv2.addWeighted(ov,0.7,frame,0.3,0,frame)
    txt=(f"Fleet {fleet.fleet_speed_mps*3.6:.0f}km/h  a={fleet.fleet_accel_mps2:+.1f}"
         f"  brk_thr={fleet.effective_brake_thr():.1f}  n={fleet.n_vehicles}  [FIXED CAM]  Frame {n}")
    cv2.putText(frame,txt,(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.33,(180,255,180),1,cv2.LINE_AA)

def draw_leaderboard(frame,behaviors):
    if not behaviors: return
    rows=sorted(behaviors.values(),key=lambda b:-b.score)
    pw=200;ph=len(rows)*22+32;px=FRAME_W-pw
    ov=frame.copy();cv2.rectangle(ov,(px,20),(FRAME_W,20+ph),(10,10,10),-1)
    cv2.addWeighted(ov,0.70,frame,0.30,0,frame)
    cv2.putText(frame,"  ID  Score  km/h  Ev",(px+4,36),cv2.FONT_HERSHEY_SIMPLEX,0.35,(220,200,0),1)
    cv2.line(frame,(px,38),(FRAME_W,38),(60,60,60),1)
    for i,b in enumerate(rows):
        y=54+i*22;col=b.score_color()
        pill=f"{b.score:.0f}"
        (ptw,_),_=cv2.getTextSize(pill,cv2.FONT_HERSHEY_SIMPLEX,0.40,1)
        cv2.rectangle(frame,(px+28,y-10),(px+28+ptw+6,y+4),col,-1)
        cv2.putText(frame,pill,(px+31,y+2),cv2.FONT_HERSHEY_SIMPLEX,0.40,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,f"#{b.rider_id:<2}        {b.speed_kmh():4.0f}  {b.total_events():2}",
                    (px+4,y+2),cv2.FONT_HERSHEY_SIMPLEX,0.35,col,1,cv2.LINE_AA)
        hc=(0,200,0) if b.helmet_status=="HELMET" else((0,40,220) if b.helmet_status=="NO_HELMET" else(140,140,140))
        cv2.circle(frame,(FRAME_W-10,y-3),4,hc,-1)
        if b.reliability<80:
            cv2.putText(frame,"~",(px+4,y+14),cv2.FONT_HERSHEY_SIMPLEX,0.28,(120,120,80),1)

def draw_footer(frame):
    ov=frame.copy();cv2.rectangle(ov,(0,FRAME_H-18),(FRAME_W,FRAME_H),(10,10,10),-1)
    cv2.addWeighted(ov,0.6,frame,0.4,0,frame)
    cv2.putText(frame,"Varroc Eureka 3.0 - PS3  |  FIXED CAMERA  |  v5",
                (4,FRAME_H-4),cv2.FONT_HERSHEY_SIMPLEX,0.32,(160,160,160),1,cv2.LINE_AA)

# ========================== REPORT ========================================== #

def print_report(all_time,n,fps):
    W=102
    COLS=[("ID",5,"<"),("Score",7,">"),("km/h",7,">"),("Brakes",7,">"),
          ("Accel",6,">"),("Weave",6,">"),("Tailg",6,">"),("Stop",5,">"),
          ("NoHlmt",7,">"),("Helmet",8,"^"),("Direction",10,"^"),
          ("Stunt",6,"^"),("Rating",8,"^")]
    def div(): return "+"+"┼".join("-"*(w+2) for _,w,_ in COLS)+"+"
    def row(vals):
        return "|"+"│".join(f" {v:{a}{w}} " for (_,w,a),v in zip(COLS,vals))+"|"
    print();print("+"+"─"*(W-2)+"+")
    print("|"+"VARROC EUREKA 3.0  --  PS3  |  FIXED CAMERA REPORT  v5".center(W-2)+"|")
    print("|"+f"Vehicles:{len(all_time)}  Frames:{n}  Duration:{n/fps:.1f}s".center(W-2)+"|")
    print("+"+"─"*(W-2)+"+")
    print(div());print(row([h for h,_,_ in COLS]));print(div())
    for i,(rid,b) in enumerate(sorted(all_time.items())):
        helm="YES" if b.helmet_status=="HELMET" else("NO " if b.helmet_status=="NO_HELMET" else " ? ")
        vals=[f"#{rid}",f"{b.score:.1f}",f"{b.speed_kmh():.1f}",str(b.hard_brake_count),
              str(b.aggr_accel_count),str(b.weave_count),str(b.tailgate_count),
              str(b.sudden_stop_count),str(b.no_helmet_count),helm,b.direction,
              "YES" if b.is_doing_stunt else " no",f"[{b.score_label()}]"]
        print(row(vals))
        if i<len(all_time)-1: print(div())
    print(div())
    if all_time:
        scores=[b.score for b in all_time.values()]
        g=sum(1 for b in all_time.values() if b.score>=80)
        f_=sum(1 for b in all_time.values() if 50<=b.score<80)
        p=sum(1 for b in all_time.values() if b.score<50)
        nh=sum(1 for b in all_time.values() if b.no_helmet_count>0)
        print();print("+"+"─"*(W-2)+"+")
        for line in [
            "  SUMMARY",
            f"  Avg:{sum(scores)/len(scores):.1f}  Best:#{max(all_time,key=lambda k:all_time[k].score)} ({max(scores):.1f})  Worst:#{min(all_time,key=lambda k:all_time[k].score)} ({min(scores):.1f})",
            f"  GOOD:{g}  FAIR:{f_}  POOR:{p}  No-Helmet:{nh}",
        ]: print("|"+line.ljust(W-2)+"|")
        print("+"+"─"*(W-2)+"+")
    print()

# ========================== MAIN ============================================ #

def main():
    print("[FIXED CAMERA MODE]")
    model=YOLO(MODEL_PATH)
    cap=cv2.VideoCapture(VIDEO_PATH)
    fps=cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    print(f"[INFO] FPS:{fps:.0f}  Conf:{CONF_THRESH}  Classes:{ACTIVE_CLASSES}")

    if BYTETRACK_AVAILABLE:
        tracker=ByteTrackWrapper(fps);print("[INFO] Tracker: ByteTrack")
    else:
        tracker=IoUTracker();print("[INFO] Tracker: IoU  (pip install boxmot for ByteTrack)")

    fleet=FleetContext();helmet_det=HelmetDetector()
    behaviors={};all_time={};n=0

    while True:
        ret,frame=cap.read()
        if not ret: break
        n+=1
        frame=cv2.resize(frame,(FRAME_W,FRAME_H))
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        results=model(frame,conf=CONF_THRESH,verbose=False)[0]
        det_bikes=[];person_boxes=[]
        for b in results.boxes:
            cls=int(b.cls[0]);box=tuple(map(int,b.xyxy[0]));conf=float(b.conf[0])
            if cls in ACTIVE_CLASSES and conf>=CONF_THRESH: det_bikes.append((box,cls))
            elif cls==0 and conf>=PERSON_CONF: person_boxes.append(box)

        tracker.update([b for b,_ in det_bikes])
        def get_cls(box):
            for b,c in det_bikes:
                if b==box: return c
            return 3
        active=tracker.active_tracks
        fleet.update(behaviors)

        for tid,td in active.items():
            if tid not in behaviors: behaviors[tid]=RiderBehavior(tid,fps)
            b=behaviors[tid]
            # Fixed cam: no ego compensation — pass raw centroid directly
            b.update(td['cx'],td['cy'],td['box'],fleet,get_cls(td['box']))
            all_time[tid]=b

        statuses=helmet_det.update(gray,active,person_boxes)
        for tid,s in statuses.items():
            if tid in behaviors: behaviors[tid].set_helmet(s)

        alive=set(tracker.tracks.keys())
        for k in [k for k in behaviors if k not in alive]: del behaviors[k]

        for tid,td in active.items():
            if tid in behaviors: draw_hud(frame,behaviors[tid],td['box'])
        draw_fleet_bar(frame,fleet,n)
        draw_leaderboard(frame,{tid:behaviors[tid] for tid in active if tid in behaviors})
        draw_footer(frame)
        cv2.imshow("Varroc Eureka 3.0 - PS3  |  FIXED CAM  [ESC to quit]",frame)
        if cv2.waitKey(1)&0xFF==27: break

    print_report(all_time,n,fps)
    cap.release();cv2.destroyAllWindows()

if __name__=="__main__":
    main()