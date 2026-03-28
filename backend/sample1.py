"""
============================================================
Varroc Eureka 3.0 — Problem Statement 3
Accurate Driving Behavior Score on a Two-Wheeler
============================================================

Approach:
  - YOLOv8 detects two-wheelers (class 3 = motorcycle) per frame
  - Simple centroid tracker assigns persistent IDs
  - Per-rider metrics computed every frame:
      • Estimated pixel-speed  → real speed via calibration factor
      • Longitudinal acceleration / deceleration
      • Lateral displacement   → lane-change / weaving count
      • Hard-brake events      → sudden deceleration spikes
  - Weighted composite score (0–100) displayed live on frame

Score Formula (starts at 100, deductions applied):
    Score = 100
            − w1 * hard_brake_penalty
            − w2 * aggressive_accel_penalty
            − w3 * weave_penalty
            − w4 * over_speed_penalty
    Clamped to [0, 100].

Judging criteria addressed:
    ✓ Novelty       – vision-only, no extra hardware
    ✓ Simplicity    – single file, standard libs
    ✓ Form-factor   – runs on dashcam / roadside camera
    ✓ Cost-effective – uses free YOLOv8n model
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import math

# ─────────────────────────── CONFIGURATION ──────────────────────────────── #

VIDEO_PATH = r"D:\hackathon\public\vlog1.mp4"
MODEL_PATH = "yolov8n.pt"   # swap to yolov8m.pt for higher accuracy

FRAME_W, FRAME_H = 1260, 720
CONF_THRESHOLD   = 0.35
FPS              = 60        # assumed / read from video

# Calibration: how many real-world metres one pixel represents.
# Measure a known object in your video to calibrate properly.
# Default: 0.05 m/px (≈ 3 m lane width ÷ 60 px lane width on screen)
METRES_PER_PIXEL = 0.05

# Two-wheeler COCO class id (motorcycle = 3, bicycle = 1)
TWO_WHEELER_CLASSES = {1, 3}

# ── Thresholds (tune to your video / road type) ───────────────────────────
SPEED_LIMIT_MPS       = 14.0   # ~50 km/h city limit
HARD_BRAKE_MPS2       = 4.0    # deceleration threshold  (m/s²)
AGGRESSIVE_ACCEL_MPS2 = 3.5    # acceleration threshold  (m/s²)
LANE_WEAVE_PX         = 30     # lateral shift (px) that counts as a weave

# ── Score weights (must sum ≤ 100) ───────────────────────────────────────
W_HARD_BRAKE   = 10   # points lost per hard-brake event
W_AGGR_ACCEL   = 7    # points lost per aggressive-accel event
W_WEAVE        = 5    # points lost per weave / lane-change event
W_OVER_SPEED   = 0.5  # points lost per second above speed limit

HISTORY_LEN = 20   # frames kept per tracker for smoothing

# Minimum frames between consecutive event detections (prevents same event
# firing repeatedly due to bounding-box jitter)
EVENT_COOLDOWN_FRAMES = 12

# ─────────────────────────── TRACKER ─────────────────────────────────────── #

class CentroidTracker:
    """
    Lightweight centroid tracker.
    Matches detections to existing tracks by nearest-centroid distance.
    Re-uses IDs so behavior history is preserved across frames.
    """

    def __init__(self, max_disappeared: int = 20, max_distance: int = 80):
        self.next_id       = 0
        self.objects       = {}          # id → centroid (cx, cy)
        self.disappeared   = defaultdict(int)
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, obj_id):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections):
        """
        detections: list of (cx, cy) tuples
        Returns: dict {id: (cx, cy)}
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        if len(self.objects) == 0:
            for d in detections:
                self.register(d)
            return self.objects

        obj_ids       = list(self.objects.keys())
        obj_centroids = list(self.objects.values())

        # Pairwise Euclidean distance matrix
        dist_matrix = np.array([
            [math.hypot(d[0] - o[0], d[1] - o[1])
             for o in obj_centroids]
            for d in detections
        ])

        # Greedy matching (row = detection, col = track)
        rows = dist_matrix.min(axis=1).argsort()
        cols = dist_matrix.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for r, c in zip(rows, cols):
            if r in used_rows or c in used_cols:
                continue
            if dist_matrix[r, c] > self.max_distance:
                continue
            obj_id = obj_ids[c]
            self.objects[obj_id] = detections[r]
            self.disappeared[obj_id] = 0
            used_rows.add(r); used_cols.add(c)

        # Unmatched detections → new tracks
        for r in range(len(detections)):
            if r not in used_rows:
                self.register(detections[r])

        # Unmatched tracks → increment disappeared
        for c in range(len(obj_ids)):
            if c not in used_cols:
                obj_id = obj_ids[c]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        return self.objects


# ─────────────────────────── BEHAVIOR ENGINE ─────────────────────────────── #

class RiderBehavior:
    """
    Maintains per-rider kinematics and score for one tracked vehicle.

    Key fixes vs naive implementation:
    ─────────────────────────────────
    1. Position smoothing  – raw centroid fed through its own EMA before any
       kinematic calculation, killing bounding-box jitter completely.
    2. Speed smoothing     – low-alpha EMA (0.15) + median-of-window filter
       so a single outlier frame cannot produce a fake speed spike.
    3. Event cooldowns     – a hard brake / aggressive accel / weave can only
       be *counted once* per EVENT_COOLDOWN_FRAMES window.  Without this,
       one real brake event fires 10-15 consecutive frames → instant zero.
    4. Weave uses smoothed position – not raw centroid – so box wobble of
       ±5-10 px never crosses the LANE_WEAVE_PX threshold.
    """

    def __init__(self, rider_id: int, fps: float):
        self.rider_id = rider_id
        self.fps      = fps
        self.dt       = 1.0 / fps

        # Smoothed position accumulator (EMA on cx, cy separately)
        self._smooth_cx: float | None = None
        self._smooth_cy: float | None = None
        POSITION_ALPHA = 0.25          # lower = smoother, more lag
        self._pos_alpha = POSITION_ALPHA

        # Rolling smoothed-position history
        self.positions_px: deque = deque(maxlen=HISTORY_LEN)

        # Derived metrics (public, shown on HUD)
        self.speed_mps        = 0.0
        self.accel_mps2       = 0.0
        self.lateral_disp_px  = 0.0

        # Penalty counters
        self.hard_brake_count   = 0
        self.aggr_accel_count   = 0
        self.weave_count        = 0
        self.over_speed_seconds = 0.0

        # Speed history for median smoothing
        self._raw_speeds: deque = deque(maxlen=HISTORY_LEN)
        self._prev_smooth_speed = 0.0

        # Cooldown counters: frames remaining before next event can be counted
        self._brake_cooldown  = 0
        self._accel_cooldown  = 0
        self._weave_cooldown  = 0

        self.score = 100.0

    # ── helpers ────────────────────────────────────────────────────────────

    def _smooth_position(self, cx: int, cy: int):
        """EMA on raw centroid → removes bounding-box jitter."""
        if self._smooth_cx is None:
            self._smooth_cx = float(cx)
            self._smooth_cy = float(cy)
        else:
            a = self._pos_alpha
            self._smooth_cx = a * cx + (1 - a) * self._smooth_cx
            self._smooth_cy = a * cy + (1 - a) * self._smooth_cy
        return self._smooth_cx, self._smooth_cy

    def _tick_cooldowns(self):
        self._brake_cooldown = max(0, self._brake_cooldown - 1)
        self._accel_cooldown = max(0, self._accel_cooldown - 1)
        self._weave_cooldown = max(0, self._weave_cooldown - 1)

    # ── called each frame ──────────────────────────────────────────────────

    def update(self, cx: int, cy: int):
        self._tick_cooldowns()

        # Step 1 – smooth the raw centroid
        scx, scy = self._smooth_position(cx, cy)
        self.positions_px.append((scx, scy))

        if len(self.positions_px) < 3:   # need a few frames to warm up
            return

        p_prev = self.positions_px[-2]
        p_curr = self.positions_px[-1]

        # ── Speed: pixel displacement of *smoothed* centroid ──────────────
        disp_px   = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        raw_speed = disp_px * METRES_PER_PIXEL * self.fps

        # Accumulate raw speeds, take median to kill outlier spikes
        self._raw_speeds.append(raw_speed)
        median_speed = float(np.median(self._raw_speeds))

        # Light EMA on top of the median (alpha=0.15 → very smooth)
        SPEED_ALPHA    = 0.15
        self.speed_mps = (SPEED_ALPHA * median_speed
                          + (1 - SPEED_ALPHA) * self._prev_smooth_speed)

        # ── Acceleration ──────────────────────────────────────────────────
        self.accel_mps2         = ((self.speed_mps - self._prev_smooth_speed)
                                   / self.dt)
        self._prev_smooth_speed = self.speed_mps

        # ── Lateral displacement (using smoothed cx) ──────────────────────
        self.lateral_disp_px = abs(p_curr[0] - p_prev[0])

        # ── Event detection with cooldown guards ──────────────────────────
        if self.accel_mps2 < -HARD_BRAKE_MPS2 and self._brake_cooldown == 0:
            self.hard_brake_count += 1
            self._brake_cooldown   = EVENT_COOLDOWN_FRAMES

        if self.accel_mps2 > AGGRESSIVE_ACCEL_MPS2 and self._accel_cooldown == 0:
            self.aggr_accel_count += 1
            self._accel_cooldown   = EVENT_COOLDOWN_FRAMES

        if self.lateral_disp_px > LANE_WEAVE_PX and self._weave_cooldown == 0:
            self.weave_count     += 1
            self._weave_cooldown  = EVENT_COOLDOWN_FRAMES

        if self.speed_mps > SPEED_LIMIT_MPS:
            self.over_speed_seconds += self.dt

        # ── Score ─────────────────────────────────────────────────────────
        penalty = (
            self.hard_brake_count   * W_HARD_BRAKE   +
            self.aggr_accel_count   * W_AGGR_ACCEL   +
            self.weave_count        * W_WEAVE         +
            self.over_speed_seconds * W_OVER_SPEED
        )
        self.score = max(0.0, min(100.0, 100.0 - penalty))

    # ── helpers ────────────────────────────────────────────────────────────

    def speed_kmh(self) -> float:
        return self.speed_mps * 3.6

    def score_label(self) -> str:
        if self.score >= 80: return "GOOD"
        if self.score >= 50: return "FAIR"
        return "POOR"

    def score_color(self):
        if self.score >= 80: return (0, 200, 0)
        if self.score >= 50: return (0, 165, 255)
        return (0, 0, 220)


# ─────────────────────────── HUD RENDERING ───────────────────────────────── #

def draw_score_hud(frame, rider: RiderBehavior, x1, y1, x2, y2):
    """Draw bounding box + score card for one rider."""
    color = rider.score_color()

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Score badge (top-left corner of box)
    badge_text  = f"#{rider.rider_id}  {rider.score:.0f}/100"
    label_text  = rider.score_label()
    speed_text  = f"{rider.speed_kmh():.1f} km/h"
    accel_text  = f"a={rider.accel_mps2:+.1f} m/s²"

    tx, ty = x1, max(y1 - 60, 0)
    for i, (txt, clr) in enumerate([
        (badge_text, color),
        (label_text, color),
        (speed_text, (255, 255, 255)),
        (accel_text, (200, 200, 200)),
    ]):
        cv2.putText(frame, txt, (tx, ty + i * 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, clr, 1, cv2.LINE_AA)


def draw_legend(frame):
    """Bottom-left legend / stats panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, FRAME_H - 80), (280, FRAME_H), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    lines = [
        ("DRIVING BEHAVIOR SCORER",     (0, 220, 0)),
        ("Green >=80  Orange >=50  Red <50", (180, 180, 180)),
        (f"Brake thr: {HARD_BRAKE_MPS2} m/s²  "
         f"Accel thr: {AGGRESSIVE_ACCEL_MPS2} m/s²",       (180, 180, 180)),
        (f"Speed limit: {SPEED_LIMIT_MPS*3.6:.0f} km/h  "
         f"Weave thr: {LANE_WEAVE_PX} px", (180, 180, 180)),
    ]
    for i, (txt, clr) in enumerate(lines):
        cv2.putText(frame, txt, (6, FRAME_H - 65 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)


def draw_summary_panel(frame, behaviors: dict):
    """Top-right panel: table of all active riders."""
    if not behaviors:
        return
    overlay = frame.copy()
    panel_w = 230
    cv2.rectangle(overlay, (FRAME_W - panel_w, 0),
                  (FRAME_W, min(len(behaviors) * 20 + 30, FRAME_H)),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "ID   Score  Speed   Events",
                (FRAME_W - panel_w + 4, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 0), 1)

    for i, (rid, b) in enumerate(behaviors.items()):
        events = b.hard_brake_count + b.aggr_accel_count + b.weave_count
        txt = (f"#{rid:<3} {b.score:5.1f}  "
               f"{b.speed_kmh():5.1f}   {events}")
        cv2.putText(frame, txt,
                    (FRAME_W - panel_w + 4, 35 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    b.score_color(), 1, cv2.LINE_AA)


# ─────────────────────────── MAIN LOOP ───────────────────────────────────── #

def main():
    model   = YOLO(MODEL_PATH)
    cap     = cv2.VideoCapture(VIDEO_PATH)

    # Use actual FPS if available
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = video_fps if video_fps > 0 else FPS
    print(f"[INFO] Video FPS: {fps:.1f}")

    tracker   = CentroidTracker(max_disappeared=25, max_distance=90)
    behaviors: dict[int, RiderBehavior] = {}

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # ── Collect detections for two-wheelers only ───────────────────────
        detections_centroids = []
        detection_boxes      = {}   # centroid → (x1,y1,x2,y2)

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in TWO_WHEELER_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            detections_centroids.append((cx, cy))
            detection_boxes[(cx, cy)] = (x1, y1, x2, y2)

        # ── Update tracker ─────────────────────────────────────────────────
        tracked = tracker.update(detections_centroids)

        # ── Update behavior engine per tracked rider ───────────────────────
        for rid, (cx, cy) in tracked.items():
            if rid not in behaviors:
                behaviors[rid] = RiderBehavior(rid, fps)
            behaviors[rid].update(cx, cy)

        # ── Remove stale behavior records ──────────────────────────────────
        active_ids = set(tracked.keys())
        stale      = [k for k in behaviors if k not in active_ids]
        for k in stale:
            del behaviors[k]

        # ── Draw HUD ───────────────────────────────────────────────────────
        for rid, (cx, cy) in tracked.items():
            # Find closest bounding box to this centroid
            if not detection_boxes:
                continue
            closest_key = min(
                detection_boxes,
                key=lambda k: math.hypot(k[0]-cx, k[1]-cy)
            )
            x1, y1, x2, y2 = detection_boxes[closest_key]
            if rid in behaviors:
                draw_score_hud(frame, behaviors[rid], x1, y1, x2, y2)

        draw_legend(frame)
        draw_summary_panel(frame, behaviors)

        # Frame counter
        cv2.putText(frame, f"Frame {frame_count}",
                    (FRAME_W - 120, FRAME_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

        cv2.imshow("Two-Wheeler Driving Behavior Score  [ESC to quit]", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # ── Final report ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  FINAL DRIVING BEHAVIOR REPORT")
    print("="*60)
    print(f"{'ID':<6} {'Score':>7} {'SpeedEvents':>12} "
          f"{'HardBrakes':>12} {'Weaves':>8}")
    print("-"*60)
    for rid, b in sorted(behaviors.items()):
        events = b.hard_brake_count + b.aggr_accel_count + b.weave_count
        print(f"#{rid:<5} {b.score:>7.1f} {events:>12} "
              f"{b.hard_brake_count:>12} {b.weave_count:>8}  "
              f"[{b.score_label()}]")
    print("="*60)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()