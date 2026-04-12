<div align="center">

#  Two-Wheeler Driving Behavior Scorer
### Varroc Eureka 3.0 — Problem Statement 3

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=flat)](https://ultralytics.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

**Real-time computer vision system for accurate two-wheeler riding behavior scoring.**  
Camera-only · No additional hardware · Dashcam + Fixed camera support · 60fps web dashboard

---

<!-- DASHBOARD SCREENSHOT 1 -->
<!-- Replace the block below with your actual screenshot -->
<!--
![Dashboard Live Feed](assets/dashboard_live.png)
-->

> *Screenshot 1 — Live feed with rider tracking, scores, and real-time event overlay*

<!-- DASHBOARD SCREENSHOT 2 -->
<!-- Replace the block below with your actual screenshot -->
<!--
![Session Report](assets/dashboard_report.png)
-->

> *Screenshot 2 — Session report with per-rider behavioral breakdown*

---

</div>

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Scoring Methodology](#scoring-methodology)
- [Speed Estimation](#speed-estimation)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Web Dashboard](#web-dashboard)
- [Camera Modes](#camera-modes)
- [Configuration](#configuration)
- [Technical Details](#technical-details)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Overview

This system provides **accurate, real-time driving behavior scoring** for two-wheelers using only a standard camera — no GPS, no IMU, no radar required. It was built for **Varroc Eureka 3.0, Problem Statement 3**.

The pipeline detects and tracks motorcycles and riders in video footage, computes per-rider behavioral scores (0–100) across six safety metrics, and streams results live to a web dashboard accessible from any browser on the same network.

**Why this approach works:**
- YOLOv8 detects vehicles and riders with high accuracy across dashcam and fixed camera perspectives
- A Constant-Acceleration Kalman filter eliminates YOLO bbox jitter, giving stable speed estimates without GPS
- Fleet-context-aware thresholds prevent traffic slowdowns from penalizing riders
- Self-calibrating MPP (metres-per-pixel) adapts to any camera, mount, or focal length automatically

---

## Key Features

| Feature | Detail |
|---|---|
| **Real-time detection** | YOLOv8n/m — motorcycle (cls 3), bicycle (cls 1), person (cls 0) |
| **Multi-rider tracking** | IoU-first tracker + centroid fallback + Re-ID by proximity |
| **Speed estimation** | CA Kalman filter on ego-compensated centroids, fleet-shared dynamic MPP |
| **6 behavioral metrics** | Hard brake, aggressive accel, lane weave, tailgating, sudden stop, no-helmet |
| **Fleet context** | Dynamic thresholds scale with live traffic deceleration |
| **Helmet detection** | Otsu blob analysis on head region, 30-frame majority vote |
| **Direction classification** | ONCOMING / APPROACH / RECEDING / ALONGSIDE from bbox area slope |
| **Two camera modes** | Dashcam (ego-motion compensation) + Fixed camera (raw centroid) |
| **60fps web dashboard** | React + FastAPI + WebSocket — live video, charts, session report |
| **Zero extra hardware** | Runs on any smartphone dashcam or existing CCTV feed |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DETECTION PIPELINE                          │
│                                                                     │
│  Video Input                                                        │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐                 │
│  │ YOLOv8n │──▶│ IoU Tracker  │──▶│  Re-ID by    │                 │
│  │ detect  │   │ + centroid   │   │  Proximity   │                 │
│  └─────────┘   │  fallback    │   └──────┬───────┘                 │
│                └──────────────┘          │                         │
│                                          ▼                         │
│  ┌──────────────────┐   ┌───────────────────────┐                  │
│  │  Ego-Motion      │──▶│  CA Kalman Filter     │                  │
│  │  Estimator (LK)  │   │  state:[x,y,vx,vy,    │                  │
│  │  [dashcam only]  │   │        ax,ay]         │                  │
│  └──────────────────┘   └──────────┬────────────┘                  │
│                                    │                               │
│                                    ▼                               │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                   RiderBehavior Engine                     │    │
│  │  FleetContext · 6 event detectors · EMA score · Reliability│    │
│  └────────────────────────────┬───────────────────────────────┘    │
│                               │                                    │
│              ┌────────────────┴─────────────────┐                  │
│              ▼                                  ▼                  │
│      OpenCV HUD overlay              FastAPI backend               │
│      (drawn on frame)                (WebSocket /ws)               │
│                                             │                      │
│                                             ▼                      │
│                                    React Web Dashboard             │
│                               (60fps · live charts · report)       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scoring Methodology

Each rider starts at **100 points**. Events deduct points based on severity.

```
Score = 100 − Σ (event_count × weight)
```

Displayed score is EMA-smoothed (α = 0.04) to eliminate flicker.

### Event Weights

| Event | Trigger Condition | Weight |
|---|---|---|
| Hard Brake | acceleration < −2.8 m/s² | 5 pts |
| Aggressive Accel | acceleration > +2.2 m/s² | 3 pts |
| Lane Weave | lateral displacement > 18px | 2 pts |
| Tailgating | bbox area > 9% frame + growth > 0.002/frame | 4 pts |
| Sudden Stop | speed drop below 1.2 m/s within 6 frames | 5 pts |
| No Helmet | Otsu blob < 8% head area, 30-frame majority vote | 8 pts |
| Over-speed | speed > 50 km/h (configurable) | 0.15 pts/sec |

### Fleet-Context Adaptive Thresholds

Brake and acceleration thresholds scale with live traffic conditions:

```
brake_thr = BASE_HARD_BRAKE (2.5) + FLEET_K (0.6) × max(0, −fleet_accel)
accel_thr = BASE_AGGR_ACCEL (2.0) + FLEET_K (0.6) × max(0,  fleet_accel)
```

This prevents mass false-positive events when all vehicles decelerate together in traffic.

### Score Ratings

| Score | Rating |
|---|---|
| 80 – 100 | 🟢 GOOD |
| 50 – 79 | 🟡 FAIR |
| 0 – 49 | 🔴 POOR |
| N/A | ⚡ STUNT (aspect ratio anomaly) |

---

## Speed Estimation

Speed estimation from a monocular camera is an inherently approximate problem — no focal length calibration, GPS, or radar is available. The system achieves stable **relative speed** using three layered techniques:

### 1. Dynamic MPP (Metres Per Pixel)
```
MPP = real_world_width_m / bbox_width_px
```
`real_world_width` is known per class (motorcycle ≈ 0.80m). This self-calibrates for any camera and any mount without manual setup.

**Fleet-shared MPP:** Only vehicles in the 5–30m range (MPP 0.006–0.030) contribute to the fleet median. Close and distant vehicles produce unreliable scale and are excluded.

### 2. Constant-Acceleration Kalman Filter
State vector: `[x, y, vx, vy, ax, ay]`

The CA model (vs the simpler CV model) explicitly tracks acceleration, meaning hard braking and acceleration events are captured in 1–2 frames instead of 5–8. Eliminates the ±10 km/h jitter from YOLO bbox variance.

### 3. Ego-Motion Compensation (Dashcam only)
Lucas-Kanade sparse optical flow on background feature points estimates the camera's own motion each frame. Each vehicle's centroid displacement is corrected by this ego vector before being fed to the Kalman filter, giving **ground-relative velocity** rather than camera-relative.

**Accuracy:** ±5 km/h relative. Absolute ground speed requires GPS or radar.

---

## Project Structure

```
├── dashcam.py              # Detection engine — dashcam / moving camera mode
├── fixed_cam.py            # Detection engine — fixed / CCTV camera mode
├── backend.py              # FastAPI server (WebSocket + REST API)
├── Dashboard.jsx           # React frontend component
├── requirements.txt        # Python dependencies
└── README.md
```

### Key Classes

| Class | File | Purpose |
|---|---|---|
| `KalmanVelocityEstimator` | both | CA Kalman filter — state [x,y,vx,vy,ax,ay] |
| `EgoMotionEstimator` | dashcam.py | LK sparse optical flow for camera motion |
| `IoUTracker` | both | IoU-first tracking + centroid fallback + Re-ID |
| `RiderBehavior` | both | Per-rider state machine — events, score, history |
| `FleetContext` | both | Fleet-wide adaptive thresholds |
| `HelmetDetector` | both | Otsu blob analysis on head region crop |

---

## Installation

### Prerequisites
- Python 3.10+
- Node.js 18+ (for the React dashboard)
- CUDA-capable GPU recommended (runs on CPU, slower)

### Python Backend

```bash
# Clone the repository
git clone https://github.com/Akshay000000/varroc-ps3-behavior-scorer
cd varroc-ps3-behavior-scorer

# Install Python dependencies
pip install -r requirements.txt

# Optional: install ByteTrack for better multi-rider tracking
pip install boxmot --no-deps
pip install lapx filterpy loguru gitpython ftfy gdown
```

### React Frontend

```bash
# Create React app (first time only)
npx create-react-app varroc-dashboard
cd varroc-dashboard

# Install chart library
npm install recharts

# Copy dashboard component
cp ../Dashboard.jsx src/Dashboard.jsx

# Update src/App.js
echo "import Dashboard from './Dashboard'; export default function App() { return <Dashboard />; }" > src/App.js
```

### `requirements.txt`

```
fastapi
uvicorn[standard]
opencv-python
ultralytics
numpy
websockets
python-multipart
```

---

## Usage

### Standalone (no dashboard)

Edit `VIDEO_PATH` in either script, then run:

```bash
# Dashcam footage (bike-mounted / moving camera)
python dashcam.py

# Fixed camera (roadside / overhead / CCTV)
python fixed_cam.py
```

Press `ESC` to stop. A full session report prints to the terminal on exit.

### With Web Dashboard

**Terminal 1 — Backend:**
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd varroc-dashboard && npm start
```

Open `http://localhost:3000` in any browser.

**On the same WiFi network**, open `http://<your-machine-ip>:3000` from any phone or tablet.

---

## Web Dashboard

The dashboard streams processed video and rider data over WebSocket at 60fps using `requestAnimationFrame` for smooth canvas rendering.

### Live Feed Tab
- **Video canvas** — processed frames with YOLO overlays, score bars, HUD labels
- **Rider cards** — per-rider: animated score arc, speed, direction, event counters, mini speed/score charts, confidence bar
- **Speed history chart** — live line chart, one coloured line per active rider
- **Score history chart** — score trend over the last 90 frames

### Session Report Tab
- Auto-populates when you click **Stop**
- Summary metrics: vehicles tracked, duration, average score, no-helmet violations
- Full table: every tracked rider with complete event breakdown

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/start` | Start detection `{"mode":"dashcam","video_path":"..."}` |
| `POST` | `/api/stop` | Stop session and finalise report |
| `GET` | `/api/session` | Full session JSON report |
| `GET` | `/api/status` | Current running status |
| `WS` | `/ws` | Live frame (base64 JPEG) + rider JSON at 60fps |

---

## Camera Modes

### `dashcam.py` — Moving Camera
Best for: bike-mounted cameras, vehicle dashcams, any footage where the camera itself is moving.

| Parameter | Value | Reason |
|---|---|---|
| Confidence threshold | 0.28 | Lower — motion blur reduces detection confidence |
| Ego-motion | ✅ LK optical flow | Compensates camera's own movement |
| `MAX_DISAPPEARED` | 45 frames | Vehicles exit quickly; long coast prevents re-registration |
| Weave threshold | 18px | Looser — ego noise adds variance |
| Kalman Q/R | 0.8 / 28.0 | More process noise — vehicles manoeuvre more |
| Over-speed | RECEDING vehicles exempt | Receding always reads high relative speed |

### `fixed_cam.py` — Stationary Camera
Best for: roadside cameras, overhead CCTV, intersection monitoring.

| Parameter | Value | Reason |
|---|---|---|
| Confidence threshold | 0.35 | Higher — stable background, cleaner detections |
| Ego-motion | ❌ Removed | Camera is stationary |
| `MAX_DISAPPEARED` | 45 frames | Occlusion by other vehicles is common |
| Weave threshold | 14px | Tighter — no ego noise contaminating signal |
| Kalman Q/R | 0.6 / 22.0 | Less process noise — stable camera = cleaner velocity |
| Over-speed | All directions scored | Fixed cam sees absolute road motion |

---

## Configuration

All tunable constants are at the top of each script.

### Key Parameters — `dashcam.py` / `fixed_cam.py`

```python
VIDEO_PATH  = r"path/to/your/video.mp4"
MODEL_PATH  = "yolov8n.pt"    # change to yolov8m.pt for better accuracy

FRAME_W, FRAME_H = 640, 360   # resize resolution
CONF_THRESH      = 0.28        # detection confidence
SPEED_LIMIT_MPS  = 13.9        # 50 km/h — over-speed threshold

# Score weights
W_HARD_BRAKE  = 5
W_AGGR_ACCEL  = 3
W_WEAVE       = 2
W_TAILGATE    = 4
W_SUDDEN_STOP = 5
W_OVER_SPEED  = 0.15  # per second
W_NO_HELMET   = 8

# Event cooldown — minimum frames between repeat events
EVENT_COOLDOWN = 18

# Kalman filter tuning
# PROCESS_NOISE: higher = tracks faster changes, more noise
# MEASURE_NOISE: higher = smoother, slower to react
# dashcam: Q=0.8, R=28.0
# fixed:   Q=0.6, R=22.0
```

### Switching to YOLOv8m (recommended for production)

```python
MODEL_PATH = "yolov8m.pt"
```

Download happens automatically on first run. Gives ~8% better mAP at ~2× compute cost. Runs at 30fps on a mid-range GPU.

---

## Technical Details

### Re-ID by Proximity
Before registering a new track ID, the tracker checks all currently-coasting (disappeared) tracks within `FRAME_DIAGONAL / 5` radius. If a recently-lost track is nearby, it is **resurrected** instead of a new ID being born. This prevents a single vehicle from accumulating 3+ IDs after brief occlusions, which would cause false brake/accel events from the Kalman cold-start.

### Helmet Detection
1. Match each motorcycle track to the nearest person bbox by IoU overlap (threshold: 0.20)
2. Crop the top 32% of the person bbox — the head region
3. Apply Otsu thresholding to find the largest contiguous blob
4. If the blob covers ≥ 8% of the head area → helmet detected
5. Maintain a 30-frame rolling vote; call NO_HELMET only if ≥ 70% of frames show no helmet

The 30-frame / 70% threshold means a single bad crop (poor lighting, motion blur) cannot trigger a false NO_HELMET event.

### Direction Classification
Linear regression on bbox area over the last 10 frames gives an area slope:
- `slope > 0.004` → ONCOMING
- `slope > 0.001` → APPROACH
- `slope < -0.001` → RECEDING
- otherwise → ALONGSIDE

### Reliability Indicator
New tracks are flagged with a reliability score `min(100, frame_count / WARMUP_FRAMES × 30)`. Tracks below 80% reliability show `~` in the leaderboard and an orange confidence bar on their rider card. Event scoring is suppressed during the warmup period.

---

## Limitations

| Limitation | Detail |
|---|---|
| **Speed accuracy** | ±5 km/h relative only. Absolute ground speed requires GPS or radar. |
| **Helmet detection accuracy** | Blob analysis is a heuristic. Performance degrades at night or with very small/distant riders. |
| **Monocular depth** | MPP estimation assumes vehicles are roughly the same known width. Unusual vehicles (cargo bikes, modified scooters) may produce incorrect scale. |
| **Dashcam pitch** | Extreme road bumps cause transient ego-motion spikes that can briefly corrupt speed readings. |

---

## Roadmap

- [x] YOLOv8 detection + IoU tracking
- [x] CA Kalman filter velocity estimation
- [x] Ego-motion compensation (dashcam mode)
- [x] 6-metric behavioral scoring with fleet context
- [x] Helmet detection (Otsu blob analysis)
- [x] Direction classification
- [x] Re-ID by proximity (ByteTrack-equivalent)
- [x] React + FastAPI web dashboard (60fps WebSocket)
- [ ] YOLOv8m fine-tuning on Indian roads dataset (Roboflow)
- [ ] ByteTrack integration (pending boxmot/numpy compatibility fix)
- [ ] Mobile app packaging (React Native)
- [ ] Edge deployment — Jetson Nano / Raspberry Pi 5
- [ ] GPS fusion for absolute speed (optional hardware path)

---

## Built By

**B.Akshay Sriram** — Indian Institute of Information Technology, Kottayam  
B.Tech Computer Science & Engineering — AI and Data Science, 2027

**Varroc Eureka 3.0 · Problem Statement 3**  
*Accurate Driving Behavior Score on a Two-Wheeler*

---

<div align="center">

Made with Python · OpenCV · YOLOv8 · React · FastAPI

</div>
