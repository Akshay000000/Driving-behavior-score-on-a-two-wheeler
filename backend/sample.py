import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(r"./public/video.mp4")

prev_positions = {}

lane_threshold = 50

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 360))

    results = model(frame, conf=0.3)[0]

    current_positions = {}

    for box in results.boxes:
        cls = int(box.cls[0])

        if cls not in [1, 3]:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cx = (x1 + x2) // 2

        track_id = hash((cx, y1)) % 10000
        current_positions[track_id] = cx

        if track_id in prev_positions:
            dx = cx - prev_positions[track_id]

            if abs(dx) > lane_threshold:
                cv2.putText(frame, "Lane Change", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    prev_positions = current_positions

    cv2.imshow("Bike Lane Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()