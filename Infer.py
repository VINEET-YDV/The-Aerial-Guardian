import os
import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
from IPython.display import Video, display

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "C:\Users\vinee\OneDrive\Desktop\project\best.onnx"

# CHANGE THIS to your dataset path
VISDRONE_SEQ_ROOT = Path("C:\Users\vinee\OneDrive\Desktop\project\VisDrone2019-MOT-val\sequences")

OUT_DIR = Path("/kaggle/working/inference_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_SIZE = 1280
CONF_THRESH = 0.50
SEQ_NAME = None   # put a sequence name here like "uav0000086_00000_v" or keep None for first sequence

# =========================================================
# SIMPLE BYTE TRACKER
# =========================================================
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])
    union = area1 + area2 - inter + 1e-6
    return inter / union

class Track:
    def __init__(self, box, score, track_id):
        self.box = box
        self.score = score
        self.id = track_id
        self.age = 0
        self.hits = 1

class BYTETrackerSimple:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        updated_tracks = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(self.tracks)))

        if len(self.tracks) > 0 and len(detections) > 0:
            iou_matrix = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)

            for t, trk in enumerate(self.tracks):
                for d, det in enumerate(detections):
                    iou_matrix[t, d] = iou(trk.box, det[:4])

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            matched = []
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_thresh:
                    matched.append((r, c))

            unmatched_trks = [t for t in range(len(self.tracks)) if t not in [m[0] for m in matched]]
            unmatched_dets = [d for d in range(len(detections)) if d not in [m[1] for m in matched]]

            for t, d in matched:
                self.tracks[t].box = detections[d][:4]
                self.tracks[t].score = detections[d][4]
                self.tracks[t].age = 0
                self.tracks[t].hits += 1
                updated_tracks.append(self.tracks[t])

        for t in unmatched_trks:
            self.tracks[t].age += 1
            if self.tracks[t].age <= self.max_age:
                updated_tracks.append(self.tracks[t])

        for d in unmatched_dets:
            new_track = Track(detections[d][:4], detections[d][4], self.next_id)
            self.next_id += 1
            updated_tracks.append(new_track)

        self.tracks = updated_tracks

        results = []
        for trk in self.tracks:
            results.append({
                "id": trk.id,
                "bbox": trk.box,
                "score": trk.score
            })
        return results

# =========================================================
# LOAD MODEL
# =========================================================
model = YOLO(MODEL_PATH)
tracker = BYTETrackerSimple(iou_thresh=0.3, max_age=30)

# =========================================================
# PICK A SEQUENCE
# =========================================================
seq_dirs = sorted([d for d in VISDRONE_SEQ_ROOT.iterdir() if d.is_dir()])

if len(seq_dirs) == 0:
    raise ValueError("No sequences found.")

if SEQ_NAME is None:
    seq_dir = seq_dirs[0]
else:
    seq_dir = VISDRONE_SEQ_ROOT / SEQ_NAME

print(f"Running inference on sequence: {seq_dir.name}")

img_paths = sorted(seq_dir.glob("*.jpg"))
if len(img_paths) == 0:
    raise ValueError(f"No images found in {seq_dir}")

# =========================================================
# VIDEO WRITER
# =========================================================
first = cv2.imread(str(img_paths[0]))
H, W = first.shape[:2]

out_video_path = OUT_DIR / f"{seq_dir.name}_tracked.mp4"
writer = cv2.VideoWriter(
    str(out_video_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    20.0,
    (W, H)
)

# =========================================================
# RUN INFERENCE
# =========================================================
total_time = 0
total_frames = 0

for frame_idx, img_path in enumerate(img_paths, start=1):
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue

    t0 = time.time()

    results = model(frame, imgsz=INPUT_SIZE, conf=CONF_THRESH, verbose=False)
    detections = []

    boxes = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy()

        for box, conf, cls in zip(xyxy, confs, clss):
            if int(cls) != 0:  # class 0 = person
                continue
            x1, y1, x2, y2 = box
            detections.append([x1, y1, x2, y2, conf])

    tracks = tracker.update(detections)

    for trk in tracks:
        x1, y1, x2, y2 = map(int, trk["bbox"])
        tid = trk["id"]
        score = trk["score"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"ID {tid} {score:.2f}", (x1, max(20, y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    dt = time.time() - t0
    total_time += dt
    total_frames += 1

    fps_now = 1 / (dt + 1e-6)
    avg_fps = total_frames / total_time

    cv2.putText(frame, f"FPS: {fps_now:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"AVG FPS: {avg_fps:.1f}", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    writer.write(frame)

writer.release()

print(f"Saved video: {out_video_path}")
print(f"Average FPS: {avg_fps:.2f}")