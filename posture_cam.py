#!/usr/bin/env python3
"""
posture_cam.py

Real-time posture classification (sleeping / sitting / standing) from a live webcam
using Ultralytics YOLO Pose (single-pass person detection + 17-keypoint pose).

Usage
------
$ python posture_cam.py
    --camera 0                   # camera index (default 0)
    --src /path/or/url           # optional: file/RTSP/HTTP source; overrides --camera
    --model yolov8n-pose.pt      # model path or name (default)
    --conf 0.25                  # detection confidence threshold
    --imgsz 640                  # inference image size
    --display True               # show OpenCV window (default True)
    --headless                   # run without window (implies --display False)
    --standing_torso_deg 25      # torso angle to vertical (< this means near-vertical)
    --sitting_knee_deg 100       # knee flexion threshold for sitting (deg)
    --sitting_hip_deg 120        # hip flexion threshold for sitting (deg)
    --sleeping_torso_deg 60      # torso angle to vertical (> this means near-horizontal)
    --min_keypoints 8            # min visible KP to attempt angles
    --max-persons 1              # consider up to N people (default 1)
    --select largest             # 'largest' bbox area or 'highest-conf' selection policy
    --log-level INFO             # DEBUG | INFO | WARNING | ERROR
    --log-file posture.log       # optional file logging
    --csv out.csv                # optional CSV for posture/time/features
    --write-reqs                 # write requirements.txt alongside this script

Dependencies
------------
- Python 3.9+
- ultralytics (YOLOv8+)
- opencv-python
- numpy
- torch (installed by ultralytics, used for CUDA detection)

Install:
    python -m pip install -r requirements.txt
On first use, Ultralytics will auto-download the model weights (yolov8n-pose.pt) if not present.
Subsequent runs work fully offline with cached weights.

Algorithm (high level)
----------------------
1) Person detection + pose: YOLO Pose returns bboxes and 17 COCO keypoints in pixels.
   We select one person (by largest bbox or highest confidence).
2) Features from keypoints:
   - Torso inclination: angle (deg) between the vector (shoulder_center → hip_center) and the vertical axis.
     Small angle ⇒ torso vertical; large angle ⇒ torso horizontal.
   - Vertical spread: (max_y - min_y) of detected keypoints normalized by the person bbox height in the frame.
     Low spread ⇒ body stretched horizontally (typical lying posture).
   - Knee & hip angles: knee flexion at each knee (angle between thigh and shank).
     Hip flexion at each hip (angle between torso and thigh).
3) Rule-based posture classification with adjustable thresholds:
   - Standing: torso near-vertical AND legs near-straight (knee angles large) AND vertical spread high.
   - Sitting: torso near-vertical AND knees flexed (and/or hips flexed) AND moderate vertical spread.
   - Sleeping: torso near-horizontal AND low vertical spread.
   Missing measures fall back to weaker heuristics (bbox aspect + spread). Otherwise "uncertain".
4) Confidence in [0,1]: computed as a soft score from distance to decision boundaries via logistic mappings
   (e.g., sigmoid of how far an angle is past/below a threshold). We combine component confidences conservatively
   (min/product) per hypothesis, then report the highest hypothesis score. If the top score < 0.5, return "uncertain".
5) UI: live OpenCV window with bbox, skeleton, posture + confidence, FPS, and a tiny legend (q: quit, s: snapshot).
   Snapshots are saved (with overlays) to ./snapshots/ when pressing 's'.

Edge cases
----------
- No person: returns "no-person" (confidence 0.0).
- Partial person / missing keypoints: gracefully degrade or "uncertain".
- Multiple people: selection policy picks which one to classify.
- Camera disconnects: handled gracefully (log warning and exit loop).
- Performance: yolov8n-pose at 640 targets ≥20 FPS on modern CPU; faster with CUDA GPU if available.

Acceptance checks (manual)
--------------------------
- Full-body person standing upright: label "standing" with confidence > 0.6.
- Person seated on a chair: label "sitting" with confidence > 0.6.
- Person lying on a couch/bed: label "sleeping" with confidence > 0.6.
- Empty scene: label "no-person".

CSV logging (optional)
----------------------
When --csv is provided, logs per-frame rows with:
timestamp, posture, confidence, fps, torso_deg, knee_l, knee_r, hip_l, hip_r, vertical_spread, bbox_w, bbox_h.

"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import logging
import math
import os
import sys
import time
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from ultralytics import YOLO
import cv2



# ----------------------------
# Utility & Geometry functions
# ----------------------------

def sigmoid(x: float) -> float:
    """Numerically-stable logistic sigmoid."""
    # Clamp to avoid overflow
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    """
    Angle (degrees) between 2D vectors v1 and v2 in [0, 180].
    """
    v1 = np.array(v1, dtype=float)
    v2 = np.array(v2, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return float("nan")
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cosang))


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-6) -> float:
    """
    Returns the interior angle ABC (degrees) at point B given three points A-B-C.
    """
    ab = a - b
    cb = c - b
    return angle_between(ab, cb, eps=eps)


def valid_point(p: Optional[np.ndarray]) -> bool:
    return p is not None and np.isfinite(p).all()


# COCO keypoint indices for YOLOv8 Pose:
# 0:nose, 1:leye, 2:reye, 3:lear, 4:rear, 5:lshoulder, 6:rshoulder, 7:lelbow, 8:relbow,
# 9:lwrist, 10:rwrist, 11:lhip, 12:rhip, 13:lknee, 14:rknee, 15:lankle, 16:rankle
KP = {
    "l_shoulder": 5, "r_shoulder": 6,
    "l_hip": 11, "r_hip": 12,
    "l_knee": 13, "r_knee": 14,
    "l_ankle": 15, "r_ankle": 16
}

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # shoulders-arms
    (5, 11), (6, 12), (11, 12),               # torso/hips
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4)            # head
]


# ----------------------------
# Data containers
# ----------------------------

@dataclasses.dataclass
class PoseFeatures:
    torso_deg: float = float("nan")   # torso inclination vs vertical
    knee_l_deg: float = float("nan")
    knee_r_deg: float = float("nan")
    hip_l_deg: float = float("nan")
    hip_r_deg: float = float("nan")
    vertical_spread: float = float("nan")
    bbox_w: float = float("nan")
    bbox_h: float = float("nan")


@dataclasses.dataclass
class PostureResult:
    label: str
    confidence: float
    features: PoseFeatures


# ----------------------------
# Pose Detector
# ----------------------------

class PoseDetector:
    def __init__(self, model_path: str, conf: float, imgsz: int, device: Optional[str] = None, log: Optional[logging.Logger] = None):
        self.log = log or logging.getLogger("PoseDetector")
        self.model_path = model_path
        self.conf = float(conf)
        self.imgsz = int(imgsz)

        # Device selection
        if device is None:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.log.info(f"Using device: {self.device}")

        # Load model
        try:
            self.model = YOLO(self.model_path)
            # Warmup (small tensor once helps compile graph on first frame)
            self.model.to(self.device)
            self.log.info(f"Loaded model: {self.model_path}")
        except Exception as e:
            self.log.exception("Failed to load YOLO model.")
            raise

    def infer(self, frame_bgr: np.ndarray):
        """
        Run pose inference on a BGR frame.
        Returns ultralytics Results list (len=1) with .boxes and .keypoints.
        """
        try:
            # Ultralytics accepts numpy BGR directly
            results = self.model.predict(
                source=frame_bgr,
                conf=self.conf,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )
            return results
        except Exception:
            self.log.exception("Inference failed.")
            return None


# ----------------------------
# Posture Classifier
# ----------------------------

class PostureClassifier:
    def __init__(
        self,
        standing_torso_deg: float = 25.0,
        sitting_knee_deg: float = 100.0,
        sitting_hip_deg: float = 120.0,
        sleeping_torso_deg: float = 60.0,
        min_keypoints: int = 8,
        log: Optional[logging.Logger] = None
    ):
        self.standing_torso_deg = float(standing_torso_deg)
        self.sitting_knee_deg = float(sitting_knee_deg)
        self.sitting_hip_deg = float(sitting_hip_deg)
        self.sleeping_torso_deg = float(sleeping_torso_deg)
        self.min_keypoints = int(min_keypoints)
        self.log = log or logging.getLogger("PostureClassifier")

        # Fixed helper thresholds (reasonable defaults)
        self.standing_knee_min = 150.0  # legs close to straight
        self.vspread_standing_min = 0.55
        self.vspread_sleeping_max = 0.40
        self.vspread_sitting_range = (0.35, 0.85)

        # Sigmoid softness (bigger -> softer)
        self.ang_soft = 8.0
        self.vspread_soft = 12.0

    @staticmethod
    def _center(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not valid_point(a) or not valid_point(b):
            return None
        return (a + b) / 2.0

    @staticmethod
    def _count_visible(kp_conf: Optional[np.ndarray], thr: float = 0.3) -> int:
        if kp_conf is None:
            return 0
        return int((kp_conf > thr).sum())

    def _compute_features(
        self,
        keypoints_xy: np.ndarray,      # (17, 2) pixel coords
        keypoints_conf: Optional[np.ndarray],  # (17,)
        bbox_xyxy: np.ndarray          # (4,)
    ) -> PoseFeatures:
        # Extract needed joints (gracefully handle NaN)
        def p(idx: int) -> Optional[np.ndarray]:
            pt = keypoints_xy[idx] if 0 <= idx < keypoints_xy.shape[0] else np.array([np.nan, np.nan])
            if not np.isfinite(pt).all():
                return None
            return pt

        l_sh = p(KP["l_shoulder"])
        r_sh = p(KP["r_shoulder"])
        l_hip = p(KP["l_hip"])
        r_hip = p(KP["r_hip"])
        l_knee = p(KP["l_knee"])
        r_knee = p(KP["r_knee"])
        l_ankle = p(KP["l_ankle"])
        r_ankle = p(KP["r_ankle"])

        shoulder_c = self._center(l_sh, r_sh)
        hip_c = self._center(l_hip, r_hip)

        torso_deg = float("nan")
        if valid_point(shoulder_c) and valid_point(hip_c):
            vec = hip_c - shoulder_c  # downwards ideally
            torso_deg = angle_between(vec, np.array([0.0, 1.0]))  # angle vs vertical

        knee_l = joint_angle(hip=l_hip, knee=l_knee, ankle=l_ankle) if False else float("nan")  # placeholder for clarity
        # Compute knee angles (A-B-C with B at knee)
        if valid_point(l_hip) and valid_point(l_knee) and valid_point(l_ankle):
            knee_l = joint_angle(l_hip, l_knee, l_ankle)
        if valid_point(r_hip) and valid_point(r_knee) and valid_point(r_ankle):
            knee_r = joint_angle(r_hip, r_knee, r_ankle)
        else:
            knee_r = float("nan")

        # Hip flexion: angle at hip between torso (shoulder->hip) and thigh (knee->hip)
        hip_l = float("nan")
        hip_r = float("nan")
        if valid_point(l_sh) and valid_point(l_hip) and valid_point(l_knee):
            # angle between vectors (shoulder->hip) and (knee->hip) with vertex at hip
            hip_l = joint_angle(l_sh, l_hip, l_knee)
        if valid_point(r_sh) and valid_point(r_hip) and valid_point(r_knee):
            hip_r = joint_angle(r_sh, r_hip, r_knee)

        # Vertical spread
        bbox_w = float(bbox_xyxy[2] - bbox_xyxy[0])
        bbox_h = float(bbox_xyxy[3] - bbox_xyxy[1])
        vs = float("nan")
        if bbox_h > 1:
            ys = keypoints_xy[:, 1]
            ys = ys[np.isfinite(ys)]
            if ys.size > 0:
                vs = float((ys.max() - ys.min()) / bbox_h)
                vs = float(np.clip(vs, 0.0, 2.0))

        return PoseFeatures(
            torso_deg=torso_deg,
            knee_l_deg=float(knee_l),
            knee_r_deg=float(knee_r),
            hip_l_deg=float(hip_l),
            hip_r_deg=float(hip_r),
            vertical_spread=vs,
            bbox_w=bbox_w,
            bbox_h=bbox_h
        )

    def _score_standing(self, f: PoseFeatures) -> float:
        # Components: torso near-vertical, knees straight, vertical spread high
        comps: List[float] = []
        if np.isfinite(f.torso_deg):
            comps.append(sigmoid((self.standing_torso_deg - f.torso_deg) / self.ang_soft))
        if np.isfinite(f.knee_l_deg) and np.isfinite(f.knee_r_deg):
            avg_knee = 0.5 * (f.knee_l_deg + f.knee_r_deg)
            comps.append(sigmoid((avg_knee - self.standing_knee_min) / self.ang_soft))
        if np.isfinite(f.vertical_spread):
            comps.append(sigmoid((f.vertical_spread - self.vspread_standing_min) / (self.vspread_soft * 0.5)))
        if not comps:
            return 0.0
        return float(min(comps))

    def _score_sitting(self, f: PoseFeatures) -> float:
        comps: List[float] = []
        # Torso near vertical
        if np.isfinite(f.torso_deg):
            comps.append(sigmoid((self.standing_torso_deg - f.torso_deg) / self.ang_soft))
        # Knees flexed
        flex_scores = []
        for k in (f.knee_l_deg, f.knee_r_deg):
            if np.isfinite(k):
                flex_scores.append(sigmoid((self.sitting_knee_deg - k) / self.ang_soft))
        if flex_scores:
            comps.append(float(np.nanmean(flex_scores)))
        # Hips flexed (optional but helps)
        hip_scores = []
        for h in (f.hip_l_deg, f.hip_r_deg):
            if np.isfinite(h):
                hip_scores.append(sigmoid((self.sitting_hip_deg - h) / self.ang_soft))
        if hip_scores:
            comps.append(float(np.nanmean(hip_scores)))
        # Vertical spread moderate
        if np.isfinite(f.vertical_spread):
            lo, hi = self.vspread_sitting_range
            # reward closeness inside [lo, hi]; use product of sigmoids to form a "band-pass"
            inside = sigmoid((f.vertical_spread - lo) / (self.vspread_soft * 0.5)) * sigmoid((hi - f.vertical_spread) / (self.vspread_soft * 0.5))
            comps.append(inside)
        if not comps:
            return 0.0
        return float(min(comps))

    def _score_sleeping(self, f: PoseFeatures) -> float:
        comps: List[float] = []
        if np.isfinite(f.torso_deg):
            comps.append(sigmoid((f.torso_deg - self.sleeping_torso_deg) / self.ang_soft))
        if np.isfinite(f.vertical_spread):
            comps.append(sigmoid(((self.vspread_sleeping_max - f.vertical_spread)) / (self.vspread_soft * 0.5)))
        # Fallback on bbox aspect ratio if angles missing
        if (not comps) and np.isfinite(f.bbox_w) and np.isfinite(f.bbox_h) and f.bbox_h > 0:
            aspect = f.bbox_w / max(f.bbox_h, 1e-3)
            # wider than tall mildly supports sleeping
            comps.append(sigmoid((aspect - 1.2)))  # soft, dimensionless
        if not comps:
            return 0.0
        return float(min(comps))

    def classify(
        self,
        keypoints_xy: np.ndarray,      # (17,2)
        keypoints_conf: Optional[np.ndarray],  # (17,)
        bbox_xyxy: np.ndarray          # (4,)
    ) -> PostureResult:
        # Quick visibility gate
        visible = self._count_visible(keypoints_conf, thr=0.3)
        if visible < 2:
            f = self._compute_features(keypoints_xy, keypoints_conf, bbox_xyxy)
            return PostureResult("no-person", 0.0, f)

        f = self._compute_features(keypoints_xy, keypoints_conf, bbox_xyxy)

        # If too few keypoints, try very soft heuristics
        if visible < self.min_keypoints:
            # Heuristic: use vertical spread + bbox aspect
            scores = {
                "sleeping": self._score_sleeping(f),
                "standing": self._score_standing(f) * 0.7,
                "sitting": self._score_sitting(f) * 0.7
            }
        else:
            scores = {
                "sleeping": self._score_sleeping(f),
                "sitting": self._score_sitting(f),
                "standing": self._score_standing(f)
            }

        # Pick top hypothesis
        labels = list(scores.keys())
        vals = [scores[k] for k in labels]
        idx = int(np.argmax(vals))
        label = labels[idx]
        conf = float(vals[idx])

        if conf < 0.5:
            return PostureResult("uncertain", conf, f)
        return PostureResult(label, conf, f)


# ----------------------------
# Visualization
# ----------------------------

class Visualizer:
    def __init__(self, kp_thr: float = 0.3):
        self.kp_thr = kp_thr

    def _kpt_ok(self, pt: np.ndarray, conf: Optional[float]) -> bool:
        if pt is None or not np.isfinite(pt).all():
            return False
        # ignore near-origin placeholders
        if pt[0] <= 1 and pt[1] <= 1:
            return False
        if conf is not None and conf <= self.kp_thr:
            return False
        return True

    def draw_detection(
        self,
        frame: np.ndarray,
        bbox_xyxy: np.ndarray,
        keypoints_xy: np.ndarray,
        keypoints_conf: Optional[np.ndarray]
    ) -> None:
        x1, y1, x2, y2 = bbox_xyxy.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 196, 255), 2)

        # draw skeleton edges only if BOTH endpoints are valid
        for i, j in SKELETON_EDGES:
            if i >= keypoints_xy.shape[0] or j >= keypoints_xy.shape[0]:
                continue
            pi, pj = keypoints_xy[i], keypoints_xy[j]
            ci = float(keypoints_conf[i]) if keypoints_conf is not None else 1.0
            cj = float(keypoints_conf[j]) if keypoints_conf is not None else 1.0
            if self._kpt_ok(pi, ci) and self._kpt_ok(pj, cj):
                cv2.line(frame, (int(pi[0]), int(pi[1])), (int(pj[0]), int(pj[1])), (0, 255, 0), 2)

        # draw keypoints only if valid
        for k, pt in enumerate(keypoints_xy):
            conf = float(keypoints_conf[k]) if keypoints_conf is not None else 1.0
            if self._kpt_ok(pt, conf):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    def overlay_text(
        self,
        frame: np.ndarray,
        posture: str,
        confidence: float,
        fps: float,
        legend: bool = True
    ) -> None:
        h = 22
        x, y = 10, 24
        cv2.putText(frame, f"Posture: {posture} ({confidence:.2f})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 255, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (x, y + h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        if legend:
            cv2.putText(frame, "q: quit, s: save snapshot", (x, y + 2 * h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


# ----------------------------
# Pipeline
# ----------------------------

def select_person(
    boxes_xyxy: np.ndarray, boxes_conf: np.ndarray, policy: str = "largest"
) -> int:
    if boxes_xyxy.size == 0:
        return -1
    if policy == "highest-conf":
        return int(np.argmax(boxes_conf))
    # default largest area
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return int(np.argmax(areas))


def run(args: argparse.Namespace) -> None:
    # Logging
    log = logging.getLogger("posture_cam")
    log.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
        fh.setFormatter(fmt)
        log.addHandler(fh)

    # Startup summary
    device = None
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    display = bool(args.display) and not args.headless

    log.info("========= Configuration =========")
    log.info(f"Source: {'--src='+str(args.src) if args.src is not None else '--camera='+str(args.camera)}")
    log.info(f"Model: {args.model} (conf={args.conf}, imgsz={args.imgsz}, device={device})")
    log.info(f"Display window: {display}")
    log.info(f"Selection policy: {args.select}, max-persons: {args.max_persons}")
    log.info(f"Thresholds: standing_torso={args.standing_torso_deg}°, sitting_knee={args.sitting_knee_deg}°, "
             f"sitting_hip={args.sitting_hip_deg}°, sleeping_torso={args.sleeping_torso_deg}°")
    log.info(f"Min keypoints: {args.min_keypoints}")
    log.info(f"CSV: {args.csv if args.csv else 'None'}")
    log.info("================================")


    # Init model
    detector = PoseDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=device,
        log=logging.getLogger("PoseDetector")
    )

    # Classifier
    classifier = PostureClassifier(
        standing_torso_deg=args.standing_torso_deg,
        sitting_knee_deg=args.sitting_knee_deg,
        sitting_hip_deg=args.sitting_hip_deg,
        sleeping_torso_deg=args.sleeping_torso_deg,
        min_keypoints=args.min_keypoints,
        log=logging.getLogger("PostureClassifier")
    )

    # Video capture
    cap_src = args.src if args.src is not None else int(args.camera)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        log.error(f"Failed to open source: {cap_src}. Exiting.")
        return

    # Prepare CSV
    csv_writer = None
    csv_fh = None
    if args.csv:
        csv_fh = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "timestamp", "posture", "confidence", "fps",
            "torso_deg", "knee_l_deg", "knee_r_deg", "hip_l_deg", "hip_r_deg",
            "vertical_spread", "bbox_w", "bbox_h"
        ])

    vis = Visualizer()

    # FPS tracking
    t_last = time.time()
    frame_times = deque(maxlen=60)
    fps_display = 0.0
    last_log_fps = time.time()

    snapshots_dir = os.path.join(".", "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Camera frame grab failed or stream ended. Exiting loop.")
                break

            t0 = time.time()
            results = detector.infer(frame)
            posture_label = "no-person"
            posture_conf = 0.0
            features = PoseFeatures()

            if results is not None and len(results) > 0:
                r0 = results[0]
                # Expect person class only in pose model; but still parse safely
                boxes_xyxy = r0.boxes.xyxy.cpu().numpy() if r0.boxes is not None else np.zeros((0, 4))
                boxes_conf = r0.boxes.conf.cpu().numpy() if r0.boxes is not None else np.zeros((0,))
                kpts_xy = r0.keypoints.xy.cpu().numpy() if r0.keypoints is not None else np.zeros((0, 17, 2))
                kpts_conf = r0.keypoints.conf.cpu().numpy() if (r0.keypoints is not None and r0.keypoints.conf is not None) else None

                if boxes_xyxy.shape[0] > 0:
                    idx = select_person(boxes_xyxy, boxes_conf, policy=args.select)
                    if idx >= 0:
                        bbox = boxes_xyxy[idx]
                        kxy = kpts_xy[idx]  # (17, 2)
                        kcf = kpts_conf[idx] if kpts_conf is not None else None

                        # Classify
                        result = classifier.classify(kxy, kcf, bbox)
                        posture_label = result.label
                        posture_conf = result.confidence
                        features = result.features

                        # Draw
                        if display:
                            vis.draw_detection(frame, bbox, kxy, kcf)

            # FPS
            t1 = time.time()
            frame_dt = t1 - t0
            frame_times.append(frame_dt)
            if frame_times:
                fps_display = 1.0 / (sum(frame_times) / len(frame_times))

            # Overlay
            if display:
                vis.overlay_text(frame, posture_label, posture_conf, fps_display, legend=True)
                cv2.imshow("PostureCam", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log.info("Quit requested by user.")
                    break
                elif key == ord('s'):
                    snap_path = os.path.join(snapshots_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame)
                    log.info(f"Saved snapshot: {snap_path}")

            # CSV
            if csv_writer is not None:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    posture_label, f"{posture_conf:.4f}", f"{fps_display:.2f}",
                    f"{features.torso_deg:.2f}", f"{features.knee_l_deg:.2f}", f"{features.knee_r_deg:.2f}",
                    f"{features.hip_l_deg:.2f}", f"{features.hip_r_deg:.2f}",
                    f"{features.vertical_spread:.3f}", f"{features.bbox_w:.1f}", f"{features.bbox_h:.1f}"
                ])

            # Periodic logs
            if time.time() - last_log_fps > 5.0:
                log.info(f"Processing FPS ~ {fps_display:.1f} | Posture: {posture_label} ({posture_conf:.2f})")
                last_log_fps = time.time()

            # Debug per-frame (optional)
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Frame posture={posture_label} conf={posture_conf:.3f} "
                          f"torso={features.torso_deg:.1f} knees=({features.knee_l_deg:.1f},{features.knee_r_deg:.1f}) "
                          f"hips=({features.hip_l_deg:.1f},{features.hip_r_deg:.1f}) vs={features.vertical_spread:.2f}")

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl-C).")
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if csv_fh is not None:
            try:
                csv_fh.close()
            except Exception:
                pass
        log.info("Stopped.")


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time posture classification from webcam using YOLO Pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Sources
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--src", type=str, default=None, help="Video/file/RTSP/HTTP source (overrides --camera)")

    # Model config
    p.add_argument("--model", type=str, default="yolov8n-pose.pt", help="YOLO pose model path or name")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    # Display / runtime
    p.add_argument("--display", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True, help="Show OpenCV window")
    p.add_argument("--headless", action="store_true", help="Disable window (for servers / testing)")

    # Thresholds
    p.add_argument("--standing_torso_deg", type=float, default=25.0, help="Max torso angle to vertical for standing")
    p.add_argument("--sitting_knee_deg", type=float, default=100.0, help="Min knee flexion angle threshold for sitting decision")
    p.add_argument("--sitting_hip_deg", type=float, default=120.0, help="Min hip flexion angle threshold for sitting decision")
    p.add_argument("--sleeping_torso_deg", type=float, default=60.0, help="Min torso angle to vertical for sleeping")
    p.add_argument("--min_keypoints", type=int, default=8, help="Min visible keypoints to attempt full angles")

    # Multi-person
    p.add_argument("--max-persons", type=int, default=1, help="Max number of persons to consider (currently selects one)")
    p.add_argument("--select", type=str, choices=("largest", "highest-conf"), default="largest", help="Selection policy when multiple persons are present")

    # Logging
    p.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    p.add_argument("--log-file", type=str, default=None, help="Optional log file path")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output file path")
    p.add_argument("--write-reqs", action="store_true", help="Write requirements.txt to disk and exit")

    args = p.parse_args(argv)

    # If headless explicitly set, force display False
    if args.headless:
        args.display = False

    return args


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()

