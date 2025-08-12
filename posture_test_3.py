#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
posture_cam.py — Robust, real-time posture classification focused on Sitting (incl. wheelchair/couch) vs Sleeping,
using Ultralytics YOLO Pose (17 keypoints) with temporal smoothing and per-second status logging.

Quick start
-----------
First time:
    $ python posture_cam.py --write-reqs
    $ pip install -r requirements.txt

Run (binary sitting/sleeping mode by default):
    $ python posture_cam.py

Useful options:
    --mode binary|tri           # default: binary (sitting vs sleeping). 'tri' adds standing.
    --camera-rot 0|90|180|270   # rotate input if your camera is mounted sideways (default 0)
    --status-interval 1.0       # print STATUS line every N seconds (default 1.0)
    --smooth-alpha 0.30         # temporal EMA (higher = faster response, lower = steadier)
    --couch-override            # (default on) allow reclined-with-bent-knees to map to sitting
    --no-couch-override         # disable that behavior
    --model ...                 # override model; else auto-tries: yolo11m/n/s -> yolov8n

Detector basics
---------------
- Opens webcam or source, runs YOLO Pose (single pass person + 17 keypoints).
- Picks one subject (largest bbox or highest-conf).
- Computes features:
    * Torso inclination (shoulder_center → hip_center) vs vertical: 0°=upright, 90°=horizontal.
    * Vertical spread: (max_y − min_y)/bbox_height.
    * Knee flexion (L/R): thigh vs shank, ~180° straight, ~90° bent.
    * Hip flexion (L/R): torso vs thigh, higher while sitting.
    * NEW: Leg-drop: mean(ankle_y − hip_y)/bbox_height; small if legs are forward / elevated (wheelchair/couch).
- Rules with soft sigmoids + EMA smoothing:
    * Sleeping: torso ≳ horizontal AND compact vertical spread.
    * Sitting: torso ≲ vertical AND (knee bent OR hip flexed OR leg-drop small). Couch override nudges near-horizontal torso
      with bent knees / small leg-drop into sitting.
- Confidence: geometric mean of rule sigmoids; displayed after EMA smoothing.

Acceptance checks (manual)
--------------------------
- Standing full-body → "standing" (tri mode) with conf > 0.6.
- Wheelchair/normal sitting (knees or hip flexion evident OR leg-drop small) → "sitting" > 0.6.
- Lying on bed → "sleeping" > 0.6.
- No person → "no-person".

Dependencies
------------
- ultralytics, opencv-python, numpy

Notes
-----
- Ultralytics auto-downloads weights on first use; afterwards you can run offline.
- This script logs a one-line STATUS every N seconds (default 1s): “posture=… conf=…”.

License
-------
MIT
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None  # type: ignore
    _ultra_import_err = e
else:
    _ultra_import_err = None

try:
    import cv2
except Exception as e:
    cv2 = None  # type: ignore
    _cv_import_err = e
else:
    _cv_import_err = None


# ---------- math helpers ----------

def _angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    """Return angle in degrees between vectors v1 and v2."""
    a = v1.astype(np.float32)
    b = v2.astype(np.float32)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return float("nan")
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))

def _sigmoid(x: float) -> float:
    try:
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# COCO-17 indices:
# 0:nose,1:leye,2:reye,3:lear,4:rear,5:lsho,6:rsho,7:lelb,8:relb,9:lwri,10:rwri,
# 11:lhip,12:rhip,13:lknee,14:rknee,15:lank,16:rank
SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

@dataclass
class Thresholds:
    # Standing (used only in tri mode)
    standing_torso_deg: float = 25.0
    standing_knee_deg_min: float = 160.0

    # Sitting
    sitting_knee_deg_min: float = 80.0
    sitting_hip_deg_min: float = 65.0
    sitting_legdrop_max: float = 0.22  # small leg-drop → legs forward/elevated → sitting/wheelchair
    sitting_torso_deg_max: float = 35.0

    # Sleeping
    sleeping_torso_deg_min: float = 60.0
    sleeping_vspread_max: float = 0.30  # stricter than before

    # Couch override
    couch_knee_deg_min: float = 90.0

    # Soft mapping scale
    angle_soft_scale: float = 10.0


@dataclass
class Features:
    torso_deg_from_vertical: float = float("nan")
    left_knee_deg: float = float("nan")
    right_knee_deg: float = float("nan")
    left_hip_deg: float = float("nan")
    right_hip_deg: float = float("nan")
    vertical_spread_norm: float = float("nan")
    leg_drop_norm: float = float("nan")   # mean((ankle_y - hip_y)/bbox_h)
    bbox_aspect: float = float("nan")
    n_keypoints: int = 0


@dataclass
class PostureResult:
    label: str
    confidence: float
    features: Features
    raw_scores: Dict[str, float]


class PoseDetector:
    def __init__(self, model_name: Optional[str], conf: float, imgsz: int, device: str):
        if YOLO is None:
            raise RuntimeError(
                f"Failed to import ultralytics.YOLO: {_ultra_import_err}\n"
                "Install via: pip install ultralytics"
            )
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: Optional[str]):
        tried: List[str] = []
        if model_name:
            tried.append(model_name)
        else:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                tried.extend(["yolo11m-pose.pt", "yolo11n-pose.pt", "yolo11s-pose.pt"])
            else:
                tried.extend(["yolov8n-pose.pt", "yolo11n-pose.pt"])
        last_err = None
        for name in tried:
            try:
                m = YOLO(name)
                logging.info(f"Loaded model: {name}")
                return m
            except Exception as e:
                logging.warning(f"Could not load model '{name}': {e}")
                last_err = e
        raise RuntimeError(f"Failed to load any pose model from {tried}: {last_err}")

    def infer(self, frame_bgr: np.ndarray):
        return self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )


class PostureClassifier:
    def __init__(self, th: Thresholds, kp_conf_min: float, min_keypoints: int,
                 mode: str = "binary", couch_override: bool = True):
        self.th = th
        self.kp_conf_min = kp_conf_min
        self.min_keypoints = min_keypoints
        self.mode = mode
        self.couch_override = couch_override

    @staticmethod
    def _center(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return (p1 + p2) / 2.0

    def _valid_kps(self, kps_xy: np.ndarray, kps_conf: Optional[np.ndarray]) -> np.ndarray:
        valid = np.isfinite(kps_xy).all(axis=1)
        if kps_conf is not None and kps_conf.shape[0] == 17:
            valid &= (kps_conf >= self.kp_conf_min)
        return valid

    def _compute_angles(self, pts: Dict[int, np.ndarray]) -> Tuple[float, float, float, float, float]:
        torso_deg = float("nan"); l_knee = float("nan"); r_knee = float("nan")
        l_hip = float("nan"); r_hip = float("nan")

        if 5 in pts and 6 in pts and 11 in pts and 12 in pts:
            sh = self._center(pts[5], pts[6])
            hipc = self._center(pts[11], pts[12])
            tvec = hipc - sh
            torso_deg = _angle_between(tvec, np.array([0.0, 1.0], dtype=np.float32))  # vs vertical

        if 11 in pts and 13 in pts and 15 in pts:
            l_thigh = pts[13] - pts[11]; l_shank = pts[15] - pts[13]
            l_knee = _angle_between(l_thigh, l_shank)
        if 12 in pts and 14 in pts and 16 in pts:
            r_thigh = pts[14] - pts[12]; r_shank = pts[16] - pts[14]
            r_knee = _angle_between(r_thigh, r_shank)

        if 5 in pts and 6 in pts and 11 in pts and 13 in pts:
            sh = self._center(pts[5], pts[6]); torso_l = pts[11] - sh; l_thigh = pts[13] - pts[11]
            l_hip = _angle_between(torso_l, l_thigh)
        if 5 in pts and 6 in pts and 12 in pts and 14 in pts:
            sh = self._center(pts[5], pts[6]); torso_r = pts[12] - sh; r_thigh = pts[14] - pts[12]
            r_hip = _angle_between(torso_r, r_thigh)

        return torso_deg, l_knee, r_knee, l_hip, r_hip

    @staticmethod
    def _nanmean(*vals: float) -> float:
        arr = np.array(vals, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        return float(np.mean(arr)) if arr.size else float("nan")

    @staticmethod
    def _bbox_aspect(xyxy: np.ndarray) -> float:
        x1, y1, x2, y2 = xyxy
        w = max(1.0, float(x2 - x1)); h = max(1.0, float(y2 - y1))
        return w / h

    @staticmethod
    def _vertical_spread(pts: Dict[int, np.ndarray], bbox_h: float) -> float:
        ys = [p[1] for p in pts.values() if np.isfinite(p).all()]
        if not ys or bbox_h <= 1e-6:
            return float("nan")
        return (max(ys) - min(ys)) / bbox_h

    @staticmethod
    def _leg_drop(pts: Dict[int, np.ndarray], bbox_h: float) -> float:
        """Mean((ankle_y - hip_y)/bbox_h) for available sides; small if legs are forward/elevated."""
        if bbox_h <= 1e-6:
            return float("nan")
        drops = []
        for hip_i, ank_i in [(11, 15), (12, 16)]:
            if hip_i in pts and ank_i in pts:
                drops.append((pts[ank_i][1] - pts[hip_i][1]) / bbox_h)
        if not drops:
            return float("nan")
        return float(np.mean(drops))

    def classify(self, kps_xy: np.ndarray, kps_conf: Optional[np.ndarray], bbox_xyxy: np.ndarray) -> PostureResult:
        if kps_xy is None or kps_xy.shape != (17, 2):
            return PostureResult("no-person", 0.0, Features(), {"sitting":0.0,"sleeping":0.0,"standing":0.0})

        valid = self._valid_kps(kps_xy, kps_conf)
        pts: Dict[int, np.ndarray] = {i: kps_xy[i] for i in range(17) if valid[i]}
        n_valid = int(valid.sum())

        torso_deg, l_knee, r_knee, l_hip, r_hip = self._compute_angles(pts)
        knee_mean = self._nanmean(l_knee, r_knee)
        hip_mean = self._nanmean(l_hip, r_hip)
        bbox_h = float(bbox_xyxy[3] - bbox_xyxy[1])
        vspread = self._vertical_spread(pts, bbox_h)
        aspect = self._bbox_aspect(bbox_xyxy)
        leg_drop = self._leg_drop(pts, bbox_h)

        feats = Features(
            torso_deg_from_vertical=torso_deg,
            left_knee_deg=l_knee, right_knee_deg=r_knee,
            left_hip_deg=l_hip, right_hip_deg=r_hip,
            vertical_spread_norm=vspread, leg_drop_norm=leg_drop,
            bbox_aspect=aspect, n_keypoints=n_valid
        )

        th = self.th; s = th.angle_soft_scale
        def SS(x: float) -> float:  # safe sigmoid
            return 0.0 if not np.isfinite(x) else _sigmoid(x)

        # --- Sleeping rule ---
        sleep_torso = SS((torso_deg - th.sleeping_torso_deg_min) / s)
        sleep_vsprd = SS((th.sleeping_vspread_max - vspread) / max(1e-6, th.sleeping_vspread_max*0.5))
        sleeping_score = math.sqrt(sleep_torso * sleep_vsprd)
        if not np.isfinite(vspread):  # degrade if spread missing
            sleeping_score *= 0.6

        # --- Sitting rule (wheelchair-friendly) ---
        sit_torso = SS((th.sitting_torso_deg_max - torso_deg) / s)
        sit_knee  = SS((knee_mean - th.sitting_knee_deg_min) / s) if np.isfinite(knee_mean) else 0.5
        sit_hip   = SS((hip_mean  - th.sitting_hip_deg_min)  / s) if np.isfinite(hip_mean)  else 0.5
        # leg-drop small = good for sitting
        if np.isfinite(leg_drop):
            sit_legdrop = SS((th.sitting_legdrop_max - leg_drop) / max(0.05, th.sitting_legdrop_max*0.4))
        else:
            sit_legdrop = 0.5
        # Combine: torso near-vertical and any of knee OR hip OR leg-drop indicates sitting
        legs_support = 1.0 - (1.0 - sit_knee) * (1.0 - sit_hip) * (1.0 - sit_legdrop)  # OR-like soft-union
        sitting_score = math.sqrt(sit_torso * legs_support)

        # --- Standing (only in tri mode) ---
        standing_score = 0.0
        if self.mode == "tri":
            stand_torso = SS((th.standing_torso_deg - torso_deg) / s)
            stand_knee  = SS((knee_mean - th.standing_knee_deg_min) / s) if np.isfinite(knee_mean) else 0.4
            standing_score = math.sqrt(stand_torso * stand_knee)

        # --- Couch override: horizontal torso BUT bent knees or small leg-drop → sitting ---
        if self.couch_override:
            couch_cond = (np.isfinite(torso_deg) and torso_deg >= th.sleeping_torso_deg_min)
            couch_support = max(
                SS((knee_mean - th.couch_knee_deg_min) / s) if np.isfinite(knee_mean) else 0.0,
                SS((th.sitting_legdrop_max - leg_drop) / max(0.05, th.sitting_legdrop_max*0.4)) if np.isfinite(leg_drop) else 0.0
            )
            if couch_cond and couch_support > 0.55:
                # Boost sitting, damp sleeping a bit
                sitting_score = max(sitting_score, 0.75 * couch_support + 0.25 * sit_torso)
                sleeping_score *= 0.8

        scores = {"sitting": float(sitting_score), "sleeping": float(sleeping_score), "standing": float(standing_score)}
        label_pool = ["sitting", "sleeping"] if self.mode == "binary" else ["sitting", "sleeping", "standing"]
        label = max(label_pool, key=lambda k: scores[k])
        conf = float(scores[label])

        if n_valid < self.min_keypoints and conf < 0.6:
            return PostureResult("uncertain", conf, feats, scores)
        if conf < 0.35:
            return PostureResult("uncertain", conf, feats, scores)
        return PostureResult(label, conf, feats, scores)


class Visualizer:
    def __init__(self, window_name: str = "PostureCam", show_window: bool = True):
        if cv2 is None:
            raise RuntimeError(f"Failed to import cv2: {_cv_import_err}")
        self.window_name = window_name
        self.show_window = show_window
        if show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw(self, frame_bgr: np.ndarray,
             bbox_xyxy: Optional[np.ndarray],
             kps_xy: Optional[np.ndarray],
             kps_conf: Optional[np.ndarray],
             posture: PostureResult, fps: float) -> np.ndarray:
        out = frame_bgr
        h, w = out.shape[:2]

        if bbox_xyxy is not None:
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if kps_xy is not None and kps_xy.shape == (17, 2):
            for i, (x, y) in enumerate(kps_xy.astype(int)):
                if kps_conf is None or kps_conf[i] > 0.3:
                    cv2.circle(out, (int(x), int(y)), 3, (255, 0, 0), -1)
            for a, b in SKELETON:
                if (kps_conf is None or (kps_conf[a] > 0.3 and kps_conf[b] > 0.3)):
                    pa = tuple(kps_xy[a].astype(int))
                    pb = tuple(kps_xy[b].astype(int))
                    cv2.line(out, pa, pb, (0, 200, 200), 2)

        label = f"Posture: {posture.label} ({posture.confidence:.2f})"
        fps_text = f"FPS: {fps:.1f}"
        y0 = 24
        cv2.putText(out, label, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 240), 2, cv2.LINE_AA)
        cv2.putText(out, fps_text, (10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 200, 20), 2, cv2.LINE_AA)

        legend = "q: quit   s: snapshot"
        tw, th = cv2.getTextSize(legend, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(out, (10, h - th - 16), (10 + tw + 10, h - 6), (0, 0, 0), -1)
        cv2.putText(out, legend, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return out

    def show(self, frame_bgr: np.ndarray) -> int:
        if not self.show_window:
            return -1
        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self.show_window:
            cv2.destroyWindow(self.window_name)


# ---------- runtime helpers ----------

def pick_device() -> str:
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"

def select_person(boxes_xyxy: np.ndarray, boxes_conf: np.ndarray, policy: str = "largest") -> int:
    if policy == "highest-conf":
        return int(np.argmax(boxes_conf))
    wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
    areas = wh[:, 0] * wh[:, 1]
    return int(np.argmax(areas))

def rotate_frame(frame: np.ndarray, rot: int) -> np.ndarray:
    if rot == 0:
        return frame
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def run(args: argparse.Namespace) -> None:
    # Logging
    lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setLevel(lvl)
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logging.getLogger().addHandler(fh)

    logging.info("=== PostureCam startup ===")
    for k, v in vars(args).items():
        logging.info(f"{k}: {v}")

    device = pick_device()
    logging.info(f"Selected device: {device}")

    try:
        detector = PoseDetector(model_name=args.model, conf=args.conf, imgsz=args.imgsz, device=device)
    except Exception as e:
        logging.error(f"Model init failed: {e}")
        return

    th = Thresholds(
        standing_torso_deg=args.standing_torso_deg,
        standing_knee_deg_min=args.standing_knee_deg_min,
        sitting_knee_deg_min=args.sitting_knee_deg_min,
        sitting_hip_deg_min=args.sitting_hip_deg_min,
        sitting_legdrop_max=args.sitting_legdrop_max,
        sitting_torso_deg_max=args.sitting_torso_deg_max,
        sleeping_torso_deg_min=args.sleeping_torso_deg_min,
        sleeping_vspread_max=args.sleeping_vspread_max,
        couch_knee_deg_min=args.couch_knee_deg_min,
        angle_soft_scale=args.angle_soft_scale,
    )
    classifier = PostureClassifier(
        th, kp_conf_min=args.kp_conf, min_keypoints=args.min_keypoints,
        mode=args.mode, couch_override=not args.no_couch_override
    )

    # Video
    if cv2 is None:
        logging.error(f"OpenCV import failed: {_cv_import_err}")
        return

    src = args.src if args.src is not None else int(args.camera)
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logging.error(f"Failed to open video source: {src}")
        return

    vis = Visualizer(show_window=bool(args.display))
    os.makedirs("snapshots", exist_ok=True)

    # CSV
    csv_writer = None; csv_file = None
    if args.csv:
        csv_file = open(args.csv, "a", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if os.path.getsize(args.csv) == 0:
            csv_writer.writerow([
                "timestamp", "posture", "confidence", "fps",
                "torso_deg", "left_knee_deg", "right_knee_deg",
                "left_hip_deg", "right_hip_deg",
                "vertical_spread_norm", "leg_drop_norm", "bbox_aspect", "n_keypoints"
            ])

    # Temporal smoothing (EMA) over class scores
    ema_scores: Dict[str, float] = {"sitting": 0.0, "sleeping": 0.0, "standing": 0.0}
    alpha = float(args.smooth_alpha)
    status_last = time.time()  # per-second status ticker

    last_log = time.time()
    t0 = time.time()
    fps_smooth = None
    ema_alpha_fps = 0.15

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                logging.warning("Frame grab failed. Reconnecting...")
                time.sleep(0.5)
                cap.release()
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    logging.error("Re-open failed. Exiting.")
                    break
                continue

            # Optional rotate before inference
            if args.camera_rot in (90, 180, 270):
                frame = rotate_frame(frame, args.camera_rot)

            # Inference
            try:
                results = detector.infer(frame)
            except Exception as e:
                logging.error(f"Inference error: {e}")
                break
            res = results[0]

            # Parse detections
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                posture = PostureResult("no-person", 0.0, Features(), {"sitting":0.0,"sleeping":0.0,"standing":0.0})
                bbox_sel = None; kps_xy_sel = None; kps_conf_sel = None
            else:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                boxes_conf = res.boxes.conf.cpu().numpy()
                kps_xy = res.keypoints.xy.cpu().numpy()       # (N,17,2)
                kps_conf = None
                if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
                    kps_conf = res.keypoints.conf.cpu().numpy()  # (N,17)

                idx = select_person(boxes_xyxy, boxes_conf, policy=args.select)
                bbox_sel = boxes_xyxy[idx]
                kps_xy_sel = kps_xy[idx]
                kps_conf_sel = kps_conf[idx] if kps_conf is not None else None
                posture = classifier.classify(kps_xy_sel, kps_conf_sel, bbox_sel)

            # FPS calc
            t1 = time.time()
            inst_fps = 1.0 / max(1e-6, t1 - t0); t0 = t1
            fps_smooth = inst_fps if fps_smooth is None else (1-ema_alpha_fps)*fps_smooth + ema_alpha_fps*inst_fps

            # EMA smoothing over class scores
            for k in ema_scores.keys():
                score = posture.raw_scores.get(k, 0.0)
                ema_scores[k] = (1 - alpha) * ema_scores[k] + alpha * score
            # choose label from EMA
            label_pool = ["sitting", "sleeping"] if args.mode == "binary" else ["sitting", "sleeping", "standing"]
            sm_label = max(label_pool, key=lambda k: ema_scores[k])
            sm_conf = float(ema_scores[sm_label])

            # Periodic logs
            if time.time() - last_log > 5.0:
                logging.info(f"Running FPS: {fps_smooth:.1f}")
                last_log = time.time()
            logging.debug(
                f"raw Posture={posture.label}({posture.confidence:.2f}) smoothed={sm_label}({sm_conf:.2f}) "
                f"torso={posture.features.torso_deg_from_vertical:.1f}° "
                f"knee(L/R)={posture.features.left_knee_deg:.1f}/{posture.features.right_knee_deg:.1f}° "
                f"hip(L/R)={posture.features.left_hip_deg:.1f}/{posture.features.right_hip_deg:.1f}° "
                f"vspread={posture.features.vertical_spread_norm:.2f} legdrop={posture.features.leg_drop_norm:.2f}"
            )

            # Per-second STATUS line (what you asked)
            now = time.time()
            if now - status_last >= float(args.status_interval):
                logging.info(f"STATUS: posture={sm_label} conf={sm_conf:.2f}")
                status_last = now

            # CSV
            if csv_writer is not None:
                csv_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    sm_label, f"{sm_conf:.3f}", f"{fps_smooth:.2f}",
                    f"{posture.features.torso_deg_from_vertical:.2f}",
                    f"{posture.features.left_knee_deg:.2f}",
                    f"{posture.features.right_knee_deg:.2f}",
                    f"{posture.features.left_hip_deg:.2f}",
                    f"{posture.features.right_hip_deg:.2f}",
                    f"{posture.features.vertical_spread_norm:.3f}",
                    f"{posture.features.leg_drop_norm:.3f}",
                    f"{posture.features.bbox_aspect:.3f}",
                    posture.features.n_keypoints
                ])
                csv_file.flush()

            # Draw & UI
            if args.display:
                # Show smoothed label/conf on frame for stability
                smooth_posture = PostureResult(sm_label, sm_conf, posture.features, posture.raw_scores)
                canvas = vis.draw(frame, bbox_sel, kps_xy_sel, kps_conf_sel, smooth_posture, fps=fps_smooth or inst_fps)
                key = vis.show(canvas)
                if key == ord("q"):
                    logging.info("Quit requested (q).")
                    break
                elif key == ord("s"):
                    fname = time.strftime("snapshots/snap_%Y%m%d_%H%M%S.png")
                    cv2.imwrite(fname, canvas)
                    logging.info(f"Saved snapshot: {fname}")
            else:
                time.sleep(0.001)

    finally:
        try: cap.release()
        except Exception: pass
        try: vis.close()
        except Exception: pass
        if csv_writer is not None: csv_file.close()
        logging.info("Stopped.")


# ---------- CLI ----------

REQUIREMENTS_TXT = r"""
# Minimal requirements for posture_cam.py
ultralytics>=8.1.0
opencv-python>=4.7.0
numpy>=1.23.0
"""

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time posture classification (sitting vs sleeping; tri-mode optional).")
    # IO / model
    p.add_argument("--camera", type=int, default=0, help="Camera index (ignored if --src is set)")
    p.add_argument("--src", type=str, default=None, help="Source path/URL; overrides --camera")
    p.add_argument("--model", type=str, default=None, help="Pose model name/path (auto-tries sensible defaults).")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--display", dest="display", action="store_true", help="Show window (default).")
    p.add_argument("--headless", dest="display", action="store_false", help="Disable window.")
    p.set_defaults(display=True)
    p.add_argument("--camera-rot", type=int, default=0, choices=[0,90,180,270], help="Rotate camera image by fixed angle.")
    p.add_argument("--max-persons", type=int, default=1, help="(Reserved) detections considered, we select 1.")
    p.add_argument("--select", type=str, default="largest", choices=["largest","highest-conf"], help="Selection policy.")

    # Modes
    p.add_argument("--mode", type=str, default="binary", choices=["binary","tri"], help="binary: sitting vs sleeping; tri: +standing")

    # Thresholds / classifier
    p.add_argument("--kp-conf", type=float, default=0.20, help="Min keypoint confidence.")
    p.add_argument("--min-keypoints", type=int, default=6, help="Min valid keypoints before confident classification.")
    p.add_argument("--standing-torso-deg", type=float, default=25.0, help="Max torso-from-vertical for standing.")
    p.add_argument("--standing-knee-deg-min", type=float, default=160.0, help="Min knee angle for standing.")
    p.add_argument("--sitting-knee-deg-min", type=float, default=80.0, help="Min knee angle for sitting.")
    p.add_argument("--sitting-hip-deg-min", type=float, default=65.0, help="Min hip flex for sitting.")
    p.add_argument("--sitting-legdrop-max", type=float, default=0.22, help="Max leg-drop (ankle_y-hip_y)/bbox_h for sitting.")
    p.add_argument("--sitting-torso-deg-max", type=float, default=25.0, help="Max torso-from-vertical for sitting.")
    p.add_argument("--sleeping-torso-deg-min", type=float, default=80.0, help="Min torso-from-vertical for sleeping.")
    p.add_argument("--sleeping-vspread-max", type=float, default=0.25, help="Max vertical spread for sleeping.")
    p.add_argument("--couch-knee-deg-min", type=float, default=90.0, help="If torso≈horizontal but knee≥this: treat as sitting (couch).")
    p.add_argument("--angle-soft-scale", type=float, default=10.0, help="Degrees scale for soft scoring.")
    p.add_argument("--couch-override", dest="no_couch_override", action="store_false", help="(default) enable couch override.")
    p.add_argument("--no-couch-override", dest="no_couch_override", action="store_true", help="Disable couch override.")
    p.set_defaults(no_couch_override=False)

    # Temporal smoothing + status
    p.add_argument("--smooth-alpha", type=float, default=0.20, help="EMA on class scores (0..1).")
    p.add_argument("--status-interval", type=float, default=1.0, help="Seconds between STATUS logs.")

    # Logging / CSV
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output file.")
    p.add_argument("--log-level", type=str, default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    p.add_argument("--log-file", type=str, default=None, help="Optional log file path.")

    # Misc
    p.add_argument("--write-reqs", action="store_true", help="Write requirements.txt and exit.")
    args = p.parse_args(argv)

    if args.write_reqs:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(REQUIREMENTS_TXT.strip() + "\n")
        print("Wrote requirements.txt")
        sys.exit(0)
    return args


def main():
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
