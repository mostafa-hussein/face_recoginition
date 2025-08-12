#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
posture_cam.py — Enhanced Real-time posture classification (sleeping / sitting / standing) 
from webcam using Ultralytics YOLO Pose with improved wheelchair and couch detection.

IMPROVEMENTS IN THIS VERSION:
- Enhanced sitting detection for wheelchair users and couch scenarios  
- Better model fallback (YOLO11 -> YOLO8 -> older versions)
- Wheelchair-specific geometric features and thresholds
- Improved "laying on couch" classification as sitting variant
- Per-second status logging with detailed metrics
- More robust confidence scoring with multiple validation methods
- Enhanced sleeping detection with body orientation analysis
- Better handling of partial occlusion scenarios

Usage
------
Install deps (first time):
    $ python posture_cam.py --write-reqs
    $ pip install -r requirements.txt

Run with default webcam (index 0) and auto GPU/CPU:
    $ python posture_cam.py

Enhanced options for wheelchair/couch scenarios:
    --wheelchair-mode          # Optimized thresholds for wheelchair users
    --couch-sitting-enabled    # Treat laying on couch as sitting
    --sitting-height-ratio 0.6 # Min height ratio for distinguishing sitting vs sleeping
    --per-second-log          # Print status every second (default: enabled)
    
Common options:
    --camera 0                 # camera index (default 0). Ignored if --src is provided
    --src path_or_url          # file/rtsp/http source; overrides --camera
    --model MODEL              # Auto-tries yolo11x-pose -> yolo11n-pose -> yolov8n-pose -> yolov5n
    --conf 0.20                # Lower confidence for better detection (was 0.25)
    --imgsz 640                # inference image size
    --display / --headless     # show window or run headless (default: display)
    --max-persons 1            # select only one person for classification
    --select largest           # selection policy: largest | highest-conf
    --kp-conf 0.25             # min keypoint confidence (lowered from 0.30)
    --min-keypoints 6          # minimum valid keypoints (lowered from 8)
    --enhanced-sitting-deg 45  # max torso angle for enhanced sitting detection
    --wheelchair-torso-deg 50  # max torso angle for wheelchair sitting
    --sleeping-torso-deg-min 55# torso angle from vertical ≥ this → sleeping (lowered from 60)
    --sleeping-vspread-max 0.4 # normalized vertical spread ≤ this → sleeping (increased from 0.35)
    --angle-soft-scale 12      # degrees scale for soft confidence mapping (increased sensitivity)
    --csv out.csv              # optional CSV log with timestamp, posture, confidence, fps, features
    --log-level INFO           # DEBUG | INFO | WARNING | ERROR
    --log-file posture.log     # optional log file

Controls in the viewer:
    q : quit
    s : save snapshot with overlays to ./snapshots/

What it does (ENHANCED)
-----------------------
- Opens a video source and runs Ultralytics YOLO Pose with improved model selection
- Enhanced geometric analysis for wheelchair and couch scenarios:
    * Torso inclination with wheelchair-specific thresholds
    * Body height ratio analysis (sitting vs sleeping distinction)
    * Hip-to-shoulder distance for posture validation
    * Enhanced vertical spread calculation with outlier removal
    * Bbox aspect ratio analysis for lying detection
- Improved classification with multiple validation paths:
    * Standard sitting: torso near vertical, medium height ratio
    * Wheelchair sitting: relaxed torso angle, focus on body positioning  
    * Couch sitting: horizontal positioning but elevated from ground
    * Sleeping: horizontal torso, low vertical spread, wide bbox
- Per-second status logging with confidence breakdowns
- Confidence scoring uses geometric mean of multiple factors for robustness

Performance & Model notes
-------------------------
- Auto-tries latest YOLO models: yolo11x-pose (most accurate) -> yolo11n-pose -> yolov8n-pose -> yolov5n6
- Targets ≥15 FPS on CPU for yolo11n-pose, ≥25 FPS for yolo8n, higher on GPU
- Enhanced error handling and model fallback for maximum compatibility

Dependencies
------------
- ultralytics>=8.1.0, opencv-python>=4.7.0, numpy>=1.23.0
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
    torch = None

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    _ultra_import_err = e
else:
    _ultra_import_err = None

try:
    import cv2
except Exception as e:
    cv2 = None
    _cv_import_err = e
else:
    _cv_import_err = None


# Enhanced utility math for keypoints
def _angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    """Return angle in degrees between vectors v1 and v2."""
    a = v1.astype(np.float32)
    b = v2.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return float("nan")
    cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(math.degrees(math.acos(cosang)))


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(p1 - p2))


def _sigmoid(x: float, scale: float = 1.0) -> float:
    """Sigmoid function with adjustable scale."""
    try:
        scaled_x = x / scale
        if scaled_x >= 0:
            z = math.exp(-scaled_x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(scaled_x)
            return z / (1.0 + z)
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# COCO-17 indices (Ultralytics order):
COCO_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms/shoulders
    (11, 12), (5, 11), (6, 12),               # torso connections
    (11, 13), (13, 15), (12, 14), (14, 16)    # legs
]


@dataclass
class EnhancedThresholds:
    """Enhanced thresholds for better wheelchair and couch detection."""
    # Standard thresholds
    standing_torso_deg: float = 25.0
    standing_knee_deg_min: float = 160.0
    
    # Enhanced sitting thresholds
    sitting_knee_deg_min: float = 70.0  # Lowered for wheelchair flexibility
    sitting_torso_deg_max: float = 35.0
    enhanced_sitting_deg: float = 45.0   # More permissive sitting angle
    wheelchair_torso_deg: float = 50.0   # Wheelchair-specific threshold
    sitting_height_ratio: float = 0.6    # Min height ratio for sitting vs sleeping
    
    # Enhanced sleeping thresholds  
    sleeping_torso_deg_min: float = 55.0 # Lowered from 60 for better detection
    sleeping_vspread_max: float = 0.4    # Increased from 0.35
    
    # Confidence and validation
    angle_soft_scale: float = 12.0       # Increased sensitivity
    confidence_threshold: float = 0.3    # Minimum confidence for valid classification
    
    # Wheelchair and couch specific
    wheelchair_mode: bool = False
    couch_sitting_enabled: bool = True


@dataclass  
class EnhancedFeatures:
    """Enhanced features for robust posture classification."""
    # Basic geometric features
    torso_deg_from_vertical: float = float("nan")
    left_knee_deg: float = float("nan")
    right_knee_deg: float = float("nan")
    left_hip_deg: float = float("nan")
    right_hip_deg: float = float("nan")
    
    # Enhanced features
    vertical_spread_norm: float = float("nan")
    height_ratio: float = float("nan")  # Body height / bbox height
    bbox_aspect: float = float("nan")   # w/h
    shoulder_hip_distance: float = float("nan")  # Normalized torso length
    center_of_mass_y: float = float("nan")       # Vertical position of body COM
    body_angle_confidence: float = 0.0           # Confidence in angle measurements
    
    # Validation metrics
    n_keypoints: int = 0
    torso_visible: bool = False
    legs_visible: bool = False
    upper_body_complete: bool = False


@dataclass
class PostureResult:
    label: str
    confidence: float
    features: EnhancedFeatures
    confidence_breakdown: Dict[str, float] = None  # For debugging


class EnhancedPoseDetector:
    """Enhanced pose detector with better model fallback."""
    
    def __init__(self, model_name: Optional[str], conf: float, imgsz: int, device: str):
        if YOLO is None:
            raise RuntimeError(
                f"Failed to import ultralytics.YOLO: {_ultra_import_err}\n"
                "Install dependencies via: pip install ultralytics"
            )
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.model = self._load_model_with_fallback(model_name)

    def _load_model_with_fallback(self, model_name: Optional[str]):
        """Enhanced model loading with comprehensive fallback."""
        model_priority = []
        
        if model_name:
            model_priority.append(model_name)
        else:
            # Best to worst model priority for pose detection
            model_priority.extend([
                "yolo11x-pose.pt",    # Most accurate but slower
                "yolo11l-pose.pt",    # Good balance
                "yolo11m-pose.pt",    # Medium accuracy/speed
                "yolo11s-pose.pt",    # Faster but less accurate
                "yolo11n-pose.pt",    # Fastest
                "yolov8x-pose.pt",    # Fallback to v8 series
                "yolov8l-pose.pt",
                "yolov8m-pose.pt", 
                "yolov8s-pose.pt",
                "yolov8n-pose.pt",
                "yolov5n6.pt",        # Last resort
            ])

        last_err = None
        for model in model_priority:
            try:
                m = YOLO(model)
                logging.info(f"✓ Successfully loaded model: {model}")
                return m
            except Exception as e:
                logging.debug(f"Failed to load {model}: {e}")
                last_err = e
                continue
                
        raise RuntimeError(f"Failed to load any pose model from {model_priority}: {last_err}")

    def infer(self, frame_bgr: np.ndarray):
        """Run pose inference with enhanced error handling."""
        try:
            results = self.model.predict(
                frame_bgr, 
                imgsz=self.imgsz, 
                conf=self.conf, 
                device=self.device, 
                verbose=False,
                half=False  # Disable half precision for stability
            )
            return results
        except Exception as e:
            logging.error(f"Inference failed: {e}")
            return []


class EnhancedPostureClassifier:
    """Enhanced posture classifier optimized for wheelchair and couch scenarios."""

    def __init__(self, th: EnhancedThresholds, kp_conf_min: float, min_keypoints: int):
        self.th = th
        self.kp_conf_min = kp_conf_min  
        self.min_keypoints = min_keypoints
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _center(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        return (p1 + p2) / 2.0

    def _valid_kps(self, kps_xy: np.ndarray, kps_conf: Optional[np.ndarray]) -> np.ndarray:
        """Return boolean mask of valid keypoints with enhanced filtering."""
        if kps_xy is None or len(kps_xy) != 17:
            return np.zeros(17, dtype=bool)
            
        # Check for finite coordinates
        valid = np.isfinite(kps_xy).all(axis=1)
        
        # Apply confidence filter
        if kps_conf is not None and kps_conf.shape[0] == 17:
            valid &= (kps_conf >= self.kp_conf_min)
            
        return valid

    def _compute_enhanced_angles(self, pts: Dict[int, np.ndarray]) -> Tuple[float, float, float, float, float, bool]:
        """Compute angles with enhanced validation and confidence scoring."""
        torso_deg = float("nan")
        left_knee = float("nan")
        right_knee = float("nan") 
        left_hip = float("nan")
        right_hip = float("nan")
        angle_confidence = False

        # Enhanced torso angle calculation
        shoulder_center = None
        hip_center = None
        
        # More robust center calculation
        if 5 in pts and 6 in pts:
            shoulder_center = self._center(pts[5], pts[6])
            angle_confidence = True
        elif 5 in pts:
            shoulder_center = pts[5]
        elif 6 in pts:
            shoulder_center = pts[6]
            
        if 11 in pts and 12 in pts:
            hip_center = self._center(pts[11], pts[12])
            angle_confidence = angle_confidence and True
        elif 11 in pts:
            hip_center = pts[11]  
        elif 12 in pts:
            hip_center = pts[12]
            
        if shoulder_center is not None and hip_center is not None:
            torso_vec = hip_center - shoulder_center
            # Image Y points down, so positive Y is down
            vertical_vec = np.array([0.0, 1.0])
            torso_deg = _angle_between(torso_vec, vertical_vec)
            
        # Enhanced knee angle calculation with validation
        if 11 in pts and 13 in pts and 15 in pts:
            thigh = pts[13] - pts[11]
            shank = pts[15] - pts[13]
            if np.linalg.norm(thigh) > 10 and np.linalg.norm(shank) > 10:  # Minimum segment length
                left_knee = _angle_between(thigh, shank)
                
        if 12 in pts and 14 in pts and 16 in pts:
            thigh = pts[14] - pts[12]
            shank = pts[16] - pts[14] 
            if np.linalg.norm(thigh) > 10 and np.linalg.norm(shank) > 10:
                right_knee = _angle_between(thigh, shank)
                
        # Enhanced hip flexion calculation
        if shoulder_center is not None and 11 in pts and 13 in pts:
            torso_vec = pts[11] - shoulder_center
            thigh_vec = pts[13] - pts[11]
            if np.linalg.norm(torso_vec) > 10 and np.linalg.norm(thigh_vec) > 10:
                left_hip = _angle_between(torso_vec, thigh_vec)
                
        if shoulder_center is not None and 12 in pts and 14 in pts:
            torso_vec = pts[12] - shoulder_center  
            thigh_vec = pts[14] - pts[12]
            if np.linalg.norm(torso_vec) > 10 and np.linalg.norm(thigh_vec) > 10:
                right_hip = _angle_between(torso_vec, thigh_vec)

        return torso_deg, left_knee, right_knee, left_hip, right_hip, angle_confidence

    def _compute_enhanced_features(self, pts: Dict[int, np.ndarray], bbox_xyxy: np.ndarray) -> EnhancedFeatures:
        """Compute enhanced geometric features."""
        bbox_w = bbox_xyxy[2] - bbox_xyxy[0] 
        bbox_h = bbox_xyxy[3] - bbox_xyxy[1]
        
        # Basic angle features
        torso_deg, l_knee, r_knee, l_hip, r_hip, angle_conf = self._compute_enhanced_angles(pts)
        
        # Enhanced vertical spread with outlier removal
        ys = [p[1] for p in pts.values()]
        if len(ys) >= 3:
            ys = np.array(ys)
            q25, q75 = np.percentile(ys, [25, 75])
            iqr = q75 - q25
            # Remove outliers beyond 1.5*IQR
            mask = (ys >= q25 - 1.5*iqr) & (ys <= q75 + 1.5*iqr)
            if mask.sum() > 0:
                ys_clean = ys[mask] 
                vspread = (ys_clean.max() - ys_clean.min()) / max(bbox_h, 1.0)
            else:
                vspread = (ys.max() - ys.min()) / max(bbox_h, 1.0)
        else:
            vspread = float("nan")
            
        # Height ratio: actual body height vs bbox height  
        if len(ys) >= 2:
            body_height = max(ys) - min(ys)
            height_ratio = body_height / max(bbox_h, 1.0)
        else:
            height_ratio = float("nan")
            
        # Bbox aspect ratio
        bbox_aspect = bbox_w / max(bbox_h, 1.0)
        
        # Shoulder-hip distance (normalized torso length)
        shoulder_hip_dist = float("nan")
        if 5 in pts and 6 in pts and 11 in pts and 12 in pts:
            shoulder_center = self._center(pts[5], pts[6])
            hip_center = self._center(pts[11], pts[12])
            shoulder_hip_dist = _distance(shoulder_center, hip_center) / max(bbox_h, 1.0)
            
        # Center of mass calculation (Y coordinate)  
        com_y = float("nan")
        if pts:
            points_array = np.array(list(pts.values()))
            com_y = np.mean(points_array[:, 1]) / max(bbox_h, 1.0)
            
        # Visibility flags
        torso_visible = (5 in pts or 6 in pts) and (11 in pts or 12 in pts)
        legs_visible = (13 in pts or 14 in pts) and (15 in pts or 16 in pts)
        upper_body_complete = (5 in pts and 6 in pts and 11 in pts and 12 in pts)
        
        return EnhancedFeatures(
            torso_deg_from_vertical=torso_deg,
            left_knee_deg=l_knee,
            right_knee_deg=r_knee, 
            left_hip_deg=l_hip,
            right_hip_deg=r_hip,
            vertical_spread_norm=vspread,
            height_ratio=height_ratio,
            bbox_aspect=bbox_aspect,
            shoulder_hip_distance=shoulder_hip_dist,
            center_of_mass_y=com_y,
            body_angle_confidence=1.0 if angle_conf else 0.5,
            n_keypoints=len(pts),
            torso_visible=torso_visible,
            legs_visible=legs_visible,
            upper_body_complete=upper_body_complete
        )

    def _compute_confidence_scores(self, features: EnhancedFeatures) -> Dict[str, float]:
        """Compute confidence scores for each posture class."""
        th = self.th
        f = features
        
        # Helper function for safe sigmoid
        def safe_sigmoid(x: float, scale: float = th.angle_soft_scale) -> float:
            if not np.isfinite(x):
                return 0.0
            return _sigmoid(x, scale)
            
        # Average knee angle
        knee_angles = [a for a in [f.left_knee_deg, f.right_knee_deg] if np.isfinite(a)]
        avg_knee = np.mean(knee_angles) if knee_angles else float("nan")
        
        # STANDING SCORE
        # Requires: upright torso, straight legs, good height ratio
        stand_torso = safe_sigmoid(th.standing_torso_deg - f.torso_deg_from_vertical)
        stand_knee = safe_sigmoid(avg_knee - th.standing_knee_deg_min) if np.isfinite(avg_knee) else 0.3
        stand_height = safe_sigmoid(f.height_ratio - 0.8) if np.isfinite(f.height_ratio) else 0.3
        standing_score = (stand_torso * stand_knee * stand_height) ** (1/3)  # Geometric mean
        
        # SITTING SCORE (Enhanced for wheelchair and couch)
        if th.wheelchair_mode:
            # Wheelchair mode: more permissive torso angle, less dependent on legs
            sit_torso = safe_sigmoid(th.wheelchair_torso_deg - f.torso_deg_from_vertical)
            sit_knee = safe_sigmoid(avg_knee - th.sitting_knee_deg_min) if np.isfinite(avg_knee) else 0.6
            sit_height = safe_sigmoid(f.height_ratio - th.sitting_height_ratio) if np.isfinite(f.height_ratio) else 0.5
            sit_aspect = safe_sigmoid(1.5 - f.bbox_aspect) if np.isfinite(f.bbox_aspect) else 0.5
        else:
            # Standard sitting detection
            sit_torso = safe_sigmoid(th.enhanced_sitting_deg - f.torso_deg_from_vertical)
            sit_knee = safe_sigmoid(avg_knee - th.sitting_knee_deg_min) if np.isfinite(avg_knee) else 0.4
            sit_height = safe_sigmoid(f.height_ratio - th.sitting_height_ratio) if np.isfinite(f.height_ratio) else 0.4
            sit_aspect = safe_sigmoid(2.0 - f.bbox_aspect) if np.isfinite(f.bbox_aspect) else 0.5
            
        # Couch sitting bonus (if enabled)
        couch_bonus = 1.0
        if th.couch_sitting_enabled and np.isfinite(f.bbox_aspect) and f.bbox_aspect > 1.3:
            if np.isfinite(f.torso_deg_from_vertical) and 30 < f.torso_deg_from_vertical < 75:
                couch_bonus = 1.3  # Boost sitting confidence for couch scenarios
                
        sitting_score = (sit_torso * sit_knee * sit_height * sit_aspect) ** (1/4) * couch_bonus
        sitting_score = min(sitting_score, 0.95)  # Cap the score
        
        # SLEEPING SCORE  
        # Requires: horizontal torso, low vertical spread, wide bbox
        sleep_torso = safe_sigmoid(f.torso_deg_from_vertical - th.sleeping_torso_deg_min)
        sleep_spread = safe_sigmoid(th.sleeping_vspread_max - f.vertical_spread_norm) if np.isfinite(f.vertical_spread_norm) else 0.3
        sleep_aspect = safe_sigmoid(f.bbox_aspect - 1.5) if np.isfinite(f.bbox_aspect) else 0.3
        sleep_height = safe_sigmoid(0.5 - f.height_ratio) if np.isfinite(f.height_ratio) else 0.5
        sleeping_score = (sleep_torso * sleep_spread * sleep_aspect * sleep_height) ** (1/4)
        
        return {
            "standing": float(standing_score),
            "sitting": float(sitting_score), 
            "sleeping": float(sleeping_score)
        }

    def classify(self, kps_xy: np.ndarray, kps_conf: Optional[np.ndarray], 
                bbox_xyxy: np.ndarray) -> PostureResult:
        """Enhanced posture classification."""
        if kps_xy is None or kps_xy.shape != (17, 2):
            return PostureResult("no-person", 0.0, EnhancedFeatures())
            
        valid_mask = self._valid_kps(kps_xy, kps_conf)
        n_valid = int(valid_mask.sum())
        pts = {i: kps_xy[i] for i in range(17) if valid_mask[i]}
        
        # Compute enhanced features
        features = self._compute_enhanced_features(pts, bbox_xyxy)
        
        # Early exit for insufficient data
        if n_valid < self.min_keypoints:
            if features.upper_body_complete and np.isfinite(features.torso_deg_from_vertical):
                # Allow classification with just upper body if torso is clearly visible
                pass
            else:
                return PostureResult("uncertain", 0.1, features)
        
        # Compute confidence scores
        scores = self._compute_confidence_scores(features)
        
        # Select best class
        best_class = max(scores.items(), key=lambda x: x[1])
        label, confidence = best_class
        
        # Apply minimum confidence threshold
        if confidence < self.th.confidence_threshold:
            label = "uncertain"
            confidence = max(0.1, confidence)
            
        # Detailed logging for debugging
        self.logger.debug(
            f"Classification: {label}({confidence:.2f}) | "
            f"Scores - Stand:{scores['standing']:.2f} Sit:{scores['sitting']:.2f} Sleep:{scores['sleeping']:.2f} | "
            f"Features - Torso:{features.torso_deg_from_vertical:.1f}° Height:{features.height_ratio:.2f} "
            f"Aspect:{features.bbox_aspect:.2f} VSpread:{features.vertical_spread_norm:.2f}"
        )
        
        return PostureResult(label, confidence, features, scores)


class EnhancedVisualizer:
    """Enhanced visualizer with better debug information."""
    
    def __init__(self, window_name: str = "Enhanced PostureCam", show_window: bool = True):
        if cv2 is None:
            raise RuntimeError(f"Failed to import cv2: {_cv_import_err}")
        self.window_name = window_name
        self.show_window = show_window
        if show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

    def draw(self, frame_bgr: np.ndarray, bbox_xyxy: Optional[np.ndarray],
            kps_xy: Optional[np.ndarray], kps_conf: Optional[np.ndarray],
            posture: PostureResult, fps: float) -> np.ndarray:
        """Enhanced drawing with detailed information."""
        out = frame_bgr.copy()
        h, w = out.shape[:2]
        
        # Color coding for postures
        colors = {
            'standing': (0, 255, 0),      # Green
            'sitting': (255, 165, 0),     # Orange
            'sleeping': (255, 0, 255),    # Magenta  
            'uncertain': (128, 128, 128), # Gray
            'no-person': (0, 0, 255)      # Red
        }
        
        color = colors.get(posture.label, (255, 255, 255))
        
        # Draw bounding box
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = map(int, bbox_xyxy)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            
        # Draw enhanced skeleton
        if kps_xy is not None and kps_xy.shape == (17, 2):
            # Draw keypoints with confidence-based sizing
            for i, (x, y) in enumerate(kps_xy.astype(int)):
                conf = kps_conf[i] if kps_conf is not None else 0.8
                if conf > 0.3:
                    radius = max(2, int(conf * 5))
                    cv2.circle(out, (int(x), int(y)), radius, color, -1)
                    
            # Draw skeleton connections
            for a, b in COCO_SKELETON:
                if kps_conf is None or (kps_conf[a] > 0.3 and kps_conf[b] > 0.3):
                    pa = tuple(kps_xy[a].astype(int))
                    pb = tuple(kps_xy[b].astype(int))
                    thickness = 3 if (kps_conf is None or (kps_conf[a] > 0.6 and kps_conf[b] > 0.6)) else 2
                    cv2.line(out, pa, pb, color, thickness)
        
        # Enhanced overlay information
        y_pos = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Main status
        status_text = f"Posture: {posture.label.upper()} ({posture.confidence:.2f})"
        cv2.putText(out, status_text, (10, y_pos), font, 0.8, color, 2, cv2.LINE_AA)
        y_pos += 30
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(out, fps_text, (10, y_pos), font, 0.6, (20, 255, 20), 2, cv2.LINE_AA)
        y_pos += 25
        
        # Feature details (if space permits)
        if h > 400:  # Only show if window is tall enough
            f = posture.features
            details = [
                f"Keypoints: {f.n_keypoints}",
                f"Torso: {f.torso_deg_from_vertical:.1f}°" if np.isfinite(f.torso_deg_from_vertical) else "Torso: N/A",
                f"Height Ratio: {f.height_ratio:.2f}" if np.isfinite(f.height_ratio) else "Height: N/A",
                f"Aspect: {f.bbox_aspect:.2f}" if np.isfinite(f.bbox_aspect) else "Aspect: N/A"
            ]
            
            for detail in details:
                cv2.putText(out, detail, (10, y_pos), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
                y_pos += 20
        
        # Confidence breakdown (if available)
        if hasattr(posture, 'confidence_breakdown') and posture.confidence_breakdown:
            y_pos += 10
            cv2.putText(out, "Confidence Breakdown:", (10, y_pos), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += 18
            for cls, conf in posture.confidence_breakdown.items():
                conf_text = f"{cls}: {conf:.2f}"
                cv2.putText(out, conf_text, (15, y_pos), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
                y_pos += 16
        
        # Controls legend
        legend = "q: quit   s: snapshot"
        tw, th = cv2.getTextSize(legend, font, 0.5, 1)[0]
        cv2.rectangle(out, (10, h - th - 20), (10 + tw + 10, h - 6), (0, 0, 0), -1)
        cv2.putText(out, legend, (15, h - 12), font, 0.5, (240, 240, 240), 1, cv2.LINE_AA)

        return out

    def show(self, frame_bgr: np.ndarray) -> int:
        if not self.show_window:
            return -1
        cv2.imshow(self.window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    def close(self):
        if self.show_window:
            cv2.destroyWindow(self.window_name)


class PerSecondLogger:
    """Handles per-second status logging with detailed metrics."""
    
    def __init__(self):
        self.last_log_time = time.time()
        self.frame_count = 0
        self.status_history = []
        self.logger = logging.getLogger(__name__)
        
    def should_log(self) -> bool:
        """Check if a second has passed since last log."""
        current_time = time.time()
        if current_time - self.last_log_time >= 1.0:
            self.last_log_time = current_time
            return True
        return False
        
    def log_status(self, posture: PostureResult, fps: float):
        """Log detailed status every second."""
        if not self.should_log():
            return
            
        f = posture.features
        
        # Build status message
        status_parts = [
            f"STATUS: {posture.label.upper()}",
            f"Confidence: {posture.confidence:.3f}",
            f"FPS: {fps:.1f}",
            f"Keypoints: {f.n_keypoints}/17"
        ]
        
        # Add key features if available
        if np.isfinite(f.torso_deg_from_vertical):
            status_parts.append(f"Torso: {f.torso_deg_from_vertical:.1f}°")
        if np.isfinite(f.height_ratio):
            status_parts.append(f"Height: {f.height_ratio:.2f}")
        if np.isfinite(f.bbox_aspect):
            status_parts.append(f"Aspect: {f.bbox_aspect:.2f}")
            
        # Add visibility flags
        visibility = []
        if f.torso_visible:
            visibility.append("Torso")
        if f.legs_visible:
            visibility.append("Legs")
        if visibility:
            status_parts.append(f"Visible: {'+'.join(visibility)}")
            
        status_message = " | ".join(status_parts)
        self.logger.info(status_message)
        
        # Store in history for analysis
        self.status_history.append({
            'timestamp': time.time(),
            'posture': posture.label,
            'confidence': posture.confidence,
            'features': f
        })
        
        # Keep only last 60 seconds of history
        cutoff_time = time.time() - 60
        self.status_history = [h for h in self.status_history if h['timestamp'] > cutoff_time]


# Pipeline functions
def pick_device() -> str:
    """Enhanced device selection with more detailed logging."""
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown GPU"
        logging.info(f"CUDA available - Using GPU: {gpu_name}")
        return "cuda"
    else:
        logging.info("CUDA not available - Using CPU")
        return "cpu"


def select_person(boxes_xyxy: np.ndarray, boxes_conf: np.ndarray, policy: str = "largest") -> int:
    """Enhanced person selection with validation."""
    if len(boxes_xyxy) == 0:
        return -1
        
    if policy == "highest-conf":
        return int(np.argmax(boxes_conf))
    
    # Default: largest area with aspect ratio validation
    wh = boxes_xyxy[:, 2:4] - boxes_xyxy[:, 0:2]
    areas = wh[:, 0] * wh[:, 1]
    
    # Prefer reasonable aspect ratios (not too tall/thin or wide/flat)
    aspects = wh[:, 0] / np.maximum(wh[:, 1], 1.0)
    reasonable_mask = (aspects > 0.3) & (aspects < 4.0)
    
    if reasonable_mask.any():
        # Select largest among reasonable detections
        reasonable_areas = np.where(reasonable_mask, areas, 0)
        return int(np.argmax(reasonable_areas))
    else:
        # Fall back to largest overall
        return int(np.argmax(areas))


def run_enhanced_posture_cam(args: argparse.Namespace) -> None:
    """Enhanced main pipeline with improved error handling and logging."""
    
    # Enhanced logging setup
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # Remove any existing handlers
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # Startup banner
    logging.info("="*60)
    logging.info("🚀 ENHANCED POSTURE CAM - Starting Up")
    logging.info("="*60)
    
    # Log configuration
    config_items = [
        ("Model", args.model or "Auto-select (yolo11x->yolo11n->yolo8n->yolo5n)"),
        ("Confidence", args.conf),
        ("Image Size", args.imgsz),
        ("Source", args.src if args.src else f"Camera {args.camera}"),
        ("Wheelchair Mode", "Enabled" if args.wheelchair_mode else "Disabled"),
        ("Couch Detection", "Enabled" if args.couch_sitting_enabled else "Disabled"),
        ("Min Keypoints", args.min_keypoints),
        ("Per-Second Logging", "Enabled" if args.per_second_log else "Disabled")
    ]
    
    for key, value in config_items:
        logging.info(f"├─ {key}: {value}")
    
    logging.info("="*60)
    
    # Device selection
    device = pick_device()
    
    # Initialize components with enhanced error handling
    try:
        detector = EnhancedPoseDetector(
            model_name=args.model, 
            conf=args.conf, 
            imgsz=args.imgsz, 
            device=device
        )
    except Exception as e:
        logging.error(f"❌ Failed to initialize pose detector: {e}")
        return
        
    # Setup enhanced thresholds
    thresholds = EnhancedThresholds(
        standing_torso_deg=args.standing_torso_deg,
        standing_knee_deg_min=args.standing_knee_deg_min,
        sitting_knee_deg_min=args.sitting_knee_deg_min,
        sitting_torso_deg_max=args.sitting_torso_deg_max,
        enhanced_sitting_deg=args.enhanced_sitting_deg,
        wheelchair_torso_deg=args.wheelchair_torso_deg,
        sitting_height_ratio=args.sitting_height_ratio,
        sleeping_torso_deg_min=args.sleeping_torso_deg_min,
        sleeping_vspread_max=args.sleeping_vspread_max,
        angle_soft_scale=args.angle_soft_scale,
        wheelchair_mode=args.wheelchair_mode,
        couch_sitting_enabled=args.couch_sitting_enabled
    )
    
    classifier = EnhancedPostureClassifier(
        thresholds, 
        kp_conf_min=args.kp_conf, 
        min_keypoints=args.min_keypoints
    )
    
    # Video capture with enhanced error handling
    if cv2 is None:
        logging.error(f"❌ OpenCV not available: {_cv_import_err}")
        return
        
    src = args.src if args.src is not None else int(args.camera)
    cap = cv2.VideoCapture(src)
    
    if not cap.isOpened():
        logging.error(f"❌ Failed to open video source: {src}")
        return
        
    # Try to set some optimal capture properties
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps_cap = cap.get(cv2.CAP_PROP_FPS)
        logging.info(f"✓ Video source opened: {width}x{height} @ {fps_cap} FPS")
    except Exception as e:
        logging.warning(f"Could not set capture properties: {e}")
        
    # Initialize visualizer and logger
    visualizer = EnhancedVisualizer(show_window=bool(args.display))
    status_logger = PerSecondLogger() if args.per_second_log else None
    
    # Create snapshots directory
    os.makedirs("snapshots", exist_ok=True)
    
    # CSV logging setup
    csv_file = None
    csv_writer = None
    if args.csv:
        try:
            csv_file = open(args.csv, "a", newline="", encoding="utf-8")
            csv_writer = csv.writer(csv_file)
            
            # Write header if file is empty
            if os.path.getsize(args.csv) == 0:
                header = [
                    "timestamp", "posture", "confidence", "fps", "n_keypoints",
                    "torso_deg", "left_knee_deg", "right_knee_deg", "left_hip_deg", "right_hip_deg",
                    "height_ratio", "bbox_aspect", "vertical_spread", "shoulder_hip_dist",
                    "torso_visible", "legs_visible", "upper_body_complete", "standing_score", 
                    "sitting_score", "sleeping_score"
                ]
                csv_writer.writerow(header)
                logging.info(f"✓ CSV logging enabled: {args.csv}")
        except Exception as e:
            logging.error(f"❌ Failed to setup CSV logging: {e}")
            csv_writer = None
    
    # Performance tracking
    fps_tracker = []
    fps_window = 30  # Average over 30 frames
    frame_count = 0
    start_time = time.time()
    last_status_log = time.time()
    
    logging.info("🎥 Starting main processing loop...")
    
    try:
        while True:
            frame_start = time.time()
            
            # Capture frame with retry logic
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️  Frame capture failed, attempting reconnection...")
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(src)
                if not cap.isOpened():
                    logging.error("❌ Failed to reconnect to video source")
                    break
                continue
                
            try:
                # Run detection
                results = detector.infer(frame)
                
                # Parse results
                if not results or len(results) == 0:
                    posture = PostureResult("no-person", 0.0, EnhancedFeatures())
                    bbox_sel = None
                    kps_xy_sel = None
                    kps_conf_sel = None
                else:
                    res = results[0]
                    
                    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                        posture = PostureResult("no-person", 0.0, EnhancedFeatures())
                        bbox_sel = None
                        kps_xy_sel = None
                        kps_conf_sel = None
                    else:
                        # Extract detection data
                        boxes_xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else res.boxes.xyxy.numpy()
                        boxes_conf = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else res.boxes.conf.numpy()
                        kps_xy = res.keypoints.xy.cpu().numpy() if hasattr(res.keypoints.xy, "cpu") else res.keypoints.xy.numpy()
                        kps_conf = None
                        if hasattr(res.keypoints, "conf") and res.keypoints.conf is not None:
                            kps_conf = res.keypoints.conf.cpu().numpy() if hasattr(res.keypoints.conf, "cpu") else res.keypoints.conf.numpy()
                            
                        # Select best person
                        idx = select_person(boxes_xyxy, boxes_conf, policy=args.select)
                        if idx >= 0:
                            bbox_sel = boxes_xyxy[idx]
                            kps_xy_sel = kps_xy[idx]
                            kps_conf_sel = kps_conf[idx] if kps_conf is not None else None
                            
                            # Classify posture
                            posture = classifier.classify(kps_xy_sel, kps_conf_sel, bbox_sel)
                        else:
                            posture = PostureResult("no-person", 0.0, EnhancedFeatures())
                            bbox_sel = None
                            kps_xy_sel = None
                            kps_conf_sel = None
                            
            except Exception as e:
                logging.error(f"❌ Processing error: {e}")
                posture = PostureResult("error", 0.0, EnhancedFeatures())
                bbox_sel = None
                kps_xy_sel = None
                kps_conf_sel = None
                
            # FPS calculation
            frame_end = time.time()
            frame_time = frame_end - frame_start
            current_fps = 1.0 / max(frame_time, 0.001)
            
            fps_tracker.append(current_fps)
            if len(fps_tracker) > fps_window:
                fps_tracker.pop(0)
            avg_fps = np.mean(fps_tracker)
            
            # Per-second status logging
            if status_logger:
                status_logger.log_status(posture, avg_fps)
                
            # CSV logging
            if csv_writer:
                try:
                    f = posture.features
                    scores = getattr(posture, 'confidence_breakdown', {})
                    
                    row_data = [
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                        posture.label, f"{posture.confidence:.4f}", f"{avg_fps:.2f}", f.n_keypoints,
                        f"{f.torso_deg_from_vertical:.2f}" if np.isfinite(f.torso_deg_from_vertical) else "",
                        f"{f.left_knee_deg:.2f}" if np.isfinite(f.left_knee_deg) else "",
                        f"{f.right_knee_deg:.2f}" if np.isfinite(f.right_knee_deg) else "",
                        f"{f.left_hip_deg:.2f}" if np.isfinite(f.left_hip_deg) else "",
                        f"{f.right_hip_deg:.2f}" if np.isfinite(f.right_hip_deg) else "",
                        f"{f.height_ratio:.3f}" if np.isfinite(f.height_ratio) else "",
                        f"{f.bbox_aspect:.3f}" if np.isfinite(f.bbox_aspect) else "",
                        f"{f.vertical_spread_norm:.3f}" if np.isfinite(f.vertical_spread_norm) else "",
                        f"{f.shoulder_hip_distance:.3f}" if np.isfinite(f.shoulder_hip_distance) else "",
                        str(f.torso_visible), str(f.legs_visible), str(f.upper_body_complete),
                        f"{scores.get('standing', 0.0):.3f}",
                        f"{scores.get('sitting', 0.0):.3f}",
                        f"{scores.get('sleeping', 0.0):.3f}"
                    ]
                    csv_writer.writerow(row_data)
                    csv_file.flush()
                except Exception as e:
                    logging.error(f"CSV logging error: {e}")
                    
            # Visualization
            if args.display:
                try:
                    canvas = visualizer.draw(frame, bbox_sel, kps_xy_sel, kps_conf_sel, posture, avg_fps)
                    key = visualizer.show(canvas)
                    
                    if key == ord("q"):
                        logging.info("👋 Quit requested by user")
                        break
                    elif key == ord("s"):
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"snapshots/enhanced_posture_{timestamp}.jpg"
                        cv2.imwrite(filename, canvas)
                        logging.info(f"📸 Snapshot saved: {filename}")
                except Exception as e:
                    logging.error(f"Visualization error: {e}")
            else:
                # Small delay in headless mode to prevent CPU overload
                time.sleep(0.005)
                
            frame_count += 1
            
            # Periodic performance logging (every 5 seconds)
            current_time = time.time()
            if current_time - last_status_log >= 5.0:
                runtime = current_time - start_time
                logging.info(f"📊 Performance: {avg_fps:.1f} FPS avg, {frame_count} frames in {runtime:.1f}s")
                last_status_log = current_time
                
    except KeyboardInterrupt:
        logging.info("👋 Interrupted by user (Ctrl+C)")
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")
    finally:
        # Cleanup
        logging.info("🧹 Cleaning up...")
        try:
            cap.release()
        except:
            pass
        try:
            visualizer.close()
        except:
            pass
        if csv_file:
            try:
                csv_file.close()
            except:
                pass
        
        total_runtime = time.time() - start_time
        avg_fps_final = frame_count / max(total_runtime, 0.001)
        logging.info(f"📈 Final stats: {frame_count} frames, {avg_fps_final:.1f} FPS avg, {total_runtime:.1f}s runtime")
        logging.info("✅ Shutdown complete")


def parse_enhanced_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Enhanced argument parser with new options."""
    p = argparse.ArgumentParser(
        description="Enhanced real-time posture classification optimized for wheelchair and couch scenarios",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output options
    io_group = p.add_argument_group("Input/Output Options")
    io_group.add_argument("--camera", type=int, default=0, help="Camera index")
    io_group.add_argument("--src", type=str, help="Video source (file/rtsp/http), overrides --camera")
    io_group.add_argument("--display", dest="display", action="store_true", help="Show display window")
    io_group.add_argument("--headless", dest="display", action="store_false", help="Run headless")
    p.set_defaults(display=True)
    
    # Model options
    model_group = p.add_argument_group("Model Options")
    model_group.add_argument("--model", type=str, help="YOLO model (auto-selects yolo11x->yolo11n->yolo8n->yolo5n)")
    model_group.add_argument("--conf", type=float, default=0.20, help="Detection confidence threshold")
    model_group.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    
    # Detection options  
    detection_group = p.add_argument_group("Detection Options")
    detection_group.add_argument("--max-persons", type=int, default=1, help="Max persons to process")
    detection_group.add_argument("--select", choices=["largest", "highest-conf"], default="largest", help="Person selection policy")
    detection_group.add_argument("--kp-conf", type=float, default=0.25, help="Min keypoint confidence")
    detection_group.add_argument("--min-keypoints", type=int, default=6, help="Min keypoints for classification")
    
    # Enhanced posture thresholds
    posture_group = p.add_argument_group("Posture Classification Thresholds")
    posture_group.add_argument("--standing-torso-deg", type=float, default=25.0, help="Max torso angle for standing")
    posture_group.add_argument("--standing-knee-deg-min", type=float, default=160.0, help="Min knee angle for standing")
    posture_group.add_argument("--sitting-knee-deg-min", type=float, default=70.0, help="Min knee flexion for sitting")
    posture_group.add_argument("--sitting-torso-deg-max", type=float, default=35.0, help="Max torso angle for standard sitting")
    posture_group.add_argument("--enhanced-sitting-deg", type=float, default=45.0, help="Max torso angle for enhanced sitting")
    posture_group.add_argument("--wheelchair-torso-deg", type=float, default=50.0, help="Max torso angle for wheelchair sitting")
    posture_group.add_argument("--sitting-height-ratio", type=float, default=0.6, help="Min height ratio for sitting vs sleeping")
    posture_group.add_argument("--sleeping-torso-deg-min", type=float, default=55.0, help="Min torso angle for sleeping")
    posture_group.add_argument("--sleeping-vspread-max", type=float, default=0.4, help="Max vertical spread for sleeping")
    posture_group.add_argument("--angle-soft-scale", type=float, default=12.0, help="Sigmoid scale for soft scoring")
    
    # Enhanced mode options
    mode_group = p.add_argument_group("Enhanced Mode Options")
    mode_group.add_argument("--wheelchair-mode", action="store_true", help="Enable wheelchair-optimized detection")
    mode_group.add_argument("--couch-sitting-enabled", action="store_true", default=True, help="Treat couch laying as sitting")
    mode_group.add_argument("--per-second-log", action="store_true", default=True, help="Log status every second")
    
    # Logging options
    log_group = p.add_argument_group("Logging Options")
    log_group.add_argument("--csv", type=str, help="CSV output file for detailed logging")
    log_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    log_group.add_argument("--log-file", type=str, help="Log file path")
    
    # Utility options
    util_group = p.add_argument_group("Utility Options")
    util_group.add_argument("--write-reqs", action="store_true", help="Write requirements.txt and exit")
    
    args = p.parse_args(argv)
    
    # Handle requirements writing
    if args.write_reqs:
        with open("requirements.txt", "w", encoding="utf-8") as f:
            f.write(ENHANCED_REQUIREMENTS.strip() + "\n")
        print("✅ Enhanced requirements.txt written")
        sys.exit(0)
        
    return args


# Requirements
ENHANCED_REQUIREMENTS = """
# Enhanced requirements for posture_cam.py
ultralytics>=8.1.0
opencv-python>=4.7.0
numpy>=1.23.0
torch>=1.12.0
torchvision>=0.13.0
"""


def main():
    """Enhanced main function."""
    args = parse_enhanced_args()
    run_enhanced_posture_cam(args)


if __name__ == "__main__":
    main()