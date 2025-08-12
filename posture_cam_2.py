#!/usr/bin/env python3
"""
posture_cam.py

Real-time posture classification (standing / sitting / sleeping) from a live webcam
using MediaPipe Pose (33-keypoint skeleton). Designed to handle partial occlusions
(e.g., legs hidden behind a desk) and fixes stray skeleton lines to (0,0).

Usage
------
$ python posture_cam.py
    --camera 0                   # camera index (default 0)
    --src /path/or/url           # optional: file path
    --display True               # show OpenCV window (default True)
    --headless                   # run without window
    --kp-thr 0.5                 # landmark visibility threshold (0..1)
    --conf-thr 0.5               # min confidence to report label (else 'uncertain')
    --smooth 5                   # majority smoothing over last N labels (0 disables)
    --csv out.csv                # optional per-frame CSV logging

Dependencies
------------
- Python 3.9+
- mediapipe
- opencv-python
- numpy

High-level algorithm
--------------------
1) Pose estimation (MediaPipe) → 33 landmarks with normalized (x,y) and 'visibility'.
2) Compute features:
   - Torso inclination: angle between (shoulder_center → hip_center) and vertical axis.
   - Knee and hip angles (when visible).
   - Vertical spread of visible keypoints (normalized by bbox height from landmarks).
   - BBox aspect ratio as a fallback (wide implies lying).
   - Legs visibility status (knees/ankles confidently visible).
3) Occlusion-aware scoring (soft, weighted evidence):
   - Standing: torso vertical + knees straight + high vertical spread (if reliable).
   - Sitting: torso vertical + knees/hips flexed; if legs are NOT visible but torso is upright,
     bias to sitting (common desk occlusion).
   - Sleeping: torso horizontal + low vertical spread OR wide bbox.
4) Pick the highest score; if score < conf-thr → 'uncertain'.
5) Optional temporal smoothing via recent-majority voting.

Keys
----
q: quit     s: save snapshot (with overlays)

Notes
-----
- This script defaults to CPU and easily meets ~10-30 FPS on a modern laptop.
"""

from __future__ import annotations
import argparse, csv, math, os, sys, time
from collections import deque, Counter
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np
import mediapipe as mp


# ----------------------------
# Geometry helpers
# ----------------------------

def sigmoid(z: float) -> float:
    # numerically stable logistic
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    else:
        ez = math.exp(z)
        return ez / (1.0 + ez)

def angle_between(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-6) -> float:
    """Angle (deg) between 2D vectors."""
    v1 = np.asarray(v1, dtype=float); v2 = np.asarray(v2, dtype=float)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps: return float('nan')
    c = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(c))

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle ABC (deg) with vertex at B."""
    return angle_between(a - b, c - b)

def valid_xy(pt: Optional[Tuple[float, float]]) -> bool:
    return (pt is not None and np.isfinite(pt).all())


# ----------------------------
# MediaPipe setup
# ----------------------------

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

# Useful landmark indices
LMS = {
    "l_shoulder": PoseLandmark.LEFT_SHOULDER.value,
    "r_shoulder": PoseLandmark.RIGHT_SHOULDER.value,
    "l_hip": PoseLandmark.LEFT_HIP.value,
    "r_hip": PoseLandmark.RIGHT_HIP.value,
    "l_knee": PoseLandmark.LEFT_KNEE.value,
    "r_knee": PoseLandmark.RIGHT_KNEE.value,
    "l_ankle": PoseLandmark.LEFT_ANKLE.value,
    "r_ankle": PoseLandmark.RIGHT_ANKLE.value,
}

# ----------------------------
# Data containers
# ----------------------------

@dataclass
class PoseFeatures:
    torso_deg: float = float('nan')
    knee_l_deg: float = float('nan')
    knee_r_deg: float = float('nan')
    hip_l_deg: float = float('nan')
    hip_r_deg: float = float('nan')
    vertical_spread: float = float('nan')
    bbox_w: float = float('nan')
    bbox_h: float = float('nan')
    aspect: float = float('nan')
    legs_visible: bool = False

@dataclass
class PostureResult:
    label: str
    confidence: float
    f: PoseFeatures

# ----------------------------
# Pose extractor
# ----------------------------

class MediaPipePose:
    def __init__(self, kp_thr: float = 0.5):
        self.pose = mp_pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.kp_thr = float(kp_thr)

    def process(self, frame_bgr: np.ndarray):
        # Mediapipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(frame_rgb)
        return res

    def landmarks_to_xyc(self, res, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            xy: (33,2) pixel coords (nan if low conf)
            c:  (33,) visibility confidence in [0,1] (0 for invalid)
        """
        xy = np.full((33, 2), np.nan, dtype=float)
        c  = np.zeros((33,), dtype=float)

        if res is None or res.pose_landmarks is None:
            return xy, c

        for i, lm in enumerate(res.pose_landmarks.landmark):
            x = lm.x; y = lm.y; v = lm.visibility
            # Filter: inside image bounds and decent visibility
            if 0.0 < x < 1.0 and 0.0 < y < 1.0 and v >= self.kp_thr:
                xy[i, 0] = x * w
                xy[i, 1] = y * h
                c[i] = float(v)
            # else leave as NaN / 0

        return xy, c


# ----------------------------
# Posture classifier (occlusion-aware)
# ----------------------------

class PostureClassifier:
    def __init__(self,
                 standing_torso_deg: float = 30.0,
                 sitting_knee_deg: float = 110.0,
                 sitting_hip_deg: float = 120.0,
                 sleeping_torso_deg: float = 65.0,
                 vs_standing_min: float = 0.50,
                 vs_sleeping_max: float = 0.45,
                 vs_sit_lo: float = 0.30,
                 vs_sit_hi: float = 0.95,
                 ang_soft: float = 8.0,
                 vs_soft: float = 12.0):
        self.standing_torso_deg = standing_torso_deg
        self.sitting_knee_deg   = sitting_knee_deg
        self.sitting_hip_deg    = sitting_hip_deg
        self.sleeping_torso_deg = sleeping_torso_deg

        self.vs_standing_min = vs_standing_min
        self.vs_sleeping_max = vs_sleeping_max
        self.vs_sit_lo = vs_sit_lo
        self.vs_sit_hi = vs_sit_hi

        self.ang_soft = ang_soft
        self.vs_soft = vs_soft

        self.standing_knee_min = 150.0  # near-straight legs

    @staticmethod
    def _center(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if not valid_xy(a) or not valid_xy(b): return None
        return (a + b) / 2.0

    def _compute_features(self, xy: np.ndarray, c: np.ndarray) -> PoseFeatures:
        def p(idx: int) -> Optional[np.ndarray]:
            if idx < 0 or idx >= xy.shape[0]: return None
            pt = xy[idx]
            if not np.isfinite(pt).all(): return None
            return pt

        l_sh = p(LMS["l_shoulder"]); r_sh = p(LMS["r_shoulder"])
        l_hp = p(LMS["l_hip"]);      r_hp = p(LMS["r_hip"])
        l_kn = p(LMS["l_knee"]);     r_kn = p(LMS["r_knee"])
        l_an = p(LMS["l_ankle"]);    r_an = p(LMS["r_ankle"])

        shoulder_c = self._center(l_sh, r_sh)
        hip_c      = self._center(l_hp, r_hp)

        torso_deg = float('nan')
        if valid_xy(shoulder_c) and valid_xy(hip_c):
            vec = hip_c - shoulder_c
            torso_deg = angle_between(vec, np.array([0.0, 1.0]))

        # Knee angles
        knee_l = float('nan'); knee_r = float('nan')
        if valid_xy(l_hp) and valid_xy(l_kn) and valid_xy(l_an):
            knee_l = joint_angle(l_hp, l_kn, l_an)
        if valid_xy(r_hp) and valid_xy(r_kn) and valid_xy(r_an):
            knee_r = joint_angle(r_hp, r_kn, r_an)

        # Hip angles
        hip_l = float('nan'); hip_r = float('nan')
        if valid_xy(l_sh) and valid_xy(l_hp) and valid_xy(l_kn):
            hip_l = joint_angle(l_sh, l_hp, l_kn)
        if valid_xy(r_sh) and valid_xy(r_hp) and valid_xy(r_kn):
            hip_r = joint_angle(r_sh, r_hp, r_kn)

        # Bounding box from visible landmarks
        vis_mask = np.isfinite(xy[:, 0]) & np.isfinite(xy[:, 1])
        vs = float('nan'); bw = float('nan'); bh = float('nan'); aspect = float('nan')
        if np.any(vis_mask):
            xs = xy[vis_mask, 0]; ys = xy[vis_mask, 1]
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            bw = max(1.0, x2 - x1); bh = max(1.0, y2 - y1)
            vs = float(np.clip((y2 - y1) / bh, 0.0, 2.0))  # normalized by bbox height → typically ~1.0
            aspect = bw / bh

        # Legs visible if at least one side has knee+ankle confident
        def conf_ok(idx): return (idx is not None) and (idx >= 0) and (idx < c.shape[0]) and (c[idx] > 0.5)
        left_leg_ok  = conf_ok(LMS["l_knee"]) and conf_ok(LMS["l_ankle"])
        right_leg_ok = conf_ok(LMS["r_knee"]) and conf_ok(LMS["r_ankle"])
        legs_visible = bool(left_leg_ok or right_leg_ok)

        return PoseFeatures(
            torso_deg=torso_deg,
            knee_l_deg=float(knee_l),
            knee_r_deg=float(knee_r),
            hip_l_deg=float(hip_l),
            hip_r_deg=float(hip_r),
            vertical_spread=vs,
            bbox_w=bw, bbox_h=bh, aspect=aspect,
            legs_visible=legs_visible
        )

    def _weighted_mean(self, comps: List[Tuple[float, float]]) -> float:
        # comps: list of (score, weight)
        acc = 0.0; wsum = 0.0
        for s, w in comps:
            if np.isfinite(s):
                acc += s * w; wsum += w
        return (acc / wsum) if wsum > 0 else 0.0

    def _score_standing(self, f: PoseFeatures) -> float:
        comps = []
        if np.isfinite(f.torso_deg):
            comps.append((sigmoid((self.standing_torso_deg - f.torso_deg) / self.ang_soft), 0.45))
        # avg knee straightness
        knees = []
        if np.isfinite(f.knee_l_deg): knees.append(f.knee_l_deg)
        if np.isfinite(f.knee_r_deg): knees.append(f.knee_r_deg)
        if knees:
            avg_k = float(np.mean(knees))
            comps.append((sigmoid((avg_k - self.standing_knee_min) / self.ang_soft), 0.35))
        # vertical spread high
        if np.isfinite(f.vertical_spread):
            comps.append((sigmoid((f.vertical_spread - self.vs_standing_min) / (self.vs_soft * 0.5)), 0.20))
        return self._weighted_mean(comps)

    def _score_sitting(self, f: PoseFeatures) -> float:
        comps = []
        if np.isfinite(f.torso_deg):
            comps.append((sigmoid((self.standing_torso_deg - f.torso_deg) / self.ang_soft), 0.35))
        # knees flexed (smaller angle means more flexion)
        kflex = []
        for k in (f.knee_l_deg, f.knee_r_deg):
            if np.isfinite(k):
                kflex.append(sigmoid((self.sitting_knee_deg - k) / self.ang_soft))
        if kflex:
            comps.append((float(np.mean(kflex)), 0.35))
        # hips flexed
        hflex = []
        for h in (f.hip_l_deg, f.hip_r_deg):
            if np.isfinite(h):
                hflex.append(sigmoid((self.sitting_hip_deg - h) / self.ang_soft))
        if hflex:
            comps.append((float(np.mean(hflex)), 0.15))
        # vertical spread moderate band
        if np.isfinite(f.vertical_spread):
            lo, hi = self.vs_sit_lo, self.vs_sit_hi
            inside = sigmoid((f.vertical_spread - lo) / (self.vs_soft * 0.5)) * \
                     sigmoid((hi - f.vertical_spread) / (self.vs_soft * 0.5))
            comps.append((inside, 0.15))

        base = self._weighted_mean(comps)

        # Special occlusion bias: upright torso + legs not visible ⇒ likely sitting at desk
        if np.isfinite(f.torso_deg) and (f.torso_deg <= self.standing_torso_deg + 8.0) and (not f.legs_visible):
            base = max(base, 0.6)  # guarantee moderate sitting confidence
        return base

    def _score_sleeping(self, f: PoseFeatures) -> float:
        comps = []
        if np.isfinite(f.torso_deg):
            comps.append((sigmoid((f.torso_deg - self.sleeping_torso_deg) / self.ang_soft), 0.60))
        if np.isfinite(f.vertical_spread):
            comps.append((sigmoid((self.vs_sleeping_max - f.vertical_spread) / (self.vs_soft * 0.5)), 0.40))
        elif np.isfinite(f.aspect):
            # fallback: wide bbox implies lying
            comps.append((sigmoid((f.aspect - 1.25) / 0.25), 0.35))
        return self._weighted_mean(comps)

    def classify(self, xy: np.ndarray, c: np.ndarray) -> PostureResult:
        f = self._compute_features(xy, c)
        scores = {
            "standing": self._score_standing(f),
            "sitting":  self._score_sitting(f),
            "sleeping": self._score_sleeping(f),
        }
        labels = list(scores.keys())
        vals = [scores[k] for k in labels]
        idx = int(np.argmax(vals))
        return PostureResult(labels[idx], float(vals[idx]), f)


# ----------------------------
# Visualization
# ----------------------------

class Visualizer:
    def __init__(self, kp_thr: float = 0.5):
        self.kp_thr = kp_thr
        self.connections = list(mp_pose.POSE_CONNECTIONS)

    def _ok(self, xy: np.ndarray, c: np.ndarray, i: int) -> bool:
        if i < 0 or i >= xy.shape[0]: return False
        if c[i] < self.kp_thr: return False
        px, py = xy[i]
        if not np.isfinite(px) or not np.isfinite(py): return False
        return True

    def draw(self, frame: np.ndarray, xy: np.ndarray, c: np.ndarray, bbox_color=(0, 200, 255)) -> None:
        # draw bones
        for (i, j) in self.connections:
            if self._ok(xy, c, i) and self._ok(xy, c, j):
                p1 = (int(xy[i,0]), int(xy[i,1]))
                p2 = (int(xy[j,0]), int(xy[j,1]))
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

        # draw joints
        for i in range(xy.shape[0]):
            if self._ok(xy, c, i):
                cv2.circle(frame, (int(xy[i,0]), int(xy[i,1])), 3, (255, 0, 0), -1)

        # bbox from visible joints
        vis = np.where(c >= self.kp_thr)[0]
        if vis.size >= 2:
            xs = xy[vis,0]; ys = xy[vis,1]
            x1, y1 = int(np.nanmin(xs)), int(np.nanmin(ys))
            x2, y2 = int(np.nanmax(xs)), int(np.nanmax(ys))
            cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

    def overlay_text(self, frame, posture: str, conf: float, fps: float, f: PoseFeatures):
        x, y, h = 10, 24, 22
        cv2.putText(frame, f"Posture: {posture} ({conf:.2f})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 255, 20), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.1f}", (x, y+h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        # tiny debug line (optional)
        # cv2.putText(frame, f"T:{f.torso_deg:.1f} K:({f.knee_l_deg:.0f},{f.knee_r_deg:.0f}) VS:{f.vertical_spread:.2f}", (x, y+2*h),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)


# ----------------------------
# Pipeline
# ----------------------------

def run(args: argparse.Namespace):
    disp = bool(args.display) and not args.headless

    # Video capture
    cap_src = args.src if args.src is not None else int(args.camera)
    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open source: {cap_src}")
        return

    pose = MediaPipePose(kp_thr=args.kp_thr)
    clf  = PostureClassifier()
    vis  = Visualizer(kp_thr=args.kp_thr)

    csv_writer = None
    csv_fh = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_fh = open(args.csv, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fh)
        csv_writer.writerow([
            "timestamp","label","conf","fps",
            "torso_deg","knee_l","knee_r","hip_l","hip_r",
            "vspread","bbox_w","bbox_h","aspect","legs_visible"
        ])

    # FPS
    t_prev = time.time()
    dt_hist = deque(maxlen=60)
    fps_display = 0.0

    # Temporal smoothing (majority vote)
    label_hist = deque(maxlen=max(0, args.smooth))

    snapshots_dir = os.path.join(".", "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[WARN] Stream ended or frame grab failed.")
                break

            h, w = frame.shape[:2]
            res = pose.process(frame)
            xy, confs = pose.landmarks_to_xyc(res, w, h)

            # Classify posture
            result = clf.classify(xy, confs)
            raw_label, raw_conf = result.label, result.confidence

            # Apply confidence gate
            label = raw_label if raw_conf >= args.conf_thr else "uncertain"
            conf  = raw_conf if raw_conf >= args.conf_thr else raw_conf

            # Temporal smoothing (majority over last N != 'uncertain')
            if args.smooth > 0:
                if label != "uncertain":
                    label_hist.append(label)
                if len(label_hist) >= 1:
                    label = Counter(label_hist).most_common(1)[0][0]

            # FPS
            t_now = time.time()
            dt = t_now - t_prev; t_prev = t_now
            if dt > 0:
                dt_hist.append(dt)
                fps_display = 1.0 / (sum(dt_hist)/len(dt_hist))

            # Draw
            if disp:
                vis.draw(frame, xy, confs)
                vis.overlay_text(frame, label, conf, fps_display, result.f)
                # Print posture info once every second
                if not hasattr(run, "_last_print") or (time.time() - run._last_print) > 1.0:
                    print(f"[INFO] Posture: {label} (conf: {conf:.2f}, fps: {fps_display:.2f})")
                    run._last_print = time.time()
                cv2.imshow("Posture (MediaPipe)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('s'):
                    snap_path = os.path.join(snapshots_dir, f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap_path, frame)
                    print(f"[INFO] Saved {snap_path}")

            # CSV
            if csv_writer is not None:
                f = result.f
                csv_writer.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    label, f"{conf:.4f}", f"{fps_display:.2f}",
                    f"{f.torso_deg:.2f}", f"{f.knee_l_deg:.2f}", f"{f.knee_r_deg:.2f}",
                    f"{f.hip_l_deg:.2f}", f"{f.hip_r_deg:.2f}",
                    f"{f.vertical_spread:.3f}", f"{f.bbox_w:.1f}", f"{f.bbox_h:.1f}",
                    f"{f.aspect:.3f}", int(f.legs_visible)
                ])

    except KeyboardInterrupt:
        pass
    finally:
        try: cap.release()
        except: pass
        if disp:
            try: cv2.destroyAllWindows()
            except: pass
        if csv_fh is not None:
            try: csv_fh.close()
            except: pass


# ----------------------------
# CLI
# ----------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Real-time standing/sitting/sleeping classifier using MediaPipe Pose.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("--camera", type=int, default=0, help="Camera index")
    p.add_argument("--src", type=str, default=None, help="Optional video file path (overrides --camera)")
    p.add_argument("--display", type=lambda x: str(x).lower() in ("1","true","yes"), default=True, help="Show OpenCV window")
    p.add_argument("--headless", action="store_true", help="Disable window")
    p.add_argument("--kp-thr", type=float, default=0.5, help="Landmark visibility threshold (0..1)")
    p.add_argument("--conf-thr", type=float, default=0.5, help="Min confidence to accept label, else 'uncertain'")
    p.add_argument("--smooth", type=int, default=5, help="Majority smoothing window (0 disables)")
    p.add_argument("--csv", type=str, default=None, help="Optional CSV output path")
    args = p.parse_args(argv)
    if args.headless: args.display = False
    return args

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
