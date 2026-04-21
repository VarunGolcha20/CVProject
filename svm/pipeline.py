"""
GestureBall CV Pipeline
=======================
Core pose extraction, feature engineering, and benchmarking harness.
Run this file directly to test the pipeline on your webcam.

Feature vector: 32 dimensions (expanded from 16)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json

# ── Landmark indices (MediaPipe Pose) ────────────────────────────────────────
LM = mp.solutions.pose.PoseLandmark
KEYPOINTS = {
    "left_hip":       LM.LEFT_HIP,
    "right_hip":      LM.RIGHT_HIP,
    "left_knee":      LM.LEFT_KNEE,
    "right_knee":     LM.RIGHT_KNEE,
    "left_ankle":     LM.LEFT_ANKLE,
    "right_ankle":    LM.RIGHT_ANKLE,
    "left_shoulder":  LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_foot":      LM.LEFT_FOOT_INDEX,
    "right_foot":     LM.RIGHT_FOOT_INDEX,
}

FEATURE_DIM = 32   # update this if you ever add more features


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class PoseFrame:
    """One frame's worth of pose data."""
    timestamp: float
    landmarks: dict    # name -> (x, y, z, visibility)
    raw: object        # mediapipe NormalizedLandmarkList, for visualisation


@dataclass
class FeatureVector:
    """
    Engineered features for one frame. 32 total.

    ── Original 16 ──────────────────────────────────────────────────────────
     0  left_knee_angle
     1  right_knee_angle
     2  left_hip_angle
     3  right_hip_angle
     4  left_ankle_vel_x
     5  left_ankle_vel_y
     6  right_ankle_vel_x
     7  right_ankle_vel_y
     8  body_facing              degrees; 0=right, 90=up
     9  hip_displacement_x
    10  hip_displacement_y
    11  left_swing_amplitude     ankle-to-hip distance
    12  right_swing_amplitude
    13  left_swing_velocity      d(amplitude)/dt
    14  right_swing_velocity
    15  mean_visibility

    ── SHOOT vs PASS (6) ────────────────────────────────────────────────────
    16  left_foot_forward_ratio  dot(ankle_vel, body_fwd) / |ankle_vel| in [-1,1]
    17  right_foot_forward_ratio shoot≈1.0 (forward), pass≈0 (lateral)
    18  left_foot_lateral_offset signed lateral distance of ankle from hip centreline
    19  right_foot_lateral_offset pass has larger lateral offset
    20  left_knee_extension_rate  d(knee_angle)/dt — shoot has a sharp positive spike
    21  right_knee_extension_rate

    ── IDLE detection (4) ───────────────────────────────────────────────────
    22  total_kinetic_energy     sum of |velocity|² across tracked joints; ~0 when idle
    23  hip_stability            variance of hip centroid over recent frames; ~0 when idle
    24  moving_joint_count       fraction of joints moving above threshold [0,1]
    25  upper_body_motion        shoulder midpoint velocity magnitude

    ── Body lean / MOVE direction (6) ───────────────────────────────────────
    26  body_lean_forward        (shoulder_mid.x - hip_mid.x) / hip_width — lean fwd/back
    27  body_lean_lateral        (shoulder_mid.y - hip_mid.y) / hip_width — lean left/right
    28  planted_foot             0=left planted, 1=right planted, 0.5=ambiguous
    29  left_ankle_accel         |d²(left_ankle)/dt²| — shoot has sharp accel spike
    30  right_ankle_accel
    31  stance_width             ankle distance / hip width — wide for defend/idle
    """

    # Original 16
    left_knee_angle: float        = 0.0
    right_knee_angle: float       = 0.0
    left_hip_angle: float         = 0.0
    right_hip_angle: float        = 0.0
    left_ankle_vel: np.ndarray    = field(default_factory=lambda: np.zeros(2))
    right_ankle_vel: np.ndarray   = field(default_factory=lambda: np.zeros(2))
    body_facing: float            = 0.0
    hip_displacement: np.ndarray  = field(default_factory=lambda: np.zeros(2))
    left_swing_amplitude: float   = 0.0
    right_swing_amplitude: float  = 0.0
    left_swing_velocity: float    = 0.0
    right_swing_velocity: float   = 0.0
    mean_visibility: float        = 0.0

    # SHOOT vs PASS
    left_foot_forward_ratio: float   = 0.0
    right_foot_forward_ratio: float  = 0.0
    left_foot_lateral_offset: float  = 0.0
    right_foot_lateral_offset: float = 0.0
    left_knee_extension_rate: float  = 0.0
    right_knee_extension_rate: float = 0.0

    # IDLE detection
    total_kinetic_energy: float  = 0.0
    hip_stability: float         = 0.0
    moving_joint_count: float    = 0.0
    upper_body_motion: float     = 0.0

    # Body lean / MOVE direction
    body_lean_forward: float  = 0.0
    body_lean_lateral: float  = 0.0
    planted_foot: float       = 0.5
    left_ankle_accel: float   = 0.0
    right_ankle_accel: float  = 0.0
    stance_width: float       = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            # Original 16
            self.left_knee_angle,       self.right_knee_angle,
            self.left_hip_angle,        self.right_hip_angle,
            *self.left_ankle_vel,       *self.right_ankle_vel,
            self.body_facing,
            *self.hip_displacement,
            self.left_swing_amplitude,  self.right_swing_amplitude,
            self.left_swing_velocity,   self.right_swing_velocity,
            self.mean_visibility,
            # SHOOT vs PASS (6)
            self.left_foot_forward_ratio,   self.right_foot_forward_ratio,
            self.left_foot_lateral_offset,  self.right_foot_lateral_offset,
            self.left_knee_extension_rate,  self.right_knee_extension_rate,
            # IDLE (4)
            self.total_kinetic_energy,  self.hip_stability,
            self.moving_joint_count,    self.upper_body_motion,
            # Lean / MOVE (6)
            self.body_lean_forward,     self.body_lean_lateral,
            self.planted_foot,
            self.left_ankle_accel,      self.right_ankle_accel,
            self.stance_width,
        ], dtype=np.float32)  # shape: (32,)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, formed by a-b-c (degrees)."""
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _lm_xy(landmarks: dict, name: str) -> np.ndarray:
    return np.array(landmarks[name][:2])


def _vel_mag(vel: np.ndarray) -> float:
    return float(np.linalg.norm(vel))


# ── EMA Smoother ──────────────────────────────────────────────────────────────
class LandmarkSmoother:
    """
    EMA per landmark. alpha close to 1 = fast response; close to 0 = heavy smoothing.
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha   = alpha
        self._state: dict = {}

    def smooth(self, landmarks: dict) -> dict:
        out = {}
        for k, v in landmarks.items():
            v = np.array(v)
            if k not in self._state:
                self._state[k] = v
            self._state[k] = self.alpha * v + (1 - self.alpha) * self._state[k]
            out[k] = tuple(self._state[k])
        return out


# ── Feature engineer ──────────────────────────────────────────────────────────
class FeatureEngineer:
    """
    Converts a sequence of PoseFrames into FeatureVectors.
    Keeps a rolling history to compute velocities, accelerations,
    and stability metrics.
    """
    def __init__(self, history: int = 10):
        self._hist: deque = deque(maxlen=history)
        # Rolling hip centroids for stability computation
        self._hip_centroids: deque = deque(maxlen=history)

    def update(self, frame: PoseFrame) -> FeatureVector:
        self._hist.append(frame)
        lm = frame.landmarks
        fv = FeatureVector()

        # ── Convenience positions ─────────────────────────────────────────────
        l_hip   = _lm_xy(lm, "left_hip")
        r_hip   = _lm_xy(lm, "right_hip")
        l_knee  = _lm_xy(lm, "left_knee")
        r_knee  = _lm_xy(lm, "right_knee")
        l_ankle = _lm_xy(lm, "left_ankle")
        r_ankle = _lm_xy(lm, "right_ankle")
        l_shldr = _lm_xy(lm, "left_shoulder")
        r_shldr = _lm_xy(lm, "right_shoulder")

        hip_mid    = (l_hip + r_hip) / 2
        shldr_mid  = (l_shldr + r_shldr) / 2
        hip_width  = float(np.linalg.norm(l_hip - r_hip)) + 1e-6

        self._hip_centroids.append(hip_mid.copy())

        # ── Original features ─────────────────────────────────────────────────

        fv.left_knee_angle  = _angle(l_hip,   l_knee, l_ankle)
        fv.right_knee_angle = _angle(r_hip,   r_knee, r_ankle)
        fv.left_hip_angle   = _angle(l_shldr, l_hip,  l_knee)
        fv.right_hip_angle  = _angle(r_shldr, r_hip,  r_knee)

        shoulder_vec   = l_shldr - r_shldr
        fv.body_facing = float(np.degrees(
            np.arctan2(-shoulder_vec[1], shoulder_vec[0])))

        fv.left_swing_amplitude  = float(np.linalg.norm(l_ankle - l_hip))
        fv.right_swing_amplitude = float(np.linalg.norm(r_ankle - r_hip))

        fv.mean_visibility = float(np.mean([v[3] for v in lm.values()]))

        # ── Features needing at least 2 frames ───────────────────────────────
        if len(self._hist) >= 2:
            prev = self._hist[-2].landmarks
            dt   = max(frame.timestamp - self._hist[-2].timestamp, 1e-6)

            pl_ankle = _lm_xy(prev, "left_ankle")
            pr_ankle = _lm_xy(prev, "right_ankle")
            pl_shldr = _lm_xy(prev, "left_shoulder")
            pr_shldr = _lm_xy(prev, "right_shoulder")
            ph_l     = _lm_xy(prev, "left_hip")
            ph_r     = _lm_xy(prev, "right_hip")

            fv.left_ankle_vel  = (l_ankle - pl_ankle) / dt
            fv.right_ankle_vel = (r_ankle - pr_ankle) / dt

            hip_prev           = (ph_l + ph_r) / 2
            fv.hip_displacement = (hip_mid - hip_prev) / dt

            prev_la = float(np.linalg.norm(pl_ankle - _lm_xy(prev, "left_hip")))
            prev_ra = float(np.linalg.norm(pr_ankle - _lm_xy(prev, "right_hip")))
            fv.left_swing_velocity  = (fv.left_swing_amplitude  - prev_la) / dt
            fv.right_swing_velocity = (fv.right_swing_amplitude - prev_ra) / dt

            # ── SHOOT vs PASS: forward ratio ──────────────────────────────────
            facing_rad = float(np.radians(fv.body_facing))
            fwd = np.array([np.cos(facing_rad), -np.sin(facing_rad)])

            for ankle_vel, attr in [
                (fv.left_ankle_vel,  "_left_fwd_ratio_tmp"),
                (fv.right_ankle_vel, "_right_fwd_ratio_tmp"),
            ]:
                vel_mag = _vel_mag(ankle_vel)
                ratio   = float(np.dot(ankle_vel / vel_mag, fwd)) if vel_mag > 1e-3 else 0.0
                setattr(fv, attr, ratio) if hasattr(fv, attr) else None

            # ── SHOOT vs PASS: swing asymmetry (replaces raw forward ratio) ───
            # Diagnostic showed swing_amp is top feature but forward_ratio fails.
            # Root cause: velocity is noisy on 1-sec clips.
            # Asymmetry is robust: shoot = one leg far out, other planted.
            # Pass = both legs stay close to neutral → small asymmetry.
            fv.left_foot_forward_ratio  = (
                fv.left_swing_amplitude - fv.right_swing_amplitude
            ) / (hip_width + 1e-6)
            fv.right_foot_forward_ratio = (
                fv.right_swing_amplitude - fv.left_swing_amplitude
            ) / (hip_width + 1e-6)

            # ── SHOOT vs PASS: lateral offset from hip centreline ─────────────
            # Project ankle position onto lateral axis (perpendicular to forward)
            # Positive = left side, negative = right side
            lat = np.array([-fwd[1], fwd[0]])   # perpendicular to fwd
            fv.left_foot_lateral_offset  = float(
                np.dot(l_ankle - hip_mid, lat) / hip_width)
            fv.right_foot_lateral_offset = float(
                np.dot(r_ankle - hip_mid, lat) / hip_width)

            # ── SHOOT vs PASS: knee extension rate ────────────────────────────
            prev_l_knee  = _lm_xy(prev, "left_knee")
            prev_r_knee  = _lm_xy(prev, "right_knee")
            prev_lka = _angle(_lm_xy(prev,"left_hip"),  prev_l_knee, pl_ankle)
            prev_rka = _angle(_lm_xy(prev,"right_hip"), prev_r_knee, pr_ankle)
            fv.left_knee_extension_rate  = (fv.left_knee_angle  - prev_lka) / dt
            fv.right_knee_extension_rate = (fv.right_knee_angle - prev_rka) / dt

            # ── Upper body motion ─────────────────────────────────────────────
            shldr_mid_prev = (pl_shldr + pr_shldr) / 2
            fv.upper_body_motion = float(
                np.linalg.norm((shldr_mid - shldr_mid_prev) / dt))

        # ── Features needing at least 3 frames (acceleration) ────────────────
        if len(self._hist) >= 3:
            prev2 = self._hist[-3].landmarks
            prev1 = self._hist[-2].landmarks
            dt1   = max(self._hist[-2].timestamp - self._hist[-3].timestamp, 1e-6)
            dt2   = max(frame.timestamp          - self._hist[-2].timestamp, 1e-6)

            # Ankle acceleration (second derivative)
            vel1_l = (_lm_xy(prev1,"left_ankle")  - _lm_xy(prev2,"left_ankle"))  / dt1
            vel2_l = (l_ankle                     - _lm_xy(prev1,"left_ankle"))  / dt2
            vel1_r = (_lm_xy(prev1,"right_ankle") - _lm_xy(prev2,"right_ankle")) / dt1
            vel2_r = (r_ankle                     - _lm_xy(prev1,"right_ankle")) / dt2

            fv.left_ankle_accel  = float(np.linalg.norm((vel2_l - vel1_l) / dt2))
            fv.right_ankle_accel = float(np.linalg.norm((vel2_r - vel1_r) / dt2))

        # ── IDLE: total kinetic energy ────────────────────────────────────────
        # Sum of squared velocity magnitudes across all tracked joints
        # Computed directly from vel fields we already have
        ke = (_vel_mag(fv.left_ankle_vel) ** 2 +
              _vel_mag(fv.right_ankle_vel) ** 2 +
              float(fv.upper_body_motion) ** 2 +
              _vel_mag(fv.hip_displacement) ** 2)
        fv.total_kinetic_energy = float(ke)

        # ── IDLE: hip stability ───────────────────────────────────────────────
        # Variance of hip centroid positions over recent frames.
        # Near-zero when standing still.
        if len(self._hip_centroids) >= 3:
            centroids = np.stack(self._hip_centroids)
            fv.hip_stability = float(np.var(centroids))
        else:
            fv.hip_stability = 0.0

        # ── IDLE: moving joint count ──────────────────────────────────────────
        # Fraction of joints whose instantaneous speed exceeds a small threshold.
        # Idle = 0.0, active gesture = > 0.3
        vel_threshold = 5.0   # pixels/second
        joints_moving = sum([
            _vel_mag(fv.left_ankle_vel)  > vel_threshold,
            _vel_mag(fv.right_ankle_vel) > vel_threshold,
            fv.upper_body_motion         > vel_threshold,
            _vel_mag(fv.hip_displacement)> vel_threshold,
        ])
        fv.moving_joint_count = joints_moving / 4.0    # normalise to [0,1]

        # ── Body lean (MOVE + SHOOT) ──────────────────────────────────────────
        # Offset of shoulder midpoint relative to hip midpoint,
        # normalised by hip width. Forward lean = negative x for typical setup.
        lean_vec = shldr_mid - hip_mid
        fv.body_lean_forward = float(lean_vec[0] / hip_width)
        fv.body_lean_lateral = float(lean_vec[1] / hip_width)

        # ── Planted foot ──────────────────────────────────────────────────────
        # The foot with lower velocity is the planted (support) foot.
        # 0 = left planted (right leg kicking), 1 = right planted (left leg kicking)
        l_speed = _vel_mag(fv.left_ankle_vel)
        r_speed = _vel_mag(fv.right_ankle_vel)
        total_speed = l_speed + r_speed
        if total_speed > 1e-3:
            fv.planted_foot = float(r_speed / total_speed)
        else:
            fv.planted_foot = 0.5

        # ── Stance width ─────────────────────────────────────────────────────
        # Ankle distance normalised by hip width.
        # Wide stance (>1.5) → defend or idle.
        # Narrow stance during kick → one leg lifted.
        fv.stance_width = float(np.linalg.norm(l_ankle - r_ankle) / hip_width)

        return fv


# ── Pose extractor (MediaPipe wrapper) ────────────────────────────────────────
class PoseExtractor:
    def __init__(self, model_complexity: int = 1, min_detection_conf: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose    = self.mp_pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=False,
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=0.5,
        )
        self.smoother = LandmarkSmoother(alpha=0.65)

    def extract(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[PoseFrame]:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result  = self.pose.process(img_rgb)

        if not result.pose_landmarks:
            return None

        h, w   = frame_bgr.shape[:2]
        raw_lm = result.pose_landmarks.landmark
        landmarks = {
            name: (
                raw_lm[idx].x * w,
                raw_lm[idx].y * h,
                raw_lm[idx].z * w,
                raw_lm[idx].visibility,
            )
            for name, idx in KEYPOINTS.items()
        }
        landmarks = self.smoother.smooth(landmarks)
        return PoseFrame(timestamp=timestamp, landmarks=landmarks,
                         raw=result.pose_landmarks)

    def close(self):
        self.pose.close()


# ── Latency / FPS benchmarker ─────────────────────────────────────────────────
class Benchmarker:
    """Per-stage timing recorder."""
    def __init__(self, window: int = 60):
        self._window = window
        self.stage_times: dict[str, deque] = {}
        self._t0: dict[str, float] = {}

    def start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str):
        dt = (time.perf_counter() - self._t0[stage]) * 1000
        if stage not in self.stage_times:
            self.stage_times[stage] = deque(maxlen=self._window)
        self.stage_times[stage].append(dt)

    def summary(self) -> dict:
        out = {}
        for stage, times in self.stage_times.items():
            arr = np.array(times)
            out[stage] = {
                "mean_ms": float(np.mean(arr)),
                "p95_ms":  float(np.percentile(arr, 95)),
                "fps":     float(1000 / np.mean(arr)) if np.mean(arr) > 0 else 0,
            }
        return out

    def overlay(self, img: np.ndarray) -> np.ndarray:
        y = 24
        for stage, times in self.stage_times.items():
            if times:
                ms  = np.mean(times)
                fps = 1000 / ms if ms > 0 else 0
                cv2.putText(img, f"{stage}: {ms:.1f}ms  {fps:.0f}fps",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,128), 1)
                y += 20
        return img


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_pipeline(classifier_fn=None, source: int = 0):
    from classifiers.rule_based import RuleBasedClassifier

    cap       = cv2.VideoCapture(source)
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=30)
    bench     = Benchmarker()
    clf       = classifier_fn or RuleBasedClassifier()
    mp_draw   = mp.solutions.drawing_utils

    print("Pipeline running — press Q to quit, B to print benchmark summary.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        t     = time.perf_counter()

        bench.start("pose")
        pose_frame = extractor.extract(frame, t)
        bench.stop("pose")

        if pose_frame:
            bench.start("features")
            fv = engineer.update(pose_frame)
            bench.stop("features")

            bench.start("classify")
            gesture = clf(fv)
            bench.stop("classify")

            mp_draw.draw_landmarks(
                frame, pose_frame.raw,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,100,200), thickness=2),
            )
            cv2.putText(frame, f"Gesture: {gesture}",
                        (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

            # Direction arrow from hip displacement
            disp = fv.hip_displacement
            if np.linalg.norm(disp) > 0.01:
                cx, cy = frame.shape[1]//2, frame.shape[0]//2
                scale  = 200
                ex = int(cx + disp[0] * scale)
                ey = int(cy + disp[1] * scale)
                cv2.arrowedLine(frame, (cx,cy), (ex,ey), (0,255,255), 2)

        bench.overlay(frame)
        cv2.imshow("GestureBall CV Pipeline", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('b'):
            import pprint
            pprint.pprint(bench.summary())
            with open("benchmark_results.json","w") as f:
                json.dump(bench.summary(), f, indent=2)
            print("Saved benchmark_results.json")

    cap.release()
    extractor.close()
    cv2.destroyAllWindows()
    return bench.summary()


if __name__ == "__main__":
    run_pipeline()
