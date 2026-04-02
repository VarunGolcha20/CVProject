"""
GestureBall CV Pipeline
=======================
Core pose extraction, feature engineering, and benchmarking harness.
Run this file directly to test the pipeline on your webcam.
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
    "left_hip":   LM.LEFT_HIP,
    "right_hip":  LM.RIGHT_HIP,
    "left_knee":  LM.LEFT_KNEE,
    "right_knee": LM.RIGHT_KNEE,
    "left_ankle": LM.LEFT_ANKLE,
    "right_ankle":LM.RIGHT_ANKLE,
    "left_shoulder":  LM.LEFT_SHOULDER,
    "right_shoulder": LM.RIGHT_SHOULDER,
    "left_foot":  LM.LEFT_FOOT_INDEX,
    "right_foot": LM.RIGHT_FOOT_INDEX,
}


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class PoseFrame:
    """One frame's worth of pose data."""
    timestamp: float
    landmarks: dict          # name -> (x, y, z, visibility)
    raw: object              # mediapipe NormalizedLandmarkList, for visualisation


@dataclass
class FeatureVector:
    """Engineered features for one frame."""
    # Angles (degrees)
    left_knee_angle: float   = 0.0
    right_knee_angle: float  = 0.0
    left_hip_angle: float    = 0.0
    right_hip_angle: float   = 0.0
    # Velocities (px/s, normalised to frame size)
    left_ankle_vel: np.ndarray  = field(default_factory=lambda: np.zeros(2))
    right_ankle_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Direction
    body_facing: float       = 0.0   # degrees, 0 = right, 90 = up
    hip_displacement: np.ndarray = field(default_factory=lambda: np.zeros(2))
    # Swing
    left_swing_amplitude: float  = 0.0
    right_swing_amplitude: float = 0.0
    left_swing_velocity: float   = 0.0
    right_swing_velocity: float  = 0.0
    # Confidence
    mean_visibility: float   = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.left_knee_angle, self.right_knee_angle,
            self.left_hip_angle,  self.right_hip_angle,
            *self.left_ankle_vel, *self.right_ankle_vel,
            self.body_facing,
            *self.hip_displacement,
            self.left_swing_amplitude,  self.right_swing_amplitude,
            self.left_swing_velocity,   self.right_swing_velocity,
            self.mean_visibility,
        ], dtype=np.float32)  # shape: (17,)


# ── Helpers ───────────────────────────────────────────────────────────────────
def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at vertex b, formed by a-b-c (degrees)."""
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _lm_xy(landmarks: dict, name: str) -> np.ndarray:
    return np.array(landmarks[name][:2])


# ── Kalman / EMA smoother ─────────────────────────────────────────────────────
class LandmarkSmoother:
    """
    Reviewer note: pose estimation is not always smooth.
    EMA per landmark gives a cheap, lag-free alternative to Kalman.
    alpha close to 1 = fast response; close to 0 = heavy smoothing.
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha
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
    Keeps a rolling window to compute velocities.
    """
    def __init__(self, history: int = 10):
        self._hist: deque = deque(maxlen=history)   # deque of PoseFrames

    def update(self, frame: PoseFrame) -> FeatureVector:
        self._hist.append(frame)
        lm = frame.landmarks
        fv = FeatureVector()

        # ── Knee & hip angles ─────────────────────────────────────────────────
        fv.left_knee_angle  = _angle(_lm_xy(lm,"left_hip"),
                                     _lm_xy(lm,"left_knee"),
                                     _lm_xy(lm,"left_ankle"))
        fv.right_knee_angle = _angle(_lm_xy(lm,"right_hip"),
                                     _lm_xy(lm,"right_knee"),
                                     _lm_xy(lm,"right_ankle"))
        fv.left_hip_angle   = _angle(_lm_xy(lm,"left_shoulder"),
                                     _lm_xy(lm,"left_hip"),
                                     _lm_xy(lm,"left_knee"))
        fv.right_hip_angle  = _angle(_lm_xy(lm,"right_shoulder"),
                                     _lm_xy(lm,"right_hip"),
                                     _lm_xy(lm,"right_knee"))

        # ── Body facing direction ─────────────────────────────────────────────
        # Vector from right shoulder to left shoulder → facing angle
        ls = _lm_xy(lm,"left_shoulder");  rs = _lm_xy(lm,"right_shoulder")
        shoulder_vec = ls - rs
        fv.body_facing = float(np.degrees(np.arctan2(-shoulder_vec[1], shoulder_vec[0])))

        # ── Velocities (need at least 2 frames) ───────────────────────────────
        if len(self._hist) >= 2:
            prev = self._hist[-2].landmarks
            dt   = frame.timestamp - self._hist[-2].timestamp + 1e-9
            fv.left_ankle_vel  = (_lm_xy(lm,"left_ankle")  - _lm_xy(prev,"left_ankle"))  / dt
            fv.right_ankle_vel = (_lm_xy(lm,"right_ankle") - _lm_xy(prev,"right_ankle")) / dt

            # Hip centroid displacement → movement direction
            hip_now  = (_lm_xy(lm,  "left_hip") + _lm_xy(lm,  "right_hip")) / 2
            hip_prev = (_lm_xy(prev,"left_hip") + _lm_xy(prev,"right_hip")) / 2
            fv.hip_displacement = (hip_now - hip_prev) / dt

        # ── Swing amplitude & velocity ─────────────────────────────────────────
        # Ankle-to-hip distance as proxy for leg extension / kick amplitude
        fv.left_swing_amplitude  = float(np.linalg.norm(
            _lm_xy(lm,"left_ankle") - _lm_xy(lm,"left_hip")))
        fv.right_swing_amplitude = float(np.linalg.norm(
            _lm_xy(lm,"right_ankle") - _lm_xy(lm,"right_hip")))

        if len(self._hist) >= 2:
            prev = self._hist[-2].landmarks
            dt   = frame.timestamp - self._hist[-2].timestamp + 1e-9
            prev_la = float(np.linalg.norm(_lm_xy(prev,"left_ankle")  - _lm_xy(prev,"left_hip")))
            prev_ra = float(np.linalg.norm(_lm_xy(prev,"right_ankle") - _lm_xy(prev,"right_hip")))
            fv.left_swing_velocity  = (fv.left_swing_amplitude  - prev_la) / dt
            fv.right_swing_velocity = (fv.right_swing_amplitude - prev_ra) / dt

        # ── Mean visibility ───────────────────────────────────────────────────
        fv.mean_visibility = float(np.mean([v[3] for v in lm.values()]))

        return fv


# ── Pose extractor (MediaPipe wrapper) ────────────────────────────────────────
class PoseExtractor:
    def __init__(self, model_complexity: int = 1, min_detection_conf: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            smooth_landmarks=False,       # we do our own smoothing
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=0.5,
        )
        self.smoother = LandmarkSmoother(alpha=0.65)

    def extract(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[PoseFrame]:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result = self.pose.process(img_rgb)

        if not result.pose_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
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
        return PoseFrame(timestamp=timestamp, landmarks=landmarks, raw=result.pose_landmarks)

    def close(self):
        self.pose.close()


# ── Latency / FPS benchmarker ─────────────────────────────────────────────────
class Benchmarker:
    """
    Reviewer note: FPS and latency analysis across methods.
    Records per-stage timing so you can identify bottlenecks.
    """
    def __init__(self, window: int = 60):
        self._window = window
        self.stage_times: dict[str, deque] = {}
        self._t0: dict[str, float] = {}

    def start(self, stage: str):
        self._t0[stage] = time.perf_counter()

    def stop(self, stage: str):
        dt = (time.perf_counter() - self._t0[stage]) * 1000  # ms
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
        """Draw live FPS + latency on the frame."""
        y = 24
        for stage, times in self.stage_times.items():
            if times:
                ms = np.mean(times)
                fps = 1000 / ms if ms > 0 else 0
                cv2.putText(img, f"{stage}: {ms:.1f}ms  {fps:.0f}fps",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,128), 1)
                y += 20
        return img


# ── Main loop ─────────────────────────────────────────────────────────────────
def run_pipeline(classifier_fn=None, source: int = 0):
    """
    Args:
        classifier_fn: callable(FeatureVector) -> str  (gesture label)
                       If None, uses rule-based fallback.
        source: cv2 camera index or video path.
    """
    from classifiers.rule_based import RuleBasedClassifier   # lazy import

    cap = mp.solutions.pose.Pose  # unused – just for import check
    cap = cv2.VideoCapture(source)
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=30)
    bench     = Benchmarker()
    clf       = classifier_fn or RuleBasedClassifier()
    mp_draw   = mp.solutions.drawing_utils

    print("Pipeline running – press Q to quit, B to print benchmark summary.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)          # mirror for natural interaction
        t = time.perf_counter()

        # Stage: pose extraction
        bench.start("pose")
        pose_frame = extractor.extract(frame, t)
        bench.stop("pose")

        if pose_frame:
            # Stage: feature engineering
            bench.start("features")
            fv = engineer.update(pose_frame)
            bench.stop("features")

            # Stage: classification
            bench.start("classify")
            gesture = clf(fv)
            bench.stop("classify")

            # Draw skeleton
            mp_draw.draw_landmarks(
                frame, pose_frame.raw,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0,200,100), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0,100,200), thickness=2),
            )
            # Gesture label
            cv2.putText(frame, f"Gesture: {gesture}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            # Direction arrow
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
