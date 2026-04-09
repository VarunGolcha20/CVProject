"""
GestureBall CV Pipeline
=======================
Core pose extraction, feature engineering, and benchmarking harness.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import json

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

FEATURE_DIM = 32   

@dataclass
class PoseFrame:
    timestamp: float
    landmarks: dict   
    raw: object       

@dataclass
class FeatureVector:
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
    left_foot_forward_ratio: float   = 0.0
    right_foot_forward_ratio: float  = 0.0
    left_foot_lateral_offset: float  = 0.0
    right_foot_lateral_offset: float = 0.0
    left_knee_extension_rate: float  = 0.0
    right_knee_extension_rate: float = 0.0
    total_kinetic_energy: float  = 0.0
    hip_stability: float         = 0.0
    moving_joint_count: float    = 0.0
    upper_body_motion: float     = 0.0
    body_lean_forward: float  = 0.0
    body_lean_lateral: float  = 0.0
    planted_foot: float       = 0.5
    left_ankle_accel: float   = 0.0
    right_ankle_accel: float  = 0.0
    stance_width: float       = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([
            self.left_knee_angle,       self.right_knee_angle,
            self.left_hip_angle,        self.right_hip_angle,
            *self.left_ankle_vel,       *self.right_ankle_vel,
            self.body_facing,
            *self.hip_displacement,
            self.left_swing_amplitude,  self.right_swing_amplitude,
            self.left_swing_velocity,   self.right_swing_velocity,
            self.mean_visibility,
            self.left_foot_forward_ratio,   self.right_foot_forward_ratio,
            self.left_foot_lateral_offset,  self.right_foot_lateral_offset,
            self.left_knee_extension_rate,  self.right_knee_extension_rate,
            self.total_kinetic_energy,  self.hip_stability,
            self.moving_joint_count,    self.upper_body_motion,
            self.body_lean_forward,     self.body_lean_lateral,
            self.planted_foot,
            self.left_ankle_accel,      self.right_ankle_accel,
            self.stance_width,
        ], dtype=np.float32)  

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b;  bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def _lm_xy(landmarks: dict, name: str) -> np.ndarray:
    return np.array(landmarks[name][:2])

def _vel_mag(vel: np.ndarray) -> float:
    return float(np.linalg.norm(vel))

class LandmarkSmoother:
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

class FeatureEngineer:
    def __init__(self, history: int = 35):
        self._hist: deque = deque(maxlen=history)
        self._hip_centroids: deque = deque(maxlen=history)

    def update(self, frame: PoseFrame) -> FeatureVector:
        self._hist.append(frame)
        lm = frame.landmarks
        fv = FeatureVector()

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

            facing_rad = float(np.radians(fv.body_facing))
            fwd = np.array([np.cos(facing_rad), -np.sin(facing_rad)])

            for ankle_vel, attr in [
                (fv.left_ankle_vel,  "left_foot_forward_ratio"),
                (fv.right_ankle_vel, "right_foot_forward_ratio"),
            ]:
                vel_mag = _vel_mag(ankle_vel)
                if vel_mag > 1e-3:
                    ratio = float(np.dot(ankle_vel / vel_mag, fwd))
                else:
                    ratio = 0.0
                setattr(fv, attr, ratio)

            lat = np.array([-fwd[1], fwd[0]])   
            fv.left_foot_lateral_offset  = float(
                np.dot(l_ankle - hip_mid, lat) / hip_width)
            fv.right_foot_lateral_offset = float(
                np.dot(r_ankle - hip_mid, lat) / hip_width)

            prev_l_knee  = _lm_xy(prev, "left_knee")
            prev_r_knee  = _lm_xy(prev, "right_knee")
            prev_lka = _angle(_lm_xy(prev,"left_hip"),  prev_l_knee, pl_ankle)
            prev_rka = _angle(_lm_xy(prev,"right_hip"), prev_r_knee, pr_ankle)
            fv.left_knee_extension_rate  = (fv.left_knee_angle  - prev_lka) / dt
            fv.right_knee_extension_rate = (fv.right_knee_angle - prev_rka) / dt

            shldr_mid_prev = (pl_shldr + pr_shldr) / 2
            fv.upper_body_motion = float(
                np.linalg.norm((shldr_mid - shldr_mid_prev) / dt))

        if len(self._hist) >= 3:
            prev2 = self._hist[-3].landmarks
            prev1 = self._hist[-2].landmarks
            dt1   = max(self._hist[-2].timestamp - self._hist[-3].timestamp, 1e-6)
            dt2   = max(frame.timestamp          - self._hist[-2].timestamp, 1e-6)

            vel1_l = (_lm_xy(prev1,"left_ankle")  - _lm_xy(prev2,"left_ankle"))  / dt1
            vel2_l = (l_ankle                     - _lm_xy(prev1,"left_ankle"))  / dt2
            vel1_r = (_lm_xy(prev1,"right_ankle") - _lm_xy(prev2,"right_ankle")) / dt1
            vel2_r = (r_ankle                     - _lm_xy(prev1,"right_ankle")) / dt2

            fv.left_ankle_accel  = float(np.linalg.norm((vel2_l - vel1_l) / dt2))
            fv.right_ankle_accel = float(np.linalg.norm((vel2_r - vel1_r) / dt2))

        ke = (_vel_mag(fv.left_ankle_vel) ** 2 +
              _vel_mag(fv.right_ankle_vel) ** 2 +
              float(fv.upper_body_motion) ** 2 +
              _vel_mag(fv.hip_displacement) ** 2)
        fv.total_kinetic_energy = float(ke)

        if len(self._hip_centroids) >= 3:
            centroids = np.stack(self._hip_centroids)
            fv.hip_stability = float(np.var(centroids))
        else:
            fv.hip_stability = 0.0

        vel_threshold = 0.05   
        joints_moving = sum([
            _vel_mag(fv.left_ankle_vel)  > vel_threshold,
            _vel_mag(fv.right_ankle_vel) > vel_threshold,
            fv.upper_body_motion         > vel_threshold,
            _vel_mag(fv.hip_displacement)> vel_threshold,
        ])
        fv.moving_joint_count = joints_moving / 4.0    

        lean_vec = shldr_mid - hip_mid
        fv.body_lean_forward = float(lean_vec[0] / hip_width)
        fv.body_lean_lateral = float(lean_vec[1] / hip_width)

        l_speed = _vel_mag(fv.left_ankle_vel)
        r_speed = _vel_mag(fv.right_ankle_vel)
        total_speed = l_speed + r_speed
        if total_speed > 1e-3:
            fv.planted_foot = float(r_speed / total_speed)
        else:
            fv.planted_foot = 0.5

        fv.stance_width = float(np.linalg.norm(l_ankle - r_ankle) / hip_width)

        return fv


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

        raw_lm = result.pose_landmarks.landmark
        
        landmarks = {
            name: (
                raw_lm[idx].x, 
                raw_lm[idx].y, 
                raw_lm[idx].z, 
                raw_lm[idx].visibility,
            )
            for name, idx in KEYPOINTS.items()
        }
        landmarks = self.smoother.smooth(landmarks)
        return PoseFrame(timestamp=timestamp, landmarks=landmarks,
                         raw=result.pose_landmarks)

    def close(self):
        self.pose.close()

class Benchmarker:
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


def run_pipeline(classifier_fn=None, source: int = 0):
    from classifiers.mobilenet_classifier import PretrainedImageClassifier

    cap       = cv2.VideoCapture(source)
    extractor = PoseExtractor()
    engineer  = FeatureEngineer(history=35)
    bench     = Benchmarker()
    
    # Defaults to MobileNet if no classifier is passed
    clf       = classifier_fn or PretrainedImageClassifier(model_path="models/mobilenet_gesture.pt")
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
            # Route frame to MobileNet, otherwise pass feature vector
            if type(clf).__name__ == "PretrainedImageClassifier":
                gesture = clf(frame)
            else:
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

            disp = fv.hip_displacement
            if np.linalg.norm(disp) > 0.005:  
                cx, cy = frame.shape[1]//2, frame.shape[0]//2
                scale  = 1000  
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