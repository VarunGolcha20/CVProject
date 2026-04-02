"""
classifiers/rule_based.py
=========================
Baseline rule-based classifier.
Addresses reviewer question: How do we distinguish pass vs shoot?

Key insight:
  SHOOT = high swing velocity + large swing amplitude + forward ankle direction
  PASS  = moderate swing velocity + lateral foot displacement
  MOVE  = hip centroid displacement direction → also handles direction question
  DEFEND= wide stance, low centre of mass, arms extended
"""

import numpy as np
from pipeline import FeatureVector


class RuleBasedClassifier:
    """
    Deterministic threshold classifier.
    Deliberately kept as the easy baseline — the reviewer is right that
    this alone is insufficient. Use it as a benchmark floor.
    """

    # Thresholds (tune these from pilot recordings)
    SHOOT_VEL_THRESH  = 0.30   # normalised ankle velocity
    PASS_VEL_THRESH   = 0.12
    MOVE_HIP_THRESH   = 0.005  # hip centroid motion
    DEFEND_KNEE_THRESH= 140    # degrees — wide bent-knee stance

    # Hysteresis / debounce
    COOLDOWN_FRAMES   = 8      # frames before another event is emitted

    def __init__(self):
        self._cooldown = 0
        self._last = "IDLE"

    def __call__(self, fv: FeatureVector) -> str:
        if self._cooldown > 0:
            self._cooldown -= 1
            return self._last

        gesture = self._classify(fv)
        if gesture != "IDLE":
            self._cooldown = self.COOLDOWN_FRAMES
            self._last = gesture
        return gesture

    def _classify(self, fv: FeatureVector) -> str:
        lsv = fv.left_swing_velocity
        rsv = fv.right_swing_velocity
        la  = fv.left_swing_amplitude
        ra  = fv.right_swing_amplitude

        # Low confidence → skip
        if fv.mean_visibility < 0.5:
            return "OCCLUDED"

        # ── SHOOT vs PASS discrimination ──────────────────────────────────────
        # SHOOT: fast swing (high |velocity|), large amplitude (leg fully extends)
        #        foot motion primarily FORWARD (ankle_vel[0] large relative to [1])
        # PASS:  moderate swing, foot motion more LATERAL
        for (sv, av, ankle_vel) in [
            (lsv, la, fv.left_ankle_vel),
            (rsv, ra, fv.right_ankle_vel),
        ]:
            vel_mag = np.linalg.norm(ankle_vel)
            if vel_mag < 1e-6:
                continue
            forward_ratio = abs(ankle_vel[0]) / (vel_mag + 1e-8)   # 1.0 = purely forward
            lateral_ratio = abs(ankle_vel[1]) / (vel_mag + 1e-8)

            if sv > self.SHOOT_VEL_THRESH and av > 0.5 and forward_ratio > 0.65:
                return "SHOOT"

            if sv > self.PASS_VEL_THRESH and lateral_ratio > 0.40:
                return "PASS"

        # ── DEFEND: bent knees, wide stance ──────────────────────────────────
        avg_knee = (fv.left_knee_angle + fv.right_knee_angle) / 2
        if avg_knee < self.DEFEND_KNEE_THRESH and fv.left_swing_amplitude > 0.35:
            return "DEFEND"

        # ── MOVE: hip centroid displacement → gives direction ─────────────────
        disp_mag = np.linalg.norm(fv.hip_displacement)
        if disp_mag > self.MOVE_HIP_THRESH:
            direction = _direction_label(fv.hip_displacement)
            return f"MOVE_{direction}"

        return "IDLE"


def _direction_label(disp: np.ndarray) -> str:
    """
    Reviewer Q: how do you decide direction from running?
    Answer: track hip centroid displacement vector between frames.
    Map the dominant axis to a cardinal direction.
    """
    x, y = disp
    if abs(x) > abs(y):
        return "RIGHT" if x > 0 else "LEFT"
    else:
        return "DOWN"  if y > 0 else "UP"


# ── Failure analysis ──────────────────────────────────────────────────────────
FAILURE_MODES = {
    "OCCLUDED":         "Landmark visibility < 0.5 — body partially off-screen or occluded.",
    "SHOOT_FALSE_POS":  "Walking briskly can mimic shoot velocity — need amplitude gate.",
    "PASS_SHOOT_CONF":  "Lateral kick ≠ pass; needs foot orientation, not just velocity direction.",
    "DIRECTION_AMBIG":  "If player stands still (no hip motion) but turns, direction misclassified.",
    "LIGHTING":         "MediaPipe landmark confidence drops in poor lighting — smoother can lag.",
}
