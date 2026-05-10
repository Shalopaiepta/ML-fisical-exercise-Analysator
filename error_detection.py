"""
error_detection.py - pure functions for exercise error detection.

The detector is reference-first: values are compared with the user's CSV
reference stats, while a few biomechanical guardrails keep noisy CSV columns
from becoming either too sensitive or impossible to trigger.
"""

import numpy as np


_THRESHOLDS = {
    "max_stable_knee_delta": 8.0,
    "depth_median_margin": 18.0,
    "depth_abs_cap": 145.0,
    "rounded_back_abs": 0.05,
    "spine_angle_3d_abs": 145.0,
    "knee_valgus_floor": 0.10,
    "knee_valgus_cap": 0.16,
    "knee_width_floor": 1.35,
    "knee_width_cap": 1.60,
    "knee_caving_floor": 0.65,
    "knee_caving_cap": 0.78,
    "knee_caving_ratio_factor": 0.80,
    "knee_symmetry_floor": 15.0,
    "knee_symmetry_cap": 20.0,
    "pushup_hip_sag_default": 0.20,
    "pushup_hip_sag_floor": 0.12,
    "pushup_hip_sag_cap": 0.30,
    "pushup_hand_width_wide_default": 2.00,
    "pushup_hand_width_wide_floor": 1.70,
    "pushup_hand_width_wide_cap": 2.40,
    "pushup_hand_width_narrow_default": 0.85,
    "pushup_hand_width_narrow_floor": 0.55,
    "pushup_hand_width_narrow_cap": 1.00,
    "pushup_foot_width_wide_default": 1.10,
    "pushup_foot_width_wide_floor": 0.90,
    "pushup_foot_width_wide_cap": 1.60,
    "pushup_foot_width_narrow_default": 0.20,
    "pushup_foot_width_narrow_floor": 0.08,
    "pushup_foot_width_narrow_cap": 0.35,
    "max_stable_elbow_delta": 30.0,
}


def calculate_angle(a, b, c):
    """Angle at point b between vectors ba and bc. Returns degrees."""
    ba = np.array(a, dtype=float) - np.array(b, dtype=float)
    bc = np.array(c, dtype=float) - np.array(b, dtype=float)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if not np.isfinite(n1 * n2) or n1 * n2 < 1e-8:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0))))


def _num(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _ref_limit(stats, key, multiplier=1.5, default=None):
    if key not in stats:
        return default
    mean = _num(stats[key].get("mean"))
    std = _num(stats[key].get("std"))
    if mean is None or std is None:
        return default
    return mean + multiplier * std


def _ref_low_limit(stats, key, multiplier=1.5, default=None):
    if key not in stats:
        return default
    mean = _num(stats[key].get("mean"))
    std = _num(stats[key].get("std"))
    if mean is None or std is None:
        return default
    return mean - multiplier * std


def _ref_stat(stats, key, *names):
    if key not in stats:
        return None
    for name in names:
        value = _num(stats[key].get(name))
        if value is not None:
            return value
    return None


def _clamp(value, low, high):
    if value is None:
        return None
    return max(low, min(high, value))


def _smallest_number(*values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return min(values) if values else None


def _knee_valgus(n_func, side):
    """
    Perpendicular displacement of knee from hip-to-ankle line.

    Positive means the knee moved inward, toward the pelvis center. This keeps the
    sign stable even when the camera feed is mirrored.
    """
    hip = n_func(f"{side}_hip")
    knee = n_func(f"{side}_knee")
    ankle = n_func(f"{side}_ankle")
    limb_vec = ankle - hip
    limb_len = np.linalg.norm(limb_vec)
    if not np.isfinite(limb_len) or limb_len < 1e-6:
        return np.nan

    limb_unit = limb_vec / limb_len
    knee_vec = knee - hip
    proj = np.dot(knee_vec, limb_unit)
    perp = knee_vec - proj * limb_unit
    direction_to_center = -np.sign(hip[0])
    if direction_to_center == 0:
        return np.nan
    return float(perp[0] * direction_to_center)


def _knee_width_ratio(n_func):
    """
    Knee width / ankle width.

    Values near 1.0 mean knees are roughly over the feet. When ankle width is
    nearly zero the view is too side-on for this metric, so return nan.
    """
    knee_w = abs(n_func("left_knee")[0] - n_func("right_knee")[0])
    ankle_w = abs(n_func("left_ankle")[0] - n_func("right_ankle")[0])
    if not np.isfinite(knee_w + ankle_w) or ankle_w < 0.10:
        return np.nan
    return float(knee_w / ankle_w)


def _horizontal_width_ratio(n_func, left_key, right_key, base_left_key, base_right_key):
    width = abs(n_func(left_key)[0] - n_func(right_key)[0])
    base_width = abs(n_func(base_left_key)[0] - n_func(base_right_key)[0])
    if not np.isfinite(width + base_width) or base_width < 0.05:
        return np.nan
    return float(width / base_width)


def _pushup_hip_sag(n_func):
    """
    Positive value means the pelvis is lower than the shoulder-to-ankle body line.

    MediaPipe y grows downward. In a good plank, mid hip lies close to the line
    between mid shoulder and mid ankle; sagging hips/belly make it fall below it.
    """
    mid_sh = (n_func("left_shoulder") + n_func("right_shoulder")) / 2
    mid_hip = (n_func("left_hip") + n_func("right_hip")) / 2
    mid_ankle = (n_func("left_ankle") + n_func("right_ankle")) / 2

    body = mid_ankle - mid_sh
    body_len_sq = float(np.dot(body, body))
    if not np.isfinite(body_len_sq) or body_len_sq < 1e-8:
        return np.nan

    hip_vec = mid_hip - mid_sh
    t = float(np.clip(np.dot(hip_vec, body) / body_len_sq, 0.0, 1.0))
    expected_hip = mid_sh + t * body
    return float(mid_hip[1] - expected_hip[1])


def _rounded_back(n_func):
    nose = n_func("nose")
    mid_sh = (n_func("left_shoulder") + n_func("right_shoulder")) / 2
    return float(mid_sh[1] - nose[1])


def _spine_flexion(n_func):
    mid_sh = (n_func("left_shoulder") + n_func("right_shoulder")) / 2
    mid_hip = (n_func("left_hip") + n_func("right_hip")) / 2
    return float(np.degrees(np.arctan2(abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1]) + 1e-6)))


def _lumbar_extension(n_func):
    mid_sh = (n_func("left_shoulder") + n_func("right_shoulder")) / 2
    mid_hip = (n_func("left_hip") + n_func("right_hip")) / 2
    return float(mid_hip[0] - mid_sh[0])


def _trunk_lean(n_func):
    return _spine_flexion(n_func)


def spine_angle_3d(coords):
    """3D angle: mid_ear -> mid_shoulder -> mid_hip."""
    mid_ear = (coords["left_ear"] + coords["right_ear"]) / 2
    mid_sh = (coords["left_shoulder"] + coords["right_shoulder"]) / 2
    mid_hip = (coords["left_hip"] + coords["right_hip"]) / 2
    return calculate_angle(mid_ear, mid_sh, mid_hip)


def extract_features_for_analysis(lm, get_coords_fn, normalize_fn):
    """Extract analysis-only biomechanical features."""
    coords = get_coords_fn(lm)
    required = [
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ]
    if any(np.isnan(coords.get(key, np.array([np.nan]))).any() for key in required):
        return None

    n_func, _ = normalize_fn(coords)
    left_knee_angle = calculate_angle(n_func("left_hip"), n_func("left_knee"), n_func("left_ankle"))
    right_knee_angle = calculate_angle(n_func("right_hip"), n_func("right_knee"), n_func("right_ankle"))

    left_elbow_angle = calculate_angle(n_func("left_shoulder"), n_func("left_elbow"), n_func("left_wrist"))
    right_elbow_angle = calculate_angle(n_func("right_shoulder"), n_func("right_elbow"), n_func("right_wrist"))

    features = {
        "avg_knee_angle": (left_knee_angle + right_knee_angle) / 2,
        "knee_symmetry": abs(left_knee_angle - right_knee_angle),
        "avg_elbow_angle": (left_elbow_angle + right_elbow_angle) / 2,
        "elbow_symmetry": abs(left_elbow_angle - right_elbow_angle),
        "knee_valgus_left": _knee_valgus(n_func, "left"),
        "knee_valgus_right": _knee_valgus(n_func, "right"),
        "knee_width_ratio": _knee_width_ratio(n_func),
        "hand_width_ratio": _horizontal_width_ratio(
            n_func, "left_wrist", "right_wrist", "left_shoulder", "right_shoulder"
        ),
        "foot_width_ratio": _horizontal_width_ratio(
            n_func, "left_ankle", "right_ankle", "left_shoulder", "right_shoulder"
        ),
        "hip_sag": _pushup_hip_sag(n_func),
        "trunk_lean": _trunk_lean(n_func),
        "rounded_back": _rounded_back(n_func),
        "spine_flexion": _spine_flexion(n_func),
        "spine_angle_3d": spine_angle_3d(coords),
        "lumbar_extension": _lumbar_extension(n_func),
        "hip_shoulder_offset": abs(
            (n_func("left_hip")[0] + n_func("right_hip")[0]) / 2
            - (n_func("left_shoulder")[0] + n_func("right_shoulder")[0]) / 2
        ),
        "stance_ratio": (
            np.linalg.norm(n_func("left_ankle") - n_func("right_ankle"))
            / (np.linalg.norm(n_func("left_shoulder") - n_func("right_shoulder")) + 1e-6)
        ),
    }

    for side in ("left", "right"):
        heel = f"{side}_heel"
        foot = f"{side}_foot_index"
        key = f"{side}_heel_lift"
        if heel in coords and foot in coords and not (np.isnan(coords[heel]).any() or np.isnan(coords[foot]).any()):
            features[key] = n_func(heel)[1] - n_func(foot)[1]
        else:
            features[key] = np.nan

    return features


def detect_errors(analysis_feat, phase, ref_data, prev_knee_angle):
    """
    Compare current frame features against phase CSV stats.

    Returns:
        tuple[list[str], float]: active errors and updated average knee angle.
    """
    errors = []
    new_knee_angle = prev_knee_angle

    if not analysis_feat:
        return errors, new_knee_angle

    def add(label):
        if label not in errors:
            errors.append(label)

    def value(key):
        return _num(analysis_feat.get(key))

    primary_angle_key = "avg_elbow_angle" if phase.startswith("pushups") else "avg_knee_angle"
    current_primary_angle = value(primary_angle_key)
    if current_primary_angle is None:
        return errors, new_knee_angle

    new_knee_angle = current_primary_angle
    new_knee_angle = current_primary_angle
    delta_threshold = (
        _THRESHOLDS["max_stable_elbow_delta"] if phase.startswith("pushups")
        else _THRESHOLDS["max_stable_knee_delta"]
    )
    if (
        prev_knee_angle is not None
        and abs(current_primary_angle - prev_knee_angle) > delta_threshold
    ):
        return errors, new_knee_angle

    stats = ref_data.get(phase, {}) if isinstance(ref_data, dict) else {}

    if phase == "squats_down":
        current_knee = value("avg_knee_angle")
        depth_ref_limit = _ref_limit(stats, "avg_knee_angle", multiplier=1.0)
        depth_median = _ref_stat(stats, "avg_knee_angle", "median", "q50")
        depth_median_limit = (
            depth_median + _THRESHOLDS["depth_median_margin"]
            if depth_median is not None
            else None
        )
        depth_limit = _smallest_number(
            depth_ref_limit,
            depth_median_limit,
            _THRESHOLDS["depth_abs_cap"],
        )
        if depth_limit is not None and current_knee is not None and current_knee > depth_limit:
            add("Insufficient depth")

        knee_width = value("knee_width_ratio")
        raw_width_limit = _ref_limit(stats, "knee_width_ratio", multiplier=2.0)
        width_limit = _clamp(
            raw_width_limit,
            _THRESHOLDS["knee_width_floor"],
            _THRESHOLDS["knee_width_cap"],
        )
        if knee_width is not None and width_limit is not None and knee_width > width_limit:
            add("Knees too wide")

        knee_width_mid = _ref_stat(stats, "knee_width_ratio", "median", "q50", "mean")
        raw_caving_limit = (
            knee_width_mid * _THRESHOLDS["knee_caving_ratio_factor"]
            if knee_width_mid is not None
            else _ref_low_limit(stats, "knee_width_ratio", multiplier=0.5)
        )
        caving_limit = _clamp(
            raw_caving_limit,
            _THRESHOLDS["knee_caving_floor"],
            _THRESHOLDS["knee_caving_cap"],
        )
        if knee_width is not None and caving_limit is not None and knee_width < caving_limit:
            add("Knee valgus (left)")
            add("Knee valgus (right)")

    elif phase == "squats_up":
        extension_limit = _ref_low_limit(stats, "avg_knee_angle", multiplier=1.5)
        current_knee = value("avg_knee_angle")
        if extension_limit is not None and current_knee is not None and current_knee < extension_limit:
            add("Not fully extended")

    elif phase.startswith("pushups"):
        hip_sag = value("hip_sag")
        hip_sag_limit = _ref_limit(
            stats,
            "hip_sag",
            multiplier=1.5,
            default=_THRESHOLDS["pushup_hip_sag_default"],
        )
        hip_sag_limit = _clamp(
            hip_sag_limit,
            _THRESHOLDS["pushup_hip_sag_floor"],
            _THRESHOLDS["pushup_hip_sag_cap"],
        )
        if hip_sag is not None and hip_sag_limit is not None and hip_sag > hip_sag_limit:
            add("Hip sagging")

        hand_width = value("hand_width_ratio")
        hand_wide_limit = _ref_limit(
            stats,
            "hand_width_ratio",
            multiplier=1.5,
            default=_THRESHOLDS["pushup_hand_width_wide_default"],
        )
        hand_wide_limit = _clamp(
            hand_wide_limit,
            _THRESHOLDS["pushup_hand_width_wide_floor"],
            _THRESHOLDS["pushup_hand_width_wide_cap"],
        )
        if hand_width is not None and hand_wide_limit is not None and hand_width > hand_wide_limit:
            add("Hands too wide")

        hand_narrow_limit = _ref_low_limit(
            stats,
            "hand_width_ratio",
            multiplier=1.5,
            default=_THRESHOLDS["pushup_hand_width_narrow_default"],
        )
        hand_narrow_limit = _clamp(
            hand_narrow_limit,
            _THRESHOLDS["pushup_hand_width_narrow_floor"],
            _THRESHOLDS["pushup_hand_width_narrow_cap"],
        )
        if hand_width is not None and hand_narrow_limit is not None and hand_width < hand_narrow_limit:
            add("Hands too narrow")

        foot_width = value("foot_width_ratio")
        foot_wide_limit = _ref_limit(
            stats,
            "foot_width_ratio",
            multiplier=1.5,
            default=_THRESHOLDS["pushup_foot_width_wide_default"],
        )
        foot_wide_limit = _clamp(
            foot_wide_limit,
            _THRESHOLDS["pushup_foot_width_wide_floor"],
            _THRESHOLDS["pushup_foot_width_wide_cap"],
        )
        if foot_width is not None and foot_wide_limit is not None and foot_width > foot_wide_limit:
            add("Feet too wide")

        foot_narrow_limit = _ref_low_limit(
            stats,
            "foot_width_ratio",
            multiplier=1.5,
            default=_THRESHOLDS["pushup_foot_width_narrow_default"],
        )
        foot_narrow_limit = _clamp(
            foot_narrow_limit,
            _THRESHOLDS["pushup_foot_width_narrow_floor"],
            _THRESHOLDS["pushup_foot_width_narrow_cap"],
        )
        if foot_width is not None and foot_narrow_limit is not None and foot_width < foot_narrow_limit:
            add("Feet too narrow")

        return errors, new_knee_angle

    rounded_back = value("rounded_back")
    rounded_limit = _ref_low_limit(stats, "rounded_back", multiplier=1.5)
    if rounded_limit is None:
        rounded_limit = _THRESHOLDS["rounded_back_abs"]
    else:
        rounded_limit = max(rounded_limit, _THRESHOLDS["rounded_back_abs"])

    spine_angle = value("spine_angle_3d")
    if (
        rounded_back is not None
        and rounded_back < rounded_limit
    ) or (
        spine_angle is not None
        and spine_angle < _THRESHOLDS["spine_angle_3d_abs"]
    ):
        add("Rounded back")

    for side, label in (
        ("left", "Knee valgus (left)"),
        ("right", "Knee valgus (right)"),
    ):
        valgus = value(f"knee_valgus_{side}")
        raw_limit = _ref_limit(stats, f"knee_valgus_{side}", multiplier=1.5)
        if raw_limit is not None:
            raw_limit = abs(raw_limit)
        valgus_limit = _clamp(
            raw_limit,
            _THRESHOLDS["knee_valgus_floor"],
            _THRESHOLDS["knee_valgus_cap"],
        )
        if valgus is not None and valgus_limit is not None and valgus > valgus_limit:
            add(label)

    knee_symmetry = value("knee_symmetry")
    raw_symmetry_limit = _ref_limit(stats, "knee_symmetry", multiplier=1.5)
    symmetry_limit = _clamp(
        raw_symmetry_limit,
        _THRESHOLDS["knee_symmetry_floor"],
        _THRESHOLDS["knee_symmetry_cap"],
    )
    if knee_symmetry is not None and symmetry_limit is not None and knee_symmetry > symmetry_limit:
        add("Leg asymmetry")

    hip_shoulder_offset = value("hip_shoulder_offset")
    hip_shift_limit = _ref_limit(stats, "hip_shoulder_offset", multiplier=1.5)
    if hip_shoulder_offset is not None and hip_shift_limit is not None and hip_shoulder_offset > hip_shift_limit:
        add("Hips shifting")

    stance_ratio = value("stance_ratio")
    stance_high = _ref_limit(stats, "stance_ratio", multiplier=1.5)
    stance_low = _ref_low_limit(stats, "stance_ratio", multiplier=1.5)
    if stance_ratio is not None and stance_high is not None and stance_low is not None:
        if stance_ratio > stance_high or stance_ratio < stance_low:
            add("Stance width off")

    return errors, new_knee_angle
