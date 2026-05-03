"""
error_detection.py — Pure functions for squat error detection.
Extracted for testability. Imported by live_classifier.py.
"""
import numpy as np


# ── Geometry helpers ──────────────────────────────────────────────────────────

def calculate_angle(a, b, c):
    """Angle at point b between vectors ba and bc. Returns degrees. Works in 2D or 3D."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 * n2 < 1e-8:
        return 0.0
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)))


# ── Biomechanical feature extractors ─────────────────────────────────────────

def _knee_valgus(n_func, side):
    """
    Perpendicular displacement of knee from hip→ankle line.
    Positive = knee caving IN (valgus). Negative = bowing out (varus).
    Best from frontal camera; still functional from diagonal.
    """
    hip   = n_func(f'{side}_hip')
    knee  = n_func(f'{side}_knee')
    ankle = n_func(f'{side}_ankle')
    limb_vec  = ankle - hip
    limb_unit = limb_vec / (np.linalg.norm(limb_vec) + 1e-6)
    knee_vec  = knee - hip
    proj      = np.dot(knee_vec, limb_unit)
    perp      = knee_vec - proj * limb_unit
    return -perp[0] if side == 'left' else perp[0]

def _knee_width_ratio(n_func):
    """
    Knee width / ankle width.
    ~1.0 = normal, >1.4 = too wide, <0.7 = caving.
    Scale-invariant ratio.
    """
    knee_w  = abs(n_func('left_knee')[0]  - n_func('right_knee')[0])
    ankle_w = abs(n_func('left_ankle')[0] - n_func('right_ankle')[0])
    return knee_w / (ankle_w + 1e-6)


def _rounded_back(n_func):
    """
    mid_shoulder_y - nose_y (MediaPipe Y grows downward).
    Straight back: nose above shoulders → positive.
    Rounding: value decreases, goes negative when severe.
    Works well from side; usable from front/back.
    """
    nose   = n_func('nose')
    mid_sh = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    return mid_sh[1] - nose[1]


def _spine_flexion(n_func):
    """
    Trunk lean angle from vertical.
    0° = perfectly upright. Grows with any forward/backward lean.
    Best from side view.
    """
    mid_sh  = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    mid_hip = (n_func('left_hip')      + n_func('right_hip'))      / 2
    return np.degrees(np.arctan2(
        abs(mid_sh[0] - mid_hip[0]),
        abs(mid_sh[1] - mid_hip[1]) + 1e-6
    ))


def _lumbar_extension(n_func):
    """
    mid_hip_x - mid_shoulder_x.
    ≈ 0 = neutral. Positive = hips forward (lumbar extension). Negative = sway back.
    abs() in detect_errors catches both directions.
    """
    mid_sh  = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    mid_hip = (n_func('left_hip')      + n_func('right_hip'))      / 2
    return mid_hip[0] - mid_sh[0]


def _trunk_lean(n_func):
    """Trunk lean angle from vertical. Same as spine_flexion; kept named for clarity."""
    mid_sh  = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    mid_hip = (n_func('left_hip')      + n_func('right_hip'))      / 2
    return np.degrees(np.arctan2(
        abs(mid_sh[0] - mid_hip[0]),
        abs(mid_sh[1] - mid_hip[1]) + 1e-6
    ))


def spine_angle_3d(coords):
    """
    3D angle: mid_ear → mid_shoulder → mid_hip.
    Uses all three MediaPipe axes. View-agnostic because depth (z) is included.
    Straight spine  : ≈ 160–180°
    Moderate rounding: 145–160°
    Severe rounding  : < 145°

    Why this works from the front:
    - When you hunch, shoulders roll forward → z_shoulder increases relative to z_hip
    - The bend at the shoulder joint is captured by the 3D angle even without 2D curvature
    """
    mid_ear = (coords['left_ear'] + coords['right_ear']) / 2     # 3-vector
    mid_sh  = (coords['left_shoulder'] + coords['right_shoulder']) / 2
    mid_hip = (coords['left_hip'] + coords['right_hip']) / 2
    return calculate_angle(mid_ear, mid_sh, mid_hip)


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features_for_analysis(lm, get_coords_fn, normalize_fn):
    """
    Biomechanical features for form analysis.
    Separated from the ML classifier features — changes here don't require retraining.

    Args:
        lm: MediaPipe landmark list
        get_coords_fn: callable(lm) → coords dict
        normalize_fn: callable(coords) → (n_func, scale)

    Returns:
        dict of features, or None if landmarks are missing
    """
    c = get_coords_fn(lm)

    required = ['left_hip', 'right_hip', 'left_knee', 'right_knee',
                'left_ankle', 'right_ankle']
    if any(np.isnan(c.get(k, np.array([np.nan]))).any() for k in required):
        return None

    n, _ = normalize_fn(c)

    left_knee_angle  = calculate_angle(n('left_hip'),  n('left_knee'),  n('left_ankle'))
    right_knee_angle = calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))

    return {
        # Knee angles
        'avg_knee_angle':      (left_knee_angle + right_knee_angle) / 2,
        'knee_symmetry':       abs(left_knee_angle - right_knee_angle),
        # Knee alignment
        'knee_valgus_left':    _knee_valgus(n, 'left'),
        'knee_valgus_right':   _knee_valgus(n, 'right'),
        'knee_width_ratio':    _knee_width_ratio(n),
        # Trunk / spine
        'trunk_lean':          _trunk_lean(n),           # ← FIX: was missing
        'rounded_back':        _rounded_back(n),
        'spine_flexion':       _spine_flexion(n),
        'spine_angle_3d':      spine_angle_3d(c),        # ← NEW: view-agnostic
        'lumbar_extension':    _lumbar_extension(n),     # ← FIX: was missing
        # Position
        'hip_shoulder_offset': abs(
            (n('left_hip')[0]  + n('right_hip')[0])  / 2 -
            (n('left_shoulder')[0] + n('right_shoulder')[0]) / 2
        ),
        'stance_ratio': (
            np.linalg.norm(n('left_ankle') - n('right_ankle')) /
            (np.linalg.norm(n('left_shoulder') - n('right_shoulder')) + 1e-6)
        ),
        # Foot contact
        'left_heel_lift':  n('left_heel')[1]  - n('left_foot_index')[1],
        'right_heel_lift': n('right_heel')[1] - n('right_foot_index')[1],
    }


# ── Error detection ───────────────────────────────────────────────────────────

# Absolute thresholds — used as fallback when reference data is absent for a key,
# or as a second independent check. Tuned to fire only on clear violations.
_ABS = {
    # (key, direction, threshold, error_label)
    # direction 'gt_abs' = |val| > threshold; 'gt' = val > threshold; 'lt' = val < threshold
    'spine_angle_3d':   ('lt',    145.0, 'Rounded back'),
    'knee_valgus_left': ('gt',    0.12,  'Knee valgus (left)'),
    'knee_valgus_right':('gt',    0.12,  'Knee valgus (right)'),
    'knee_width_ratio': ('gt',    1.4,   'Knees too wide'),
    'lumbar_extension': ('gt_abs', 0.15, 'Lumbar hyperextension'),
}

HEEL_LIFT_THRESHOLD = -0.07


def detect_errors(analysis_feat, phase, ref_data, prev_knee_angle):
    """
    Compare current frame features against reference stats and absolute thresholds.

    Returns:
        (list[str], float): active error labels, updated knee angle
    """
    errors = []
    new_knee_angle = prev_knee_angle

    if not analysis_feat:
        return errors, new_knee_angle

    current_knee   = analysis_feat.get('avg_knee_angle', 0)
    new_knee_angle = current_knee
    is_stable      = (
        prev_knee_angle is None or
        abs(current_knee - prev_knee_angle) <= 8.0
    )

    stats = ref_data.get(phase, {})

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _add(label):
        if label not in errors:
            errors.append(label)

    def check_ref(key, label, condition):
        """Reference-based check: fires when condition(val, mean, std) is True."""
        if key not in analysis_feat or label in errors:
            return
        val = analysis_feat[key]
        if np.isnan(val):
            return
        if key in stats:
            m, s = stats[key]['mean'], stats[key]['std']
            if condition(val, m, s):
                _add(label)

    def check_abs_threshold(key, label):
        """Absolute threshold fallback. Fires independently of reference stats."""
        if key not in _ABS or key not in analysis_feat or label in errors:
            return
        val = analysis_feat[key]
        if np.isnan(val):
            return
        direction, threshold, _ = _ABS[key]
        if direction == 'lt'     and val < threshold:
            _add(label)
        elif direction == 'gt'   and val > threshold:
            _add(label)
        elif direction == 'gt_abs' and abs(val) > threshold:
            _add(label)

    # ── Heel lift — not gated by stability, fires during movement too ─────────
    if phase == 'squats_down':
        if (analysis_feat.get('left_heel_lift',  0) < HEEL_LIFT_THRESHOLD or
                analysis_feat.get('right_heel_lift', 0) < HEEL_LIFT_THRESHOLD):
            _add('Heels lifting')

    # ── Stability gate: skip positional checks during rapid transitions ───────
    if not is_stable:
        return errors, new_knee_angle

    # ── Phase-specific checks ─────────────────────────────────────────────────
    if phase == 'squats_down':
        check_ref('avg_knee_angle', 'Insufficient depth',
                  lambda v, m, s: v > m + 1.0 * s)
        check_ref('trunk_lean', 'Excessive forward lean',
                  lambda v, m, s: v > m + 1.5 * s)

    elif phase == 'squats_up':
        check_ref('avg_knee_angle', 'Not fully extended',
                  lambda v, m, s: v < m - 1.5 * s)
        check_ref('hip_shoulder_offset', 'Hips shifting',
                  lambda v, m, s: v > m + 1.5 * s)
        check_ref('trunk_lean', 'Leaning at top',
                  lambda v, m, s: v > m + 1.0 * s)

    # ── Checks active in BOTH phases ──────────────────────────────────────────

    # Rounded back — reference then absolute view-agnostic fallback
    check_ref('rounded_back', 'Rounded back',        # ← FIX: was 'spine_angle'
              lambda v, m, s: v < m - 1.5 * s)
    check_abs_threshold('spine_angle_3d', 'Rounded back')  # ← NEW: works from any angle

    # Knee alignment
    check_ref('knee_valgus_left',  'Knee valgus (left)',   # ← FIX: was absent
              lambda v, m, s: v > m + 1.5 * s)
    check_abs_threshold('knee_valgus_left', 'Knee valgus (left)')

    check_ref('knee_valgus_right', 'Knee valgus (right)',  # ← FIX: was absent
              lambda v, m, s: v > m + 1.5 * s)
    check_abs_threshold('knee_valgus_right', 'Knee valgus (right)')

    check_ref('knee_width_ratio', 'Knees too wide',        # ← FIX: was absent
              lambda v, m, s: v > m + 1.5 * s)
    check_abs_threshold('knee_width_ratio', 'Knees too wide')

    # Lumbar
    check_ref('lumbar_extension', 'Lumbar hyperextension', # ← FIX: was absent
              lambda v, m, s: abs(v) > abs(m) + 1.5 * s)
    check_abs_threshold('lumbar_extension', 'Lumbar hyperextension')

    # Stance and symmetry
    check_ref('stance_ratio', 'Stance width off',
              lambda v, m, s: abs(v - m) > 1.5 * s)
    check_ref('knee_symmetry', 'Leg asymmetry',
              lambda v, m, s: v > m + 1.5 * s)

    return errors, new_knee_angle
