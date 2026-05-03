"""
TDD test suite for squat error detection.

Run with:
    pytest tests/test_error_detection.py -v
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from error_detection import detect_errors, extract_features_for_analysis, _knee_valgus, _rounded_back, spine_angle_3d, _lumbar_extension, _knee_width_ratio

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_ref_stats(overrides=None):
    """
    Minimal reference stats that mimic a correct squat session.
    Values chosen so absolute thresholds and reference thresholds are consistent.
    """
    defaults = {
        "squats_down": {
            "avg_knee_angle":      {"mean": 80.0,  "std": 8.0},
            "trunk_lean":          {"mean": 15.0,  "std": 5.0},
            "stance_ratio":        {"mean": 1.2,   "std": 0.1},
            "knee_symmetry":       {"mean": 3.0,   "std": 3.0},
            "hip_shoulder_offset": {"mean": 0.05,  "std": 0.05},
            "rounded_back":        {"mean": 0.35,  "std": 0.05},
            "spine_flexion":       {"mean": 15.0,  "std": 5.0},
            "lumbar_extension":    {"mean": 0.0,   "std": 0.05},
            "knee_valgus_left":    {"mean": 0.0,   "std": 0.04},
            "knee_valgus_right":   {"mean": 0.0,   "std": 0.04},
            "knee_width_ratio":    {"mean": 1.0,   "std": 0.1},
            "left_heel_lift":      {"mean": -0.04, "std": 0.02},
            "right_heel_lift":     {"mean": -0.04, "std": 0.02},
        },
        "squats_up": {
            "avg_knee_angle":      {"mean": 170.0, "std": 6.0},
            "trunk_lean":          {"mean": 5.0,   "std": 5.0},
            "stance_ratio":        {"mean": 1.2,   "std": 0.1},
            "knee_symmetry":       {"mean": 3.0,   "std": 3.0},
            "hip_shoulder_offset": {"mean": 0.04,  "std": 0.04},
            "rounded_back":        {"mean": 0.35,  "std": 0.05},
            "spine_flexion":       {"mean": 5.0,   "std": 5.0},
            "lumbar_extension":    {"mean": 0.0,   "std": 0.05},
            "knee_valgus_left":    {"mean": 0.0,   "std": 0.04},
            "knee_valgus_right":   {"mean": 0.0,   "std": 0.04},
            "knee_width_ratio":    {"mean": 1.0,   "std": 0.1},
        },
    }
    if overrides:
        for phase, keys in overrides.items():
            defaults[phase].update(keys)
    return defaults


def good_feat_down():
    """Analysis features for a correct squats_down position."""
    return {
        "avg_knee_angle":      80.0,
        "knee_symmetry":       2.0,
        "knee_valgus_left":    0.02,
        "knee_valgus_right":   0.02,
        "knee_width_ratio":    1.0,
        "rounded_back":        0.35,
        "spine_flexion":       12.0,
        "spine_angle_3d":      165.0,
        "trunk_lean":          14.0,
        "lumbar_extension":    0.02,
        "hip_shoulder_offset": 0.04,
        "stance_ratio":        1.2,
        "left_heel_lift":      -0.04,
        "right_heel_lift":     -0.04,
    }


def good_feat_up():
    return {
        "avg_knee_angle":      170.0,
        "knee_symmetry":       2.0,
        "knee_valgus_left":    0.02,
        "knee_valgus_right":   0.02,
        "knee_width_ratio":    1.0,
        "rounded_back":        0.38,
        "spine_flexion":       5.0,
        "spine_angle_3d":      168.0,
        "trunk_lean":          5.0,
        "lumbar_extension":    0.02,
        "hip_shoulder_offset": 0.03,
        "stance_ratio":        1.2,
        "left_heel_lift":      -0.04,
        "right_heel_lift":     -0.04,
    }


REF = make_ref_stats()


# ── Good form (no false positives) ────────────────────────────────────────────

class TestNoFalsePositives:
    def test_no_errors_good_down(self):
        errors, _ = detect_errors(good_feat_down(), "squats_down", REF, None)
        assert errors == [], f"False positives in squats_down: {errors}"

    def test_no_errors_good_up(self):
        errors, _ = detect_errors(good_feat_up(), "squats_up", REF, None)
        assert errors == [], f"False positives in squats_up: {errors}"

    def test_no_errors_empty_feat(self):
        errors, _ = detect_errors(None, "squats_down", REF, None)
        assert errors == []

    def test_no_errors_missing_ref(self):
        """If reference stats are missing, absolute thresholds still work."""
        errors, _ = detect_errors(good_feat_down(), "squats_down", {}, None)
        assert errors == []


# ── Insufficient depth ────────────────────────────────────────────────────────

class TestInsufficientDepth:
    def test_detected_via_reference(self):
        feat = {**good_feat_down(), "avg_knee_angle": 120.0}  # way too high
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" in errors

    def test_not_triggered_in_squats_up(self):
        feat = {**good_feat_up(), "avg_knee_angle": 120.0}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Insufficient depth" not in errors

    def test_borderline_fine(self):
        """Exactly at mean + 1.0*std → NOT an error."""
        feat = {**good_feat_down(), "avg_knee_angle": 88.0}  # mean(80) + 1*std(8)
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" not in errors


# ── Rounded back ──────────────────────────────────────────────────────────────

class TestRoundedBack:
    def test_detected_via_reference_rounded_back(self):
        """rounded_back drops well below reference mean."""
        feat = {**good_feat_down(), "rounded_back": 0.10, "spine_angle_3d": 165.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Rounded back" in errors

    def test_detected_via_absolute_spine_angle_3d(self):
        """spine_angle_3d < 145° = rounded back regardless of reference."""
        feat = {**good_feat_down(), "rounded_back": 0.35, "spine_angle_3d": 130.0}
        errors, _ = detect_errors(feat, "squats_down", {}, None)
        assert "Rounded back" in errors

    def test_not_duplicated_when_both_fire(self):
        """If reference AND absolute both detect it, error appears only once."""
        feat = {**good_feat_down(), "rounded_back": 0.05, "spine_angle_3d": 125.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert errors.count("Rounded back") == 1

    def test_not_triggered_when_fine(self):
        feat = {**good_feat_down(), "rounded_back": 0.35, "spine_angle_3d": 165.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Rounded back" not in errors

    def test_detected_in_squats_up(self):
        feat = {**good_feat_up(), "rounded_back": 0.05}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Rounded back" in errors


# ── Knee valgus (caving in) ───────────────────────────────────────────────────

class TestKneeValgus:
    def test_left_detected_via_reference(self):
        feat = {**good_feat_down(), "knee_valgus_left": 0.20}  # mean(0)+1.5*std(0.04)=0.06
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" in errors

    def test_right_detected_via_reference(self):
        feat = {**good_feat_down(), "knee_valgus_right": 0.20}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (right)" in errors

    def test_left_detected_via_absolute(self):
        """Absolute threshold: > 0.12"""
        feat = {**good_feat_down(), "knee_valgus_left": 0.15}
        errors, _ = detect_errors(feat, "squats_down", {}, None)
        assert "Knee valgus (left)" in errors

    def test_right_detected_via_absolute(self):
        feat = {**good_feat_down(), "knee_valgus_right": 0.15}
        errors, _ = detect_errors(feat, "squats_down", {}, None)
        assert "Knee valgus (right)" in errors

    def test_not_triggered_when_fine(self):
        feat = {**good_feat_down(), "knee_valgus_left": 0.02, "knee_valgus_right": 0.02}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" not in errors
        assert "Knee valgus (right)" not in errors

    def test_not_duplicated(self):
        feat = {**good_feat_down(), "knee_valgus_left": 0.20}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert errors.count("Knee valgus (left)") == 1


# ── Knees too wide ────────────────────────────────────────────────────────────

class TestKneesTooWide:
    def test_detected_via_reference(self):
        feat = {**good_feat_down(), "knee_width_ratio": 1.6}  # mean(1)+1.5*std(0.1)=1.15
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knees too wide" in errors

    def test_detected_via_absolute(self):
        """Absolute threshold: > 1.4"""
        feat = {**good_feat_down(), "knee_width_ratio": 1.5}
        errors, _ = detect_errors(feat, "squats_down", {}, None)
        assert "Knees too wide" in errors

    def test_not_triggered_when_fine(self):
        feat = {**good_feat_down(), "knee_width_ratio": 1.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knees too wide" not in errors


# ── Lumbar hyperextension ─────────────────────────────────────────────────────

class TestLumbarHyperextension:
    def test_detected_via_reference(self):
        feat = {**good_feat_down(), "lumbar_extension": 0.25}  # abs > mean(0)+1.5*std(0.05)=0.075
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Lumbar hyperextension" in errors

    def test_detected_negative_direction(self):
        """Swaying backward also fires."""
        feat = {**good_feat_down(), "lumbar_extension": -0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Lumbar hyperextension" in errors

    def test_detected_via_absolute(self):
        """Absolute threshold: abs > 0.15"""
        feat = {**good_feat_down(), "lumbar_extension": 0.20}
        errors, _ = detect_errors(feat, "squats_down", {}, None)
        assert "Lumbar hyperextension" in errors

    def test_not_triggered_when_fine(self):
        feat = {**good_feat_down(), "lumbar_extension": 0.03}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Lumbar hyperextension" not in errors


# ── Excessive forward lean ────────────────────────────────────────────────────

class TestForwardLean:
    def test_detected_in_squats_down(self):
        # mean(15) + 1.5*std(5) = 22.5
        feat = {**good_feat_down(), "trunk_lean": 35.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Excessive forward lean" in errors

    def test_not_triggered_in_squats_up(self):
        """Squats_up uses different label ('Leaning at top')."""
        feat = {**good_feat_up(), "trunk_lean": 35.0}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Excessive forward lean" not in errors

    def test_leaning_at_top_in_squats_up(self):
        # mean(5) + 1.0*std(5) = 10.0
        feat = {**good_feat_up(), "trunk_lean": 25.0}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Leaning at top" in errors


# ── Heels lifting ─────────────────────────────────────────────────────────────

class TestHeelsLifting:
    def test_left_heel_detected(self):
        feat = {**good_feat_down(), "left_heel_lift": -0.10}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Heels lifting" in errors

    def test_right_heel_detected(self):
        feat = {**good_feat_down(), "right_heel_lift": -0.10}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Heels lifting" in errors

    def test_not_triggered_in_squats_up(self):
        feat = {**good_feat_up(), "left_heel_lift": -0.10}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Heels lifting" not in errors


# ── Stability guard (transitional frames skipped) ─────────────────────────────

class TestStabilityGuard:
    def test_checks_skipped_during_transition(self):
        """Rapid knee angle change → transitional frame → most checks skip."""
        feat = {**good_feat_down(), "avg_knee_angle": 120.0}
        # prev_knee = 80, current = 120 → delta=40 > 8 → not stable
        errors, _ = detect_errors(feat, "squats_down", REF, prev_knee_angle=80.0)
        assert "Insufficient depth" not in errors

    def test_heel_lift_fires_even_during_transition(self):
        """Heel detection is not gated by stability."""
        feat = {**good_feat_down(), "avg_knee_angle": 120.0, "left_heel_lift": -0.10}
        errors, _ = detect_errors(feat, "squats_down", REF, prev_knee_angle=80.0)
        assert "Heels lifting" in errors

    def test_prev_knee_none_treated_as_stable(self):
        """First frame (prev=None) should behave as stable."""
        feat = {**good_feat_down(), "avg_knee_angle": 120.0}
        errors, _ = detect_errors(feat, "squats_down", REF, prev_knee_angle=None)
        assert "Insufficient depth" in errors


# ── Return value ──────────────────────────────────────────────────────────────

class TestReturnValue:
    def test_returns_updated_knee_angle(self):
        feat = {**good_feat_down(), "avg_knee_angle": 75.0}
        _, new_angle = detect_errors(feat, "squats_down", REF, None)
        assert new_angle == pytest.approx(75.0)

    def test_returns_prev_angle_on_none_feat(self):
        _, new_angle = detect_errors(None, "squats_down", REF, 88.0)
        assert new_angle == 88.0

    def test_no_duplicate_errors(self):
        """No error string appears more than once in the returned list."""
        feat = {
            **good_feat_down(),
            "rounded_back": 0.05,
            "spine_angle_3d": 120.0,
            "knee_valgus_left": 0.20,
        }
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert len(errors) == len(set(errors)), f"Duplicates in: {errors}"
