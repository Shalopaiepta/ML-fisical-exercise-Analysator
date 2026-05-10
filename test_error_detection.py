"""Tests for CSV-reference-based exercise error detection."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from error_detection import detect_errors, _knee_valgus


def ref_stats():
    """Reference stats shaped like Datasets/reference_*.csv after loading."""
    return {
        "squats_down": {
            "avg_knee_angle": {"mean": 132.93, "std": 24.07, "median": 116.94, "q50": 116.94},
            "knee_symmetry": {"mean": 14.96, "std": 12.00},
            "trunk_lean": {"mean": 19.91, "std": 16.17},
            "stance_ratio": {"mean": 1.59, "std": 0.63},
            "hip_shoulder_offset": {"mean": 0.33, "std": 0.26},
            "knee_valgus_left": {"mean": 0.10, "std": 0.34},
            "knee_valgus_right": {"mean": -0.02, "std": 0.27},
            "knee_width_ratio": {"mean": 0.77, "std": 0.33, "median": 0.925, "q50": 0.925},
            "rounded_back": {"mean": 0.19, "std": 0.085},
            "lumbar_extension": {"mean": -0.06, "std": 0.42},
        },
        "squats_up": {
            "avg_knee_angle": {"mean": 177.33, "std": 5.0, "median": 177.75, "q50": 177.75},
            "knee_symmetry": {"mean": 2.62, "std": 1.43},
            "trunk_lean": {"mean": 2.17, "std": 5.0},
            "stance_ratio": {"mean": 1.17, "std": 0.34},
            "hip_shoulder_offset": {"mean": 0.038, "std": 0.05},
            "knee_valgus_left": {"mean": 0.0, "std": 0.012},
            "knee_valgus_right": {"mean": 0.019, "std": 0.024},
            "knee_width_ratio": {"mean": 5.52, "std": 27.53, "median": 0.795, "q50": 0.795},
            "rounded_back": {"mean": 0.324, "std": 0.076},
            "lumbar_extension": {"mean": 0.009, "std": 0.044},
        },
        "pushups_down": {
            "avg_elbow_angle": {"mean": 85.0, "std": 5.0, "median": 85.0, "q50": 85.0},
            "elbow_symmetry": {"mean": 3.0, "std": 2.0},
            "hip_sag": {"mean": 0.02, "std": 0.03, "median": 0.02, "q50": 0.02},
            "hand_width_ratio": {"mean": 1.35, "std": 0.10, "median": 1.35, "q50": 1.35},
            "foot_width_ratio": {"mean": 0.45, "std": 0.08, "median": 0.45, "q50": 0.45},
        },
        "pushups_up": {
            "avg_elbow_angle": {"mean": 170.0, "std": 5.0, "median": 170.0, "q50": 170.0},
            "elbow_symmetry": {"mean": 3.0, "std": 2.0},
            "hip_sag": {"mean": 0.02, "std": 0.03, "median": 0.02, "q50": 0.02},
            "hand_width_ratio": {"mean": 1.35, "std": 0.10, "median": 1.35, "q50": 1.35},
            "foot_width_ratio": {"mean": 0.45, "std": 0.08, "median": 0.45, "q50": 0.45},
        },
    }


REF = ref_stats()


def good_down():
    return {
        "avg_knee_angle": 132.93,
        "knee_symmetry": 14.96,
        "knee_valgus_left": 0.10,
        "knee_valgus_right": -0.02,
        "knee_width_ratio": 0.77,
        "rounded_back": 0.19,
        "spine_angle_3d": 165.0,
        "trunk_lean": 19.91,
        "lumbar_extension": -0.06,
        "hip_shoulder_offset": 0.33,
        "stance_ratio": 1.59,
        "left_heel_lift": -0.10,
        "right_heel_lift": -0.19,
    }


def good_up():
    return {
        "avg_knee_angle": 177.33,
        "knee_symmetry": 2.62,
        "knee_valgus_left": 0.02,
        "knee_valgus_right": 0.02,
        "knee_width_ratio": 1.0,
        "rounded_back": 0.324,
        "spine_angle_3d": 168.0,
        "trunk_lean": 2.17,
        "lumbar_extension": 0.01,
        "hip_shoulder_offset": 0.038,
        "stance_ratio": 1.17,
        "left_heel_lift": -0.05,
        "right_heel_lift": -0.06,
    }


def good_pushup_down():
    return {
        "avg_elbow_angle": 85.0,
        "elbow_symmetry": 3.0,
        "hip_sag": 0.02,
        "hand_width_ratio": 1.35,
        "foot_width_ratio": 0.45,
        "avg_knee_angle": 170.0,
        "knee_symmetry": 0.0,
        "rounded_back": -1.0,
        "spine_angle_3d": 90.0,
    }


def good_pushup_up():
    return {
        **good_pushup_down(),
        "avg_elbow_angle": 170.0,
    }


class TestNoFalsePositives:
    def test_reference_like_bottom_position_has_no_errors(self):
        errors, _ = detect_errors(good_down(), "squats_down", REF, None)
        assert errors == []

    def test_reference_like_top_position_has_no_errors(self):
        errors, _ = detect_errors(good_up(), "squats_up", REF, None)
        assert errors == []

    def test_empty_features(self):
        errors, new_angle = detect_errors(None, "squats_down", REF, 90.0)
        assert errors == []
        assert new_angle == 90.0


class TestPushups:
    def test_reference_like_pushup_bottom_has_no_errors(self):
        errors, _ = detect_errors(good_pushup_down(), "pushups_down", REF, None)
        assert errors == []

    def test_reference_like_pushup_top_has_no_errors(self):
        errors, _ = detect_errors(good_pushup_up(), "pushups_up", REF, None)
        assert errors == []

    def test_hip_sagging_triggers(self):
        feat = {**good_pushup_down(), "hip_sag": 0.25}
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Hip sagging" in errors

    def test_hands_too_wide_triggers(self):
        feat = {**good_pushup_down(), "hand_width_ratio": 1.80}
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Hands too wide" in errors

    def test_hands_too_narrow_triggers(self):
        feat = {**good_pushup_down(), "hand_width_ratio": 0.80}
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Hands too narrow" in errors

    def test_feet_too_wide_triggers(self):
        feat = {**good_pushup_down(), "foot_width_ratio": 1.00}
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Feet too wide" in errors

    def test_feet_too_narrow_triggers(self):
        feat = {**good_pushup_down(), "foot_width_ratio": 0.30}
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Feet too narrow" in errors

    def test_pushup_phase_does_not_emit_squat_errors(self):
        feat = {
            **good_pushup_down(),
            "avg_knee_angle": 175.0,
            "knee_symmetry": 80.0,
            "rounded_back": -1.0,
            "knee_width_ratio": 2.0,
        }
        errors, _ = detect_errors(feat, "pushups_down", REF, None)
        assert "Insufficient depth" not in errors
        assert "Rounded back" not in errors
        assert "Leg asymmetry" not in errors
        assert "Knees too wide" not in errors

    def test_pushup_stability_uses_elbow_angle(self):
        feat = {**good_pushup_down(), "avg_elbow_angle": 120.0, "hand_width_ratio": 1.80}
        errors, new_angle = detect_errors(feat, "pushups_down", REF, prev_knee_angle=85.0)
        assert errors == []
        assert new_angle == pytest.approx(120.0)


class TestDepthAndExtension:
    def test_insufficient_depth_uses_csv_threshold(self):
        feat = {**good_down(), "avg_knee_angle": 140.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" in errors

    def test_sitting_reference_angle_does_not_trip_depth_absolute_limit(self):
        feat = {**good_down(), "avg_knee_angle": 133.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" not in errors

    def test_skewed_csv_mean_does_not_make_depth_too_lenient(self):
        feat = {**good_down(), "avg_knee_angle": 145.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" in errors

    def test_ref_data_changes_depth_threshold(self):
        custom_ref = ref_stats()
        custom_ref["squats_down"]["avg_knee_angle"] = {"mean": 90.0, "std": 5.0}
        feat = {**good_down(), "avg_knee_angle": 100.0}
        errors, _ = detect_errors(feat, "squats_down", custom_ref, None)
        assert "Insufficient depth" in errors

    def test_not_fully_extended_uses_csv_threshold(self):
        feat = {**good_up(), "avg_knee_angle": 160.0}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Not fully extended" in errors

    def test_not_fully_extended_not_checked_in_bottom_phase(self):
        feat = {**good_down(), "avg_knee_angle": 160.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Not fully extended" not in errors


class TestRoundedBack:
    def test_rounded_back_via_csv_threshold(self):
        feat = {**good_down(), "rounded_back": 0.02, "spine_angle_3d": 165.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Rounded back" in errors

    def test_rounded_back_via_3d_angle(self):
        feat = {**good_down(), "rounded_back": 0.19, "spine_angle_3d": 130.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Rounded back" in errors

    def test_rounded_back_not_duplicated(self):
        feat = {**good_down(), "rounded_back": -0.10, "spine_angle_3d": 130.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert errors.count("Rounded back") == 1


class TestKnees:
    def test_knee_valgus_geometry_is_positive_inward_left(self):
        points = {
            "left_hip": [0.5, 0.0],
            "left_knee": [0.45, 0.5],
            "left_ankle": [0.8, 1.0],
        }

        def n_func(key):
            return np.array(points[key], dtype=float)

        assert _knee_valgus(n_func, "left") > 0

    def test_knee_valgus_geometry_is_negative_outward_left(self):
        points = {
            "left_hip": [0.5, 0.0],
            "left_knee": [0.85, 0.5],
            "left_ankle": [0.8, 1.0],
        }

        def n_func(key):
            return np.array(points[key], dtype=float)

        assert _knee_valgus(n_func, "left") < 0

    def test_knee_valgus_geometry_is_positive_inward_right(self):
        points = {
            "right_hip": [-0.5, 0.0],
            "right_knee": [-0.45, 0.5],
            "right_ankle": [-0.8, 1.0],
        }

        def n_func(key):
            return np.array(points[key], dtype=float)

        assert _knee_valgus(n_func, "right") > 0

    def test_left_knee_valgus_positive(self):
        feat = {**good_down(), "knee_valgus_left": 0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" in errors

    def test_left_knee_varus_negative_is_not_valgus(self):
        feat = {**good_down(), "knee_valgus_left": -0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" not in errors

    def test_right_knee_valgus_positive(self):
        feat = {**good_down(), "knee_valgus_right": 0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (right)" in errors

    def test_right_knee_varus_negative_is_not_valgus(self):
        feat = {**good_down(), "knee_valgus_right": -0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (right)" not in errors

    def test_knee_valgus_floor_prevents_tiny_csv_std_false_positive(self):
        feat = {**good_up(), "knee_valgus_left": 0.08}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Knee valgus (left)" not in errors

    def test_knee_valgus_cap_keeps_noisy_down_csv_triggerable(self):
        feat = {**good_down(), "knee_valgus_left": 0.25}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" in errors

    def test_knees_too_wide_uses_reference_with_biomech_cap(self):
        feat = {**good_down(), "knee_width_ratio": 1.50}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knees too wide" in errors

    def test_knees_too_wide_not_triggered_after_small_threshold_lift(self):
        feat = {**good_down(), "knee_width_ratio": 1.35}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knees too wide" not in errors

    def test_knees_too_wide_not_triggered_near_reference(self):
        feat = {**good_down(), "knee_width_ratio": 1.10}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knees too wide" not in errors

    def test_knees_too_wide_only_checked_in_bottom_phase(self):
        feat = {**good_up(), "knee_width_ratio": 1.80}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Knees too wide" not in errors

    def test_knees_caving_ratio_triggers_both_valgus_errors(self):
        feat = {**good_down(), "knee_width_ratio": 0.70}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Knee valgus (left)" in errors
        assert "Knee valgus (right)" in errors

    def test_leg_asymmetry_uses_capped_reference_limit(self):
        feat = {**good_down(), "knee_symmetry": 25.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Leg asymmetry" in errors

    def test_reference_like_down_asymmetry_is_allowed(self):
        feat = {**good_down(), "knee_symmetry": 15.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Leg asymmetry" not in errors


class TestPosition:
    def test_hips_shifting_uses_csv_reference(self):
        feat = {**good_up(), "hip_shoulder_offset": 0.20}
        errors, _ = detect_errors(feat, "squats_up", REF, None)
        assert "Hips shifting" in errors

    def test_reference_like_down_hip_offset_is_allowed(self):
        feat = {**good_down(), "hip_shoulder_offset": 0.33}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Hips shifting" not in errors

    def test_stance_width_high(self):
        feat = {**good_down(), "stance_ratio": 3.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Stance width off" in errors

    def test_stance_width_low(self):
        feat = {**good_down(), "stance_ratio": 0.5}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Stance width off" in errors


class TestStabilityAndReturnValue:
    def test_transition_frame_skips_checks(self):
        feat = {**good_down(), "avg_knee_angle": 165.0, "knee_width_ratio": 1.50}
        errors, new_angle = detect_errors(feat, "squats_down", REF, prev_knee_angle=120.0)
        assert errors == []
        assert new_angle == pytest.approx(165.0)

    def test_small_knee_change_runs_checks(self):
        feat = {**good_down(), "avg_knee_angle": 165.0}
        errors, _ = detect_errors(feat, "squats_down", REF, prev_knee_angle=160.0)
        assert "Insufficient depth" in errors

    def test_no_duplicate_errors(self):
        feat = {**good_down(), "rounded_back": -0.10, "spine_angle_3d": 120.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert len(errors) == len(set(errors))

    def test_multiple_errors_returned_together(self):
        feat = {
            **good_down(),
            "avg_knee_angle": 165.0,
            "rounded_back": -0.10,
            "knee_valgus_left": 0.25,
            "knee_width_ratio": 1.50,
        }
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Insufficient depth" in errors
        assert "Rounded back" in errors
        assert "Knee valgus (left)" in errors
        assert "Knees too wide" in errors


class TestDisabledLegacyErrors:
    def test_heel_lift_does_not_fire(self):
        feat = {**good_down(), "left_heel_lift": -0.90, "right_heel_lift": -0.90}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Heels lifting" not in errors

    def test_forward_lean_does_not_fire(self):
        feat = {**good_down(), "trunk_lean": 80.0}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Excessive forward lean" not in errors

    def test_lumbar_extension_does_not_fire(self):
        feat = {**good_down(), "lumbar_extension": 0.90}
        errors, _ = detect_errors(feat, "squats_down", REF, None)
        assert "Lumbar hyperextension" not in errors
