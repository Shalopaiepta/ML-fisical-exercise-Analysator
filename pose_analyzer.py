import os
from collections import Counter, deque
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from error_detection import (
    detect_errors as shared_detect_errors,
    extract_features_for_analysis as shared_extract_features_for_analysis,
)


LABEL_SMOOTHING_FRAMES = 8
ERROR_SMOOTHING_FRAMES = 12
ERROR_CONFIRM_FRAMES = 7


class PoseAnalyzer:
    SUPPORTED_PHASES = ("squats_down", "squats_up", "pushups_down", "pushups_up")
    PHASE_LABELS = {
        "waiting": "Ожидание...",
        "squats_down": "Приседания: нижняя точка",
        "squats_up": "Приседания: верхняя точка",
        "pushups_down": "Отжимания: нижняя точка",
        "pushups_up": "Отжимания: верхняя точка",
    }
    REFERENCE_GROUPS = {
        "squats": ("squats_down", "squats_up"),
        "pushups": ("pushups_down", "pushups_up"),
    }
    REPORT_TITLES = {
        "squats": "АНАЛИЗ ТЕХНИКИ ПРИСЕДАНИЙ",
        "pushups": "АНАЛИЗ ТЕХНИКИ ОТЖИМАНИЙ",
        "mixed": "АНАЛИЗ ТЕХНИКИ УПРАЖНЕНИЙ",
        None: "АНАЛИЗ ТЕХНИКИ УПРАЖНЕНИЙ",
    }
    SUCCESS_MESSAGES = {
        "squats": "Техника приседаний выглядит хорошо. Продолжайте в том же духе!",
        "pushups": "Техника отжиманий выглядит хорошо. Продолжайте в том же духе!",
        "mixed": "Техника упражнения выглядит хорошо. Продолжайте в том же духе!",
        None: "Техника упражнения выглядит хорошо. Продолжайте в том же духе!",
    }
    ERROR_DESCRIPTIONS = {
        "Insufficient depth": (
            "Глубина приседания недостаточна.",
            "Опускайтесь ниже — бедра должны быть как минимум параллельны полу.",
        ),
        "Not fully extended": (
            "В верхней точке колени не разгибаются полностью.",
            "Вставайте до конца в верхней точке каждого повторения.",
        ),
        "Rounded back": (
            "Спина округляется.",
            "Держите грудь выше и не позволяйте плечам уходить вперед.",
        ),
        "Hips shifting": (
            "Таз смещается в сторону во время движения.",
            "Это часто указывает на дисбаланс. Добавьте односторонние упражнения и следите за равномерным давлением в стопы.",
        ),
        "Stance width off": (
            "Ширина постановки стоп нестабильна.",
            "Найдите комфортную ширину стойки и держите ее одинаковой от повтора к повтору.",
        ),
        "Leg asymmetry": (
            "Обнаружена заметная асимметрия работы ног.",
            "Полезно добавить односторонние упражнения и отдельно проверить подвижность левой и правой сторон.",
        ),
        "Knee valgus (left)": (
            "Левое колено заваливается внутрь.",
            "Старайтесь вести колено наружу — по линии стопы.",
        ),
        "Knee valgus (right)": (
            "Правое колено заваливается внутрь.",
            "Старайтесь вести колено наружу — по линии стопы.",
        ),
        "Knees too wide": (
            "Колени расходятся слишком широко.",
            "Держите колени над стопами без лишнего разведения.",
        ),
        "Hip sagging": (
            "Таз провисает в отжимании.",
            "Держите корпус одной линией и заранее напрягайте пресс перед опусканием.",
        ),
        "Hands too wide": (
            "Руки стоят шире эталонной стойки для отжиманий.",
            "Подведите ладони ближе к линии плеч.",
        ),
        "Hands too narrow": (
            "Руки стоят уже эталонной стойки для отжиманий.",
            "Слегка расширьте постановку рук для устойчивой опоры.",
        ),
        "Feet too wide": (
            "Стопы расставлены шире эталонной стойки для отжиманий.",
            "Сведите стопы ближе к вашей обычной стойке.",
        ),
        "Feet too narrow": (
            "Стопы расставлены слишком узко для эталонной стойки.",
            "Слегка расширьте постановку стоп, если не хватает устойчивости.",
        ),
    }

    def __init__(self):
        self.model = joblib.load("exercise_classifier_final.pkl")
        self.scaler = joblib.load("angle_scaler_final.pkl")
        self.le = joblib.load("label_encoder_final.pkl")
        self.feature_order = joblib.load("feature_order.pkl")

        self.reload_reference_stats()

        base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.CONNECTIONS = [
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (24, 26), (26, 28),
            (27, 31), (28, 32),
        ]

        self.prev_model_feat = None
        self.prev_knee_angle = None
        self.label_buffer = deque(maxlen=LABEL_SMOOTHING_FRAMES)
        self.error_buffer = deque(maxlen=ERROR_SMOOTHING_FRAMES)

    def reload_reference_stats(self):
        self.ref_stats = {}
        ref_dir = "Datasets"
        for phase in self.SUPPORTED_PHASES:
            path = os.path.join(ref_dir, f"reference_{phase}.csv")
            if not os.path.exists(path):
                continue

            df = pd.read_csv(path)
            stats = {}
            for col in df.columns:
                if not col.startswith("ref_"):
                    continue

                key = col[4:]
                mean = df[col].mean()
                std = df[col].std()
                if "angle" in key or "lean" in key:
                    std = max(std, 5.0)
                elif "ratio" in key or "offset" in key:
                    std = max(std, 0.05)

                stats[key] = {
                    "mean": mean,
                    "std": std,
                    "median": df[col].median(),
                    "q10": df[col].quantile(0.10),
                    "q50": df[col].quantile(0.50),
                    "q90": df[col].quantile(0.90),
                }

            if stats:
                self.ref_stats[phase] = stats

    def reference_ready(self, family):
        phases = self.REFERENCE_GROUPS.get(family, ())
        return bool(phases) and all(self.ref_stats.get(phase) for phase in phases)

    def missing_reference_phases(self, family=None):
        phases = self.SUPPORTED_PHASES if family is None else self.REFERENCE_GROUPS.get(family, ())
        return [phase for phase in phases if not self.ref_stats.get(phase)]

    def pretty_label(self, phase):
        return self.PHASE_LABELS.get(phase, phase)

    def detect_session_family(self, session_label_counts):
        if not session_label_counts:
            return None

        family_counts = Counter()
        for label, count in session_label_counts.items():
            if label.startswith("squats"):
                family_counts["squats"] += count
            elif label.startswith("pushups"):
                family_counts["pushups"] += count

        if not family_counts:
            return None

        if len(family_counts) == 1:
            return next(iter(family_counts))

        top_two = family_counts.most_common(2)
        if top_two[0][1] >= top_two[1][1] * 1.5:
            return top_two[0][0]
        return "mixed"

    def calculate_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 * n2 < 1e-8:
            return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)))

    def get_coords(self, lm):
        idx_map = {
            "nose": 0, "left_ear": 7, "right_ear": 8,
            "left_shoulder": 11, "right_shoulder": 12,
            "left_elbow": 13, "right_elbow": 14, "left_wrist": 15, "right_wrist": 16,
            "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
            "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
            "left_foot_index": 31, "right_foot_index": 32,
        }
        coords = {}
        for name, idx in idx_map.items():
            if idx < len(lm):
                coords[name] = np.array([lm[idx].x, lm[idx].y, lm[idx].z])
            else:
                coords[name] = np.array([np.nan, np.nan, np.nan])
        return coords

    def normalize(self, coords):
        l_hip, r_hip = coords["left_hip"][:2], coords["right_hip"][:2]
        center = (l_hip + r_hip) / 2
        l_sh, r_sh = coords["left_shoulder"][:2], coords["right_shoulder"][:2]
        scale = np.linalg.norm((l_sh + r_sh) / 2 - center) + 1e-6

        def n(key):
            return (coords[key][:2] - center) / scale

        return n, scale

    def extract_features_for_model(self, lm, prev=None):
        features = {}
        coords = self.get_coords(lm)
        if any(np.isnan(coords[key]).any() for key in ["nose", "left_hip", "right_hip"]):
            return None

        n_func, _ = self.normalize(coords)

        mid_sh = (n_func("left_shoulder") + n_func("right_shoulder")) / 2
        mid_hip = (n_func("left_hip") + n_func("right_hip")) / 2
        nose = n_func("nose")

        features["left_elbow_angle"] = self.calculate_angle(n_func("left_shoulder"), n_func("left_elbow"), n_func("left_wrist"))
        features["right_elbow_angle"] = self.calculate_angle(n_func("right_shoulder"), n_func("right_elbow"), n_func("right_wrist"))
        features["left_shoulder_angle"] = self.calculate_angle(n_func("left_elbow"), n_func("left_shoulder"), n_func("left_hip"))
        features["right_shoulder_angle"] = self.calculate_angle(n_func("right_elbow"), n_func("right_shoulder"), n_func("right_hip"))
        features["left_knee_angle"] = self.calculate_angle(n_func("left_hip"), n_func("left_knee"), n_func("left_ankle"))
        features["right_knee_angle"] = self.calculate_angle(n_func("right_hip"), n_func("right_knee"), n_func("right_ankle"))
        features["left_hip_angle"] = self.calculate_angle(n_func("left_shoulder"), n_func("left_hip"), n_func("left_knee"))
        features["right_hip_angle"] = self.calculate_angle(n_func("right_shoulder"), n_func("right_hip"), n_func("right_knee"))
        features["left_ankle_angle"] = self.calculate_angle(n_func("left_knee"), n_func("left_ankle"), n_func("left_foot_index"))
        features["right_ankle_angle"] = self.calculate_angle(n_func("right_knee"), n_func("right_ankle"), n_func("right_foot_index"))

        features["trunk_lean"] = np.degrees(np.arctan2(abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1]) + 1e-6))
        features["head_tilt"] = np.degrees(np.arctan2(abs(nose[0] - mid_sh[0]), abs(nose[1] - mid_sh[1]) + 1e-6))
        features["body_verticality"] = abs(mid_sh[1] - mid_hip[1])
        features["body_horizontal"] = abs(mid_sh[0] - mid_hip[0])
        features["orientation_ratio"] = features["body_verticality"] / (features["body_horizontal"] + 1e-6)

        model_landmarks = [
            "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle", "left_foot_index", "right_foot_index",
        ]
        all_pts = [n_func(key) for key in model_landmarks if key in coords and not np.isnan(coords[key]).any()]
        all_y = [point[1] for point in all_pts]
        all_x = [point[0] for point in all_pts]
        features["bbox_aspect"] = (max(all_y) - min(all_y) + 1e-6) / (max(all_x) - min(all_x) + 1e-6)

        features["knee_symmetry"] = abs(features["left_knee_angle"] - features["right_knee_angle"])
        features["elbow_symmetry"] = abs(features["left_elbow_angle"] - features["right_elbow_angle"])
        features["shoulder_symmetry"] = abs(features["left_shoulder_angle"] - features["right_shoulder_angle"])
        features["hip_symmetry"] = abs(features["left_hip_angle"] - features["right_hip_angle"])
        features["feet_width"] = np.linalg.norm(n_func("left_ankle") - n_func("right_ankle"))
        features["shoulder_width"] = np.linalg.norm(n_func("left_shoulder") - n_func("right_shoulder"))
        features["feet_to_shoulder_ratio"] = features["feet_width"] / (features["shoulder_width"] + 1e-6)
        features["hip_height"] = mid_hip[1] - mid_sh[1]
        features["height_ratio"] = np.linalg.norm(mid_sh - mid_hip) / (features["shoulder_width"] + 1e-6)

        vel_cols = [
            "left_elbow_angle", "right_elbow_angle", "left_shoulder_angle", "right_shoulder_angle",
            "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
            "trunk_lean", "hip_height",
        ]
        for col in vel_cols:
            features[f"vel_{col}"] = (features[col] - prev[col]) if prev and col in prev else 0.0

        return features

    def extract_features_for_analysis(self, lm):
        return shared_extract_features_for_analysis(lm, self.get_coords, self.normalize)

    def predict_exercise(self, feat_dict):
        col_order = self.feature_order
        frame = pd.DataFrame([{key: feat_dict.get(key, 0.0) for key in col_order}])
        scaled = self.scaler.transform(frame)
        pred = self.model.predict(scaled)[0]
        proba = self.model.predict_proba(scaled)[0]
        return self.le.inverse_transform([pred])[0], float(proba[pred])

    def detect_errors(self, analysis_feat, phase, ref_data, prev_knee_angle):
        return shared_detect_errors(analysis_feat, phase, ref_data, prev_knee_angle)

    def reset_runtime_state(self):
        self.prev_model_feat = None
        self.prev_knee_angle = None
        self.label_buffer.clear()
        self.error_buffer.clear()

    def reset_session_state(self):
        self.prev_knee_angle = None
        self.error_buffer.clear()

    def generate_report(self, session_error_counts, total_frames, duration_sec, session_label_counts=None):
        family = self.detect_session_family(session_label_counts)
        title = self.REPORT_TITLES.get(family, self.REPORT_TITLES[None])

        lines = []
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append("=" * 60)
        lines.append(f"      {title}")
        lines.append("=" * 60)
        lines.append(f"Дата          : {now}")
        lines.append(f"Продолжит.    : {duration_sec:.1f} сек")
        lines.append(f"Кадров        : {total_frames}")
        lines.append("")

        significant = {
            error: count
            for error, count in session_error_counts.items()
            if total_frames > 0 and (count / total_frames) >= 0.05
        }

        if not significant:
            lines.append("РЕЗУЛЬТАТ: значимых ошибок не обнаружено.")
            lines.append(self.SUCCESS_MESSAGES.get(family, self.SUCCESS_MESSAGES[None]))
            lines.append("")
            lines.append("=" * 60)
            return "\n".join(lines)

        sorted_errors = sorted(significant.items(), key=lambda item: item[1], reverse=True)
        freq_list = [(error, count / total_frames * 100) for error, count in sorted_errors]

        lines.append(f"РЕЗУЛЬТАТ: обнаружено нарушений — {len(significant)}.")
        lines.append("")
        lines.append("СВОДКА")
        lines.append("-" * 40)
        for error, pct in freq_list:
            bar = "█" * int(pct / 5)
            lines.append(f"  {error:<28} {pct:5.1f}%  {bar}")
        lines.append("")

        lines.append("ПОДРОБНЫЙ РАЗБОР")
        lines.append("-" * 40)
        for index, (error, pct) in enumerate(freq_list, 1):
            short_desc, advice = self.ERROR_DESCRIPTIONS.get(
                error,
                (error, "Дополнительных рекомендаций нет."),
            )
            lines.append(f"{index}. {error}  ({pct:.1f}% кадров)")
            lines.append(f"   Что произошло : {short_desc}")
            lines.append(f"   Как исправить : {advice}")
            lines.append("")

        lines.append("ПРИОРИТЕТ")
        lines.append("-" * 40)
        top_error, top_pct = freq_list[0]
        _, advice = self.ERROR_DESCRIPTIONS.get(top_error, (top_error, ""))
        lines.append(f'Сосредоточьтесь сначала на "{top_error}" — встречалось в {top_pct:.1f}% кадров.')
        if advice:
            lines.append(advice)

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def process_frame(self, frame, timestamp_ms, is_recording):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        errors = []
        current_label = "waiting"
        confidence = 0.0

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            h, w, _ = frame.shape

            for start_idx, end_idx in self.CONNECTIONS:
                x1 = int(lm[start_idx].x * w)
                y1 = int(lm[start_idx].y * h)
                x2 = int(lm[end_idx].x * w)
                y2 = int(lm[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            model_feat = self.extract_features_for_model(lm, self.prev_model_feat)
            if model_feat:
                predicted_label, confidence = self.predict_exercise(model_feat)
                self.label_buffer.append(predicted_label)
                current_label = Counter(self.label_buffer).most_common(1)[0][0]
                self.prev_model_feat = model_feat

                if current_label in self.SUPPORTED_PHASES and is_recording:
                    analysis_feat = self.extract_features_for_analysis(lm)
                    if analysis_feat:
                        raw_errors, self.prev_knee_angle = self.detect_errors(
                            analysis_feat,
                            current_label,
                            self.ref_stats,
                            self.prev_knee_angle,
                        )
                        self.error_buffer.append(set(raw_errors))
                        stable_errors = Counter(
                            error
                            for frame_errors in self.error_buffer
                            for error in frame_errors
                        )
                        errors = sorted(
                            error
                            for error, count in stable_errors.items()
                            if count >= ERROR_CONFIRM_FRAMES
                        )
                    else:
                        self.prev_knee_angle = None
                        self.error_buffer.clear()
                else:
                    self.prev_knee_angle = None
                    self.error_buffer.clear()

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        return frame_rgb, current_label, confidence, errors
