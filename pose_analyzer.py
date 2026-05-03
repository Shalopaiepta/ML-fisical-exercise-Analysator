# pose_analyzer.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import os
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from error_detection import (
    detect_errors as shared_detect_errors,
    extract_features_for_analysis as shared_extract_features_for_analysis,
)


class PoseAnalyzer:
    def __init__(self):
        # 1. Загрузка ML-моделей (точно как в вашем live_classifier.py)
        self.model = joblib.load('exercise_classifier_final.pkl')
        self.scaler = joblib.load('angle_scaler_final.pkl')
        self.le = joblib.load('label_encoder_final.pkl')
        self.feature_order = joblib.load('feature_order.pkl')

        # 2. Загрузка эталонов
        self.ref_stats = {}
        REF_DIR = r'Datasets'
        for phase in ['squats_down', 'squats_up']:
            path = os.path.join(REF_DIR, f'reference_{phase}.csv')
            if os.path.exists(path):
                df = pd.read_csv(path)
                stats = {}
                for col in df.columns:
                    if col.startswith('ref_'):
                        key = col[4:]
                        m, s = df[col].mean(), df[col].std()
                        if 'angle' in key or 'lean' in key:
                            s = max(s, 5.0)
                        elif 'ratio' in key or 'offset' in key:
                            s = max(s, 0.05)
                        stats[key] = {'mean': m, 'std': s}
                self.ref_stats[phase] = stats

        # 3. MediaPipe
        base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

        # Константы
        self.CONNECTIONS = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),
            (23,25),(25,27),(24,26),(26,28),
            (27,31),(28,32)
        ]

        # Состояние между кадрами
        self.prev_model_feat = None
        self.prev_knee_angle = None

    # ====================== ВАШИ ФУНКЦИИ (1:1) ======================
    def calculate_angle(self, a, b, c):
        ba = np.array(a) - np.array(b)
        bc = np.array(c) - np.array(b)
        n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
        if n1 * n2 < 1e-8:
            return 0.0
        return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1.0, 1.0)))

    def get_coords(self, lm):
        idx_map = {
            'nose': 0, 'left_ear': 7, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32,
        }
        coords = {}
        for name, idx in idx_map.items():
            if idx < len(lm):
                coords[name] = np.array([lm[idx].x, lm[idx].y, lm[idx].z])
            else:
                coords[name] = np.array([np.nan, np.nan, np.nan])
        return coords

    def normalize(self, coords):
        l_hip, r_hip = coords['left_hip'][:2], coords['right_hip'][:2]
        center = (l_hip + r_hip) / 2
        l_sh, r_sh = coords['left_shoulder'][:2], coords['right_shoulder'][:2]
        scale = np.linalg.norm((l_sh + r_sh) / 2 - center) + 1e-6
        def n(key):
            return (coords[key][:2] - center) / scale
        return n, scale

    def extract_features_for_model(self, lm, prev=None):
        f = {}
        c = self.get_coords(lm)
        if any(np.isnan(c[k]).any() for k in ['nose', 'left_hip', 'right_hip']):
            return None
        n, _ = self.normalize(c)

        mid_sh = (n('left_shoulder') + n('right_shoulder')) / 2
        mid_hip = (n('left_hip') + n('right_hip')) / 2
        nose = n('nose')

        f['left_elbow_angle']     = self.calculate_angle(n('left_shoulder'),  n('left_elbow'),  n('left_wrist'))
        f['right_elbow_angle']    = self.calculate_angle(n('right_shoulder'), n('right_elbow'), n('right_wrist'))
        f['left_shoulder_angle']  = self.calculate_angle(n('left_elbow'),     n('left_shoulder'),  n('left_hip'))
        f['right_shoulder_angle'] = self.calculate_angle(n('right_elbow'),    n('right_shoulder'), n('right_hip'))
        f['left_knee_angle']      = self.calculate_angle(n('left_hip'),  n('left_knee'),  n('left_ankle'))
        f['right_knee_angle']     = self.calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))
        f['left_hip_angle']       = self.calculate_angle(n('left_shoulder'),  n('left_hip'),  n('left_knee'))
        f['right_hip_angle']      = self.calculate_angle(n('right_shoulder'), n('right_hip'), n('right_knee'))
        f['left_ankle_angle']     = self.calculate_angle(n('left_knee'),  n('left_ankle'),  n('left_foot_index'))
        f['right_ankle_angle']    = self.calculate_angle(n('right_knee'), n('right_ankle'), n('right_foot_index'))

        f['trunk_lean']  = np.degrees(np.arctan2(abs(mid_sh[0]-mid_hip[0]), abs(mid_sh[1]-mid_hip[1])+1e-6))
        f['head_tilt']   = np.degrees(np.arctan2(abs(nose[0]-mid_sh[0]),    abs(nose[1]-mid_sh[1])+1e-6))
        f['body_verticality']  = abs(mid_sh[1] - mid_hip[1])
        f['body_horizontal']   = abs(mid_sh[0] - mid_hip[0])
        f['orientation_ratio'] = f['body_verticality'] / (f['body_horizontal'] + 1e-6)

        model_landmarks = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle', 'left_foot_index', 'right_foot_index',
        ]
        all_pts = [n(k) for k in model_landmarks if k in c and not np.isnan(c[k]).any()]
        all_y   = [p[1] for p in all_pts]
        all_x   = [p[0] for p in all_pts]
        f['bbox_aspect'] = (max(all_y)-min(all_y)+1e-6) / (max(all_x)-min(all_x)+1e-6)

        f['knee_symmetry']     = abs(f['left_knee_angle']     - f['right_knee_angle'])
        f['elbow_symmetry']    = abs(f['left_elbow_angle']    - f['right_elbow_angle'])
        f['shoulder_symmetry'] = abs(f['left_shoulder_angle'] - f['right_shoulder_angle'])
        f['hip_symmetry']      = abs(f['left_hip_angle']      - f['right_hip_angle'])
        f['feet_width']             = np.linalg.norm(n('left_ankle') - n('right_ankle'))
        f['shoulder_width']         = np.linalg.norm(n('left_shoulder') - n('right_shoulder'))
        f['feet_to_shoulder_ratio'] = f['feet_width'] / (f['shoulder_width'] + 1e-6)
        f['hip_height']    = mid_hip[1] - mid_sh[1]
        f['height_ratio']  = np.linalg.norm(mid_sh - mid_hip) / (f['shoulder_width'] + 1e-6)

        vel_cols = ['left_elbow_angle','right_elbow_angle','left_shoulder_angle','right_shoulder_angle',
                    'left_knee_angle','right_knee_angle','left_hip_angle','right_hip_angle',
                    'trunk_lean','hip_height']
        for col in vel_cols:
            f[f'vel_{col}'] = (f[col] - prev[col]) if prev and col in prev else 0.0

        return f

    def extract_features_for_analysis(self, lm):
        return shared_extract_features_for_analysis(lm, self.get_coords, self.normalize)

    def predict_exercise(self, feat_dict):
        col_order = self.feature_order
        X = pd.DataFrame([{k: feat_dict.get(k, 0.0) for k in col_order}])
        X_sc = self.scaler.transform(X)
        pred  = self.model.predict(X_sc)[0]
        proba = self.model.predict_proba(X_sc)[0]
        return self.le.inverse_transform([pred])[0], float(proba[pred])

    def detect_errors(self, analysis_feat, phase, ref_data, prev_knee_angle):
        return shared_detect_errors(analysis_feat, phase, ref_data, prev_knee_angle)

    ERROR_DESCRIPTIONS = {
        'Insufficient depth': ('Глубина приседания недостаточна.', 'Опускайтесь ниже — бёдра должны быть как минимум параллельны полу.'),
        'Excessive forward lean': ('Корпус слишком сильно наклоняется вперёд.', 'Работайте над подвижностью голеностопа и держите грудь поднятой.'),
        'Not fully extended': ('В верхней точке колени не разгибаются полностью.', 'Вставайте до конца в верхней точке каждого повторения.'),
        'Rounded upper back': ('Верхняя часть спины округляется в верхней точке.', 'Держите грудь высоко, сводите лопатки вместе.'),
        'Hips shifting': ('Таз смещается в сторону во время движения.', 'Это часто указывает на дисбаланс.'),
        'Leaning at top': ('В верхней точке корпус наклоняется вперёд.', 'Концентрируйтесь на выталкивании бёдер вперёд.'),
        'Stance width off': ('Ширина постановки стоп нестабильна.', 'Найдите комфортную ширину стойки.'),
        'Leg asymmetry': ('Обнаружена значительная асимметрия углов в коленях.', 'Односторонние упражнения помогут.'),
        'Heels lifting': ('Пятки отрываются от пола.', 'Растягивайте икроножные мышцы или используйте подпятник.'),
    }
    ERROR_DESCRIPTIONS.update({
        'Rounded back': ('Спина округляется.', 'Держите грудь выше и не позволяйте плечам уходить вперед.'),
        'Knee valgus (left)': ('Левое колено заваливается внутрь.', 'Старайтесь вести колено наружу — по линии стопы.'),
        'Knee valgus (right)': ('Правое колено заваливается внутрь.', 'Старайтесь вести колено наружу — по линии стопы.'),
        'Knees too wide': ('Колени расходятся слишком широко.', 'Держите колени над стопами и без лишнего разведения.'),
        'Lumbar hyperextension': ('Поясница уходит в чрезмерный прогиб.', 'Держите корпус собранным и сохраняйте нейтральную поясницу.'),
    })

    def generate_report(self, session_error_counts: dict, total_frames: int, duration_sec: float) -> str:
        lines = []
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines.append('=' * 60)
        lines.append('      АНАЛИЗ ТЕХНИКИ ПРИСЕДАНИЙ')
        lines.append('=' * 60)
        lines.append(f'Дата          : {now}')
        lines.append(f'Продолжит.    : {duration_sec:.1f} сек')
        lines.append(f'Кадров        : {total_frames}')
        lines.append('')

        significant = {err: cnt for err, cnt in session_error_counts.items() if total_frames > 0 and (cnt / total_frames) >= 0.05}

        if not significant:
            lines.append('РЕЗУЛЬТАТ: Значимых ошибок не обнаружено.')
            lines.append('Техника приседаний выглядит хорошо. Продолжайте в том же духе!')
        else:
            sorted_errors = sorted(significant.items(), key=lambda x: x[1], reverse=True)
            freq_list = [(err, cnt / total_frames * 100) for err, cnt in sorted_errors]
            lines.append(f'РЕЗУЛЬТАТ: обнаружено нарушений — {len(significant)}.')
            lines.append('')
            lines.append('СВОДКА')
            lines.append('-' * 40)
            for err, pct in freq_list:
                bar = '█' * int(pct / 5)
                lines.append(f'  {err:<28} {pct:5.1f}%  {bar}')
            lines.append('')
            lines.append('ПОДРОБНЫЙ РАЗБОР')
            lines.append('-' * 40)
            for i, (err, pct) in enumerate(freq_list, 1):
                short_desc, advice = self.ERROR_DESCRIPTIONS.get(err, (err, 'Дополнительных рекомендаций нет.'))
                lines.append(f'{i}. {err}  ({pct:.1f}% of frames)')
                lines.append(f'   Что произошло : {short_desc}')
                lines.append(f'   Как исправить : {advice}')
                lines.append('')
            lines.append('ПРИОРИТЕТ')
            lines.append('-' * 40)
            top_err, top_pct = freq_list[0]
            short_desc, advice = self.ERROR_DESCRIPTIONS.get(top_err, (top_err, ''))
            lines.append(f'Сосредоточьтесь сначала на "{top_err}" — встречалось в {top_pct:.1f}% кадров.')
            lines.append(f'{advice}')

        lines.append('')
        lines.append('=' * 60)
        return '\n'.join(lines)

    # ====================== ОБРАБОТКА КАДРА ======================
    def process_frame(self, frame, timestamp_ms, is_recording):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect_for_video(mp_image, timestamp_ms)

        errors = []
        current_label = "Ожидание..."
        confidence = 0.0

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            h, w, _ = frame.shape

            # Рисуем скелет
            for start_idx, end_idx in self.CONNECTIONS:
                x1 = int(lm[start_idx].x * w)
                y1 = int(lm[start_idx].y * h)
                x2 = int(lm[end_idx].x * w)
                y2 = int(lm[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Полный анализ
            model_feat = self.extract_features_for_model(lm, self.prev_model_feat)
            if model_feat:
                current_label, confidence = self.predict_exercise(model_feat)
                self.prev_model_feat = model_feat

                if 'squats' in current_label and is_recording:
                    analysis_feat = self.extract_features_for_analysis(lm)
                    if analysis_feat:
                        raw_errors, self.prev_knee_angle = self.detect_errors(
                            analysis_feat, current_label, self.ref_stats, self.prev_knee_angle)
                        errors = raw_errors

        frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        return frame_rgb, current_label, confidence, errors
