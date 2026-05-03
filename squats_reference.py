import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.PoseLandmarker.create_from_options(options)

FRAMES_PER_CLASS = 300

CLASSES = {
    '1': 'squats_down',
    '2': 'squats_up',
}

COLORS = {
    'squats_down': (0, 100, 255),
    'squats_up':   (0, 100, 255),
}

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]

LANDMARK_NAMES = [
    'nose','left_eye_inner','left_eye','left_eye_outer',
    'right_eye_inner','right_eye','right_eye_outer',
    'left_ear','right_ear','mouth_left','mouth_right',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_pinky_1','right_pinky_1',
    'left_index_1','right_index_1','left_thumb_2','right_thumb_2',
    'left_hip','right_hip','left_knee','right_knee',
    'left_ankle','right_ankle','left_heel','right_heel',
    'left_foot_index','right_foot_index',
]

def calculate_angle(a, b, c):
    """Угол в точке b между векторами ba и bc. Возвращает градусы."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba * norm_bc < 1e-8:
        return 0.0
    cos_val = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

# ─── PATCH 1: Вспомогательные функции ────────────────────────────────────────
def _knee_valgus(n_func, side):
    hip   = n_func(f'{side}_hip')
    knee  = n_func(f'{side}_knee')
    ankle = n_func(f'{side}_ankle')
    limb_vec  = ankle - hip
    limb_unit = limb_vec / (np.linalg.norm(limb_vec) + 1e-6)
    knee_vec  = knee - hip
    proj      = np.dot(knee_vec, limb_unit)
    perp      = knee_vec - proj * limb_unit
    return perp[0] if side == 'left' else -perp[0]

def _knee_width_ratio(n_func):
    knee_w  = abs(n_func('left_knee')[0]  - n_func('right_knee')[0])
    ankle_w = abs(n_func('left_ankle')[0] - n_func('right_ankle')[0])
    return knee_w / (ankle_w + 1e-6)

def _rounded_back(n_func):
    nose   = n_func('nose')
    mid_sh = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    return mid_sh[1] - nose[1]

def _spine_flexion(n_func):
    mid_sh  = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    mid_hip = (n_func('left_hip')      + n_func('right_hip'))      / 2
    return np.degrees(np.arctan2(
        abs(mid_sh[0] - mid_hip[0]),
        abs(mid_sh[1] - mid_hip[1]) + 1e-6
    ))

def _lumbar_extension(n_func):
    mid_sh  = (n_func('left_shoulder') + n_func('right_shoulder')) / 2
    mid_hip = (n_func('left_hip')      + n_func('right_hip'))      / 2
    return mid_hip[0] - mid_sh[0]
# ──────────────────────────────────────────────────────────────────────────────

def extract_reference_features(lm):
    indices = {
        'nose': 0,
        'left_ear': 7, 'right_ear': 8,
        'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14,
        'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24,
        'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28,
        'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32,
    }

    coords = {}
    for name, idx in indices.items():
        if idx < len(lm):
            coords[name] = np.array([lm[idx].x, lm[idx].y, lm[idx].z])
        else:
            coords[name] = np.array([np.nan, np.nan, np.nan])

    if any(np.isnan(coords[k]).any() for k in ['left_hip', 'right_hip', 'left_knee']):
        return None

    l_hip, r_hip = coords['left_hip'][:2], coords['right_hip'][:2]
    center = (l_hip + r_hip) / 2
    l_sh, r_sh = coords['left_shoulder'][:2], coords['right_shoulder'][:2]
    scale = np.linalg.norm((l_sh + r_sh)/2 - center) + 1e-6

    def n(name):
        return (coords[name][:2] - center) / scale

    mid_sh  = (n('left_shoulder') + n('right_shoulder')) / 2
    mid_hip = (n('left_hip')      + n('right_hip'))      / 2

    f = {}
    f['avg_knee_angle'] = (
        calculate_angle(n('left_hip'),  n('left_knee'),  n('left_ankle')) +
        calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))
    ) / 2
    f['knee_symmetry'] = abs(
        calculate_angle(n('left_hip'),  n('left_knee'),  n('left_ankle')) -
        calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))
    )
    f['trunk_lean'] = np.degrees(np.arctan2(
        abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1]) + 1e-6))
    f['stance_ratio'] = np.linalg.norm(n('left_ankle') - n('right_ankle')) / (
                        np.linalg.norm(n('left_shoulder') - n('right_shoulder')) + 1e-6)
    f['hip_shoulder_offset'] = abs(
        (n('left_hip')[0]  + n('right_hip')[0])  / 2 -
        (n('left_shoulder')[0] + n('right_shoulder')[0]) / 2
    )
    f['left_heel_lift']  = n('left_heel')[1]  - n('left_foot_index')[1]
    f['right_heel_lift'] = n('right_heel')[1] - n('right_foot_index')[1]

    # ─── PATCH 5: Новые метрики (вместо старых avg_shoulder_angle и spine_angle) ─
    f['knee_valgus_left']   = _knee_valgus(n, 'left')
    f['knee_valgus_right']  = _knee_valgus(n, 'right')
    f['knee_width_ratio']   = _knee_width_ratio(n)
    f['rounded_back']       = _rounded_back(n)
    f['spine_flexion']      = _spine_flexion(n)
    f['lumbar_extension']   = _lumbar_extension(n)

    return f


# ── Главный цикл ──────────────────────────────────────────────────────────────
all_rows = []
pose_id = 0
current_class = None
recording = False
frames_recorded = 0
countdown = 0
countdown_start = 0

cap = cv2.VideoCapture(0)
timestamp_counter = 0

print('ЗАПИСЬ ЭТАЛОННОЙ ФОРМЫ ПРИСЕДАНИЙ (Patch 5)')
print('Инструкция:')
print('  1 — начать запись squats_down')
print('  2 — начать запись squats_up')
print('  S — сохранить данные в CSV')
print('  Q — выйти')
print(f'На каждый класс: {FRAMES_PER_CLASS} кадров (~10 сек)')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    timestamp_counter += 33
    result = detector.detect_for_video(mp_image, timestamp_counter)
    skeleton_color = (100, 100, 100)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        if current_class: skeleton_color = COLORS.get(current_class, (0, 255, 0))

        for s_idx, e_idx in CONNECTIONS:
            x1, y1 = int(lm[s_idx].x * w), int(lm[s_idx].y * h)
            x2, y2 = int(lm[e_idx].x * w), int(lm[e_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 2)
        for point in lm:
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 255), -1)

        if countdown > 0:
            elapsed = time.time() - countdown_start
            remaining = countdown - int(elapsed)
            if remaining > 0:
                cv2.putText(frame, f'Get ready: {remaining}', (w//2 - 100, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 3)
            else:
                countdown = 0
                recording = True
                frames_recorded = 0

        elif recording and current_class:
            row = {'pose_id': pose_id, 'pose': current_class}
            for i, name in enumerate(LANDMARK_NAMES):
                row[f'x_{name}'] = lm[i].x if i < len(lm) else np.nan
                row[f'y_{name}'] = lm[i].y if i < len(lm) else np.nan
                row[f'z_{name}'] = lm[i].z if i < len(lm) else np.nan

            ref_features = extract_reference_features(lm)
            if ref_features:
                for k, v in ref_features.items():
                    if not np.isnan(v):
                        row[f'ref_{k}'] = v

            all_rows.append(row)
            pose_id += 1
            frames_recorded += 1

            # Обновлённый UI: показываем rounded_back вместо spine_angle
            if ref_features and 'rounded_back' in ref_features:
                rb = ref_features['rounded_back']
                rb_color = (0, 255, 0) if rb > 0.05 else (0, 165, 255)
                cv2.putText(frame, f'Rounded back: {rb:.2f}',
                            (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rb_color, 2)

            progress = int((frames_recorded / FRAMES_PER_CLASS) * (w - 20))
            cv2.rectangle(frame, (10, h-30), (w-10, h-10), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, h-30), (10 + progress, h-10), skeleton_color, -1)
            cv2.putText(frame, f'Recording {current_class}: {frames_recorded}/{FRAMES_PER_CLASS}',
                        (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, skeleton_color, 2)

            if frames_recorded >= FRAMES_PER_CLASS:
                recording = False
                print(f'✅ Записано {FRAMES_PER_CLASS} кадров для {current_class}')
                current_class = None

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    recorded = {cls: sum(1 for r in all_rows if r['pose'] == cls) for cls in CLASSES.values()}
    status_parts = [f'{key}:{cls}({recorded[cls]})' for key, cls in CLASSES.items()]
    cv2.putText(frame, '  '.join(status_parts), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    if not recording and countdown == 0:
        cv2.putText(frame, 'Press 1-2 to record | S to save | Q to quit',
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow('Record Reference Form', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        if all_rows:
            df = pd.DataFrame(all_rows)
            os.makedirs(r'Datasets', exist_ok=True)
            for cls in CLASSES.values():
                cls_df = df[df['pose'] == cls].copy()
                if len(cls_df) == 0: continue

                filename = f'Datasets/reference_{cls}.csv'
                cls_df.to_csv(filename, index=False)
                print(f'💾 Сохранено {len(cls_df)} строк в {filename}')

                ref_cols = [c for c in cls_df.columns if c.startswith('ref_')]
                if ref_cols:
                    print(f'   Метрики ({cls}):')
                    for col in ['ref_avg_knee_angle', 'ref_trunk_lean', 'ref_stance_ratio',
                                'ref_rounded_back', 'ref_spine_flexion', 'ref_lumbar_extension',
                                'ref_knee_valgus_left', 'ref_knee_valgus_right']:
                        if col in cls_df.columns:
                            valid = cls_df[col].dropna()
                            if len(valid) > 0:
                                m, s = valid.mean(), valid.std()
                                print(f'     {col[4:]:22s}: {m:6.2f} ± {s:4.2f}')
            print('\n🚀 Готово! Теперь обнови live_classifier.py теми же функциями.')
        else: print('⚠️ Нет данных для сохранения')
    elif not recording and countdown == 0:
        cls_key = chr(key) if key < 128 else None
        if cls_key in CLASSES:
            current_class = CLASSES[cls_key]
            countdown = 3
            countdown_start = time.time()

cap.release()
cv2.destroyAllWindows()