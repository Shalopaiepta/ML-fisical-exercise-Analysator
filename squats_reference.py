import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Инициализируем MediaPipe Pose Landmarker в режиме VIDEO.
# Режим VIDEO использует трекинг между кадрами — координаты стабильнее, меньше дребезга.
base_options = python.BaseOptions(model_asset_path='pose_landmarker_full.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.PoseLandmarker.create_from_options(options)

# Сколько кадров записывать на одну фазу. 300 кадров ≈ 10 секунд при 30 FPS.
# Больше данных = стабильнее статистика (mean/std), меньше ложных срабатываний при анализе.
FRAMES_PER_CLASS = 300

# Маппинг клавиш на классы. Разделяем фазы приседаний, чтобы эталоны не смешивались.
CLASSES = {
    '1': 'squats_down',  # Нижняя точка: колени согнуты, бёдра параллельны полу
    '2': 'squats_up',    # Верхняя точка: ноги выпрямлены, корпус вертикален
}

# Цвета для отрисовки скелета в UI (BGR-формат OpenCV)
COLORS = {
    'squats_down': (0, 100, 255),  # Оранжевый для нижней фазы
    'squats_up':   (0, 100, 255),  # Тот же цвет — визуально не нужно различать
}

# Пары индексов ландмарков для отрисовки линий скелета.
# Используется подмножество ключевых соединений, чтобы не перегружать UI.
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # Руки и плечи
    (11, 23), (12, 24), (23, 24),                       # Торс
    (23, 25), (25, 27), (24, 26), (26, 28),             # Ноги до лодыжек
    (27, 31), (28, 32),                                  # Стопы
]

# Имена ландмарков в порядке индексов MediaPipe.
# Нужны для сохранения сырых координат в CSV с человекочитаемыми колонками.
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
    """Считает угол в точке b между векторами ba и bc.
    Возвращает градусы. Эпсилон и clip() защищают от деления на ноль и math domain error."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    norm_ba, norm_bc = np.linalg.norm(ba), np.linalg.norm(bc)
    if norm_ba * norm_bc < 1e-8:
        return 0.0
    cos_val = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))

def extract_reference_features(lm):
    """Извлекает биомеханические метрики для эталона.
    Важно: логика должна быть идентична live_classifier.py, иначе сравнение будет некорректным."""
    f = {}

    # Маппинг индексов MediaPipe на имена суставов для читаемости кода
    indices = {
        'nose': 0, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16,
        'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
        'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
        'left_foot_index': 31, 'right_foot_index': 32
    }

    # Извлекаем координаты, подставляя NaN для пропущенных точек (бывает при заслонах)
    coords = {}
    for name, idx in indices.items():
        if idx < len(lm):
            coords[name] = np.array([lm[idx].x, lm[idx].y])
        else:
            coords[name] = np.array([np.nan, np.nan])

    # Если критические точки потеряны — пропускаем кадр, чтобы не портить статистику
    if any(np.isnan(coords[k]).any() for k in ['left_hip', 'right_hip', 'left_knee']):
        return None

    # Нормализуем координаты относительно центра таза и масштаба торса.
    # Это делает метрики инвариантными к дистанции до камеры и росту человека.
    l_hip, r_hip = coords['left_hip'], coords['right_hip']
    center = (l_hip + r_hip) / 2
    l_sh, r_sh = coords['left_shoulder'], coords['right_shoulder']
    scale = np.linalg.norm((l_sh + r_sh)/2 - center) + 1e-6

    # Удобная функция для доступа к нормализованным координатам
    def n(name):
        return (coords[name] - center) / scale

    mid_sh = (n('left_shoulder') + n('right_shoulder')) / 2
    mid_hip = (n('left_hip') + n('right_hip')) / 2

    # Средний угол в коленях — основной индикатор глубины приседа
    f['avg_knee_angle'] = (
        calculate_angle(n('left_hip'), n('left_knee'), n('left_ankle')) +
        calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))
    ) / 2

    # Асимметрия ног: большая разница = перекос нагрузки
    f['knee_symmetry'] = abs(
        calculate_angle(n('left_hip'), n('left_knee'), n('left_ankle')) -
        calculate_angle(n('right_hip'), n('right_knee'), n('right_ankle'))
    )

    # Угол в плечах: меньше = более округлённая спина
    f['avg_shoulder_angle'] = (
        calculate_angle(n('left_elbow'), n('left_shoulder'), n('left_hip')) +
        calculate_angle(n('right_elbow'), n('right_shoulder'), n('right_hip'))
    ) / 2

    # Наклон корпуса: угол между линией плечи-бёдра и вертикалью
    f['trunk_lean'] = np.degrees(np.arctan2(
        abs(mid_sh[0] - mid_hip[0]), abs(mid_sh[1] - mid_hip[1]) + 1e-6))

    # Пропорции стойки: отношение ширины стоп к ширине плеч
    f['shoulder_width'] = np.linalg.norm(n('left_shoulder') - n('right_shoulder'))
    f['feet_width'] = np.linalg.norm(n('left_ankle') - n('right_ankle'))
    f['stance_ratio'] = f['feet_width'] / (f['shoulder_width'] + 1e-6)

    # Горизонтальное смещение таза относительно плеч (компенсация, "клевок" тазом)
    f['hip_shoulder_offset'] = abs(
        (n('left_hip')[0] + n('right_hip')[0])/2 -
        (n('left_shoulder')[0] + n('right_shoulder')[0])/2
    )

    # Отрыв пяток: в системе MediaPipe Y растёт вниз.
    # Если Y пятки < Y носка — пятка физически приподнята над полом.
    f['left_heel_lift'] = n('left_heel')[1] - n('left_foot_index')[1]
    f['right_heel_lift'] = n('right_heel')[1] - n('right_foot_index')[1]

    return f

# Состояние скрипта: буфер для строк, счётчики, флаги записи
all_rows = []
pose_id = 0
current_class = None
recording = False
frames_recorded = 0
countdown = 0
countdown_start = 0

cap = cv2.VideoCapture(0)
timestamp_counter = 0  # Эмуляция таймстемпов для MediaPipe (~33 мс = 30 FPS)

print(' ЗАПИСЬ ЭТАЛОННОЙ ФОРМЫ ПРИСЕДАНИЙ')
print('Инструкция:')
print('  1 — начать запись squats_down  (присядь ВНИЗ и держи ПРАВИЛЬНУЮ позу)')
print('  2 — начать запись squats_up    (встань ВВЕРХ и держи ПРАВИЛЬНУЮ позу)')
print('  S — сохранить данные в CSV')
print('  Q — выйти')
print()
print(f'На каждый класс записывается {FRAMES_PER_CLASS} кадров (~10 сек)')
print('Совет: сделай 3-4 медленных повторения в каждой фазе для лучшего покрытия')

# Главный цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # Инкремент таймстемпа для корректного трекинга в MediaPipe
    timestamp_counter += 33
    result = detector.detect_for_video(mp_image, timestamp_counter)

    skeleton_color = (100, 100, 100)  # Серый по умолчанию

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        # Меняем цвет скелета, если идёт запись конкретной фазы
        if current_class:
            skeleton_color = COLORS.get(current_class, (0, 255, 0))

        # Отрисовка скелета: линии между ключевыми суставами
        for start_idx, end_idx in CONNECTIONS:
            x1, y1 = int(lm[start_idx].x * w), int(lm[start_idx].y * h)
            x2, y2 = int(lm[end_idx].x * w), int(lm[end_idx].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), skeleton_color, 2)

        # Точки суставов для наглядности
        for point in lm:
            cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 255), -1)

        # Обратный отсчёт перед стартом записи — даёт пользователю время занять позу
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

        # Запись кадра в буфер
        elif recording and current_class:
            # Сохраняем сырые координаты ландмарков — для совместимости с другими скриптами
            row = {'pose_id': pose_id, 'pose': current_class}
            for i, name in enumerate(LANDMARK_NAMES):
                row[f'x_{name}'] = lm[i].x if i < len(lm) else np.nan
                row[f'y_{name}'] = lm[i].y if i < len(lm) else np.nan
                row[f'z_{name}'] = lm[i].z if i < len(lm) else np.nan

            # Добавляем вычисленные биомеханические признаки с префиксом ref_
            ref_features = extract_reference_features(lm)
            if ref_features:
                for k, v in ref_features.items():
                    row[f'ref_{k}'] = v

            all_rows.append(row)
            pose_id += 1
            frames_recorded += 1

            # Прогресс-бар внизу кадра — визуальная обратная связь
            progress = int((frames_recorded / FRAMES_PER_CLASS) * (w - 20))
            cv2.rectangle(frame, (10, h-30), (w-10, h-10), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, h-30), (10 + progress, h-10), skeleton_color, -1)
            cv2.putText(frame, f'Recording {current_class}: {frames_recorded}/{FRAMES_PER_CLASS}',
                        (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, skeleton_color, 2)

            # Авто-остановка после набора нужного количества кадров
            if frames_recorded >= FRAMES_PER_CLASS:
                recording = False
                print(f'Записано {FRAMES_PER_CLASS} кадров для {current_class}')
                current_class = None

    # UI: полупрозрачная подложка для читаемости текста поверх видео
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Статус: сколько кадров уже записано по каждому классу
    recorded = {cls: sum(1 for r in all_rows if r['pose'] == cls) for cls in CLASSES.values()}
    status_parts = [f'{key}:{cls}({recorded[cls]})' for key, cls in CLASSES.items()]
    cv2.putText(frame, '  '.join(status_parts), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Подсказка по управлению, если не идёт запись и не активен отсчёт
    if not recording and countdown == 0:
        cv2.putText(frame, 'Press 1-2 to record | S to save | Q to quit',
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    cv2.imshow('Record Reference Form', frame)
    key = cv2.waitKey(1) & 0xFF

    # Обработка клавиш
    if key == ord('q'):
        break

    elif key == ord('s'):
        # Сохранение данных по нажатию S
        if all_rows:
            df = pd.DataFrame(all_rows)
            os.makedirs(r'Datasets', exist_ok=True)

            # Сохраняем отдельно для каждой фазы — так удобнее загружать в анализатор
            for cls in CLASSES.values():
                cls_df = df[df['pose'] == cls].copy()
                if len(cls_df) > 0:
                    filename = f'Datasets/reference_{cls}.csv'
                    cls_df.to_csv(filename, index=False)
                    print(f'📁 Сохранено {len(cls_df)} строк в {filename}')

                    # Печатаем примеры метрик, чтобы пользователь сразу видел, что записалось
                    ref_cols = [c for c in cls_df.columns if c.startswith('ref_')]
                    if ref_cols:
                        print(f'   Пример метрик ({cls}):')
                        for col in ['ref_avg_knee_angle', 'ref_trunk_lean', 'ref_stance_ratio']:
                            if col in cls_df.columns:
                                mean, std = cls_df[col].mean(), cls_df[col].std()
                                print(f'     {col[4:]:20s}: {mean:6.1f} ± {std:4.1f}')
            print('Все эталоны сохранены. Теперь запусти live_classifier.py')
        else:
            print(' Нет данных для сохранения')

    # Запуск записи по клавишам 1 или 2 с 3-секундной задержкой на подготовку
    elif not recording and countdown == 0:
        cls_key = chr(key) if key < 128 else None
        if cls_key in CLASSES:
            current_class = CLASSES[cls_key]
            countdown = 3
            countdown_start = time.time()
            print(f'Подготовка к записи: {current_class}...')

cap.release()
cv2.destroyAllWindows()