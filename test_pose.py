import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(
    model_asset_path='pose_landmarker_full.task'
)

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO,  
)

detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр с камеры")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.pose_landmarks:
        for pose_landmarks in detection_result.pose_landmarks:
            h, w, _ = frame.shape
            for landmark in pose_landmarks:
                # Mediapipe возвращает не в пикселях, а от 0 - 1, запомни
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)

            # Рисуем соединения между точками
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # руки
                (11, 23), (12, 24), (23, 24),                        # торс
                (23, 25), (25, 27), (24, 26), (26, 28),              # ноги
            ]
            for start_idx, end_idx in connections:
                x1 = int(pose_landmarks[start_idx].x * w) #Начало
                y1 = int(pose_landmarks[start_idx].y * h)
                x2 = int(pose_landmarks[end_idx].x * w) #Конец
                y2 = int(pose_landmarks[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()