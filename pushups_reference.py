import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from error_detection import extract_features_for_analysis


base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    running_mode=vision.RunningMode.VIDEO,
)
detector = vision.PoseLandmarker.create_from_options(options)

FRAMES_PER_CLASS = 300

CLASSES = {
    "1": "pushups_down",
    "2": "pushups_up",
}

COLORS = {
    "pushups_down": (0, 255, 100),
    "pushups_up": (0, 180, 255),
}

CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]

LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky_1", "right_pinky_1",
    "left_index_1", "right_index_1", "left_thumb_2", "right_thumb_2",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

IDX_MAP = {
    "nose": 0,
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}


def get_coords(lm):
    coords = {}
    for name, idx in IDX_MAP.items():
        if idx < len(lm):
            coords[name] = np.array([lm[idx].x, lm[idx].y, lm[idx].z])
        else:
            coords[name] = np.array([np.nan, np.nan, np.nan])
    return coords


def normalize(coords):
    l_hip, r_hip = coords["left_hip"][:2], coords["right_hip"][:2]
    center = (l_hip + r_hip) / 2
    l_sh, r_sh = coords["left_shoulder"][:2], coords["right_shoulder"][:2]
    scale = np.linalg.norm((l_sh + r_sh) / 2 - center) + 1e-6

    def n(key):
        return (coords[key][:2] - center) / scale

    return n, scale


def draw_skeleton(frame, lm, color):
    h, w, _ = frame.shape
    for s_idx, e_idx in CONNECTIONS:
        cv2.line(
            frame,
            (int(lm[s_idx].x * w), int(lm[s_idx].y * h)),
            (int(lm[e_idx].x * w), int(lm[e_idx].y * h)),
            color,
            2,
        )
    for point in lm:
        cv2.circle(frame, (int(point.x * w), int(point.y * h)), 4, (255, 255, 255), -1)


def save_rows(rows):
    if not rows:
        print("No data to save.")
        return

    df = pd.DataFrame(rows)
    os.makedirs("Datasets", exist_ok=True)

    for cls in CLASSES.values():
        cls_df = df[df["pose"] == cls].copy()
        if len(cls_df) == 0:
            continue

        filename = f"Datasets/reference_{cls}.csv"
        cls_df.to_csv(filename, index=False)
        print(f"Saved {len(cls_df)} rows to {filename}")

        print(f"Metrics ({cls}):")
        for col in [
            "ref_avg_elbow_angle",
            "ref_elbow_symmetry",
            "ref_hip_sag",
            "ref_hand_width_ratio",
            "ref_foot_width_ratio",
        ]:
            if col in cls_df.columns:
                valid = cls_df[col].dropna()
                if len(valid) > 0:
                    print(f"  {col[4:]:22s}: {valid.mean():7.3f} +/- {valid.std():6.3f}")


def main():
    all_rows = []
    pose_id = 0
    current_class = None
    recording = False
    frames_recorded = 0
    countdown = 0
    countdown_start = 0.0
    timestamp_counter = 0

    cap = cv2.VideoCapture(0)

    print("PUSH-UP REFERENCE RECORDING")
    print("Controls:")
    print("  1 - record pushups_down (bottom: chest near floor, plank straight)")
    print("  2 - record pushups_up   (top plank)")
    print("  S - save reference CSV files")
    print("  Q - quit")
    print(f"Frames per class: {FRAMES_PER_CLASS}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_counter += 33
        result = detector.detect_for_video(mp_image, timestamp_counter)
        skeleton_color = COLORS.get(current_class, (100, 100, 100))

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            draw_skeleton(frame, lm, skeleton_color)

            if countdown > 0:
                elapsed = time.time() - countdown_start
                remaining = countdown - int(elapsed)
                if remaining > 0:
                    cv2.putText(
                        frame,
                        f"Get ready: {remaining}",
                        (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2.0,
                        (0, 255, 255),
                        3,
                    )
                else:
                    countdown = 0
                    recording = True
                    frames_recorded = 0

            elif recording and current_class:
                row = {"pose_id": pose_id, "pose": current_class}
                for i, name in enumerate(LANDMARK_NAMES):
                    row[f"x_{name}"] = lm[i].x if i < len(lm) else np.nan
                    row[f"y_{name}"] = lm[i].y if i < len(lm) else np.nan
                    row[f"z_{name}"] = lm[i].z if i < len(lm) else np.nan

                ref_features = extract_features_for_analysis(lm, get_coords, normalize)
                if ref_features:
                    for key, value in ref_features.items():
                        if np.isfinite(value):
                            row[f"ref_{key}"] = value

                all_rows.append(row)
                pose_id += 1
                frames_recorded += 1

                if ref_features:
                    cv2.putText(
                        frame,
                        f"Hip sag: {ref_features.get('hip_sag', np.nan):.3f}",
                        (10, h - 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Hands: {ref_features.get('hand_width_ratio', np.nan):.2f}  "
                        f"Feet: {ref_features.get('foot_width_ratio', np.nan):.2f}",
                        (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                progress = int((frames_recorded / FRAMES_PER_CLASS) * (w - 20))
                cv2.rectangle(frame, (10, h - 30), (w - 10, h - 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, h - 30), (10 + progress, h - 10), skeleton_color, -1)
                cv2.putText(
                    frame,
                    f"Recording {current_class}: {frames_recorded}/{FRAMES_PER_CLASS}",
                    (10, h - 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    skeleton_color,
                    2,
                )

                if frames_recorded >= FRAMES_PER_CLASS:
                    recording = False
                    print(f"Recorded {FRAMES_PER_CLASS} frames for {current_class}")
                    current_class = None

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        recorded = {cls: sum(1 for row in all_rows if row["pose"] == cls) for cls in CLASSES.values()}
        status_parts = [f"{key}:{cls}({recorded[cls]})" for key, cls in CLASSES.items()]
        cv2.putText(frame, "  ".join(status_parts), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        if not recording and countdown == 0:
            cv2.putText(
                frame,
                "Press 1-2 to record | S to save | Q to quit",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (150, 150, 150),
                1,
            )

        cv2.imshow("Record Push-up Reference", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("s"):
            save_rows(all_rows)
        elif not recording and countdown == 0:
            cls_key = chr(key) if key < 128 else None
            if cls_key in CLASSES:
                current_class = CLASSES[cls_key]
                countdown = 3
                countdown_start = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
