import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier



landmarks = pd.read_csv(r'Datasets\landmarks.csv')
labels    = pd.read_csv(r'Datasets\labels.csv')

df = pd.merge(landmarks, labels, on='pose_id')

features = pd.DataFrame()
features['pose_id'] = df['pose_id']
features['pose']    = df['pose']

def angle(row, a, b, c):
    ax, ay = row[f'x_{a}'], row[f'y_{a}']
    bx, by = row[f'x_{b}'], row[f'y_{b}']
    cx, cy = row[f'x_{c}'], row[f'y_{c}']
    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))



# Углы суставов
features['left_elbow_angle']     = df.apply(lambda r: angle(r, 'left_shoulder',  'left_elbow',  'left_wrist'),   axis=1)
features['right_elbow_angle']    = df.apply(lambda r: angle(r, 'right_shoulder', 'right_elbow', 'right_wrist'),  axis=1)
features['left_shoulder_angle']  = df.apply(lambda r: angle(r, 'left_elbow',     'left_shoulder',  'left_hip'),  axis=1)
features['right_shoulder_angle'] = df.apply(lambda r: angle(r, 'right_elbow',    'right_shoulder', 'right_hip'), axis=1)
features['left_knee_angle']      = df.apply(lambda r: angle(r, 'left_hip',   'left_knee',   'left_ankle'),  axis=1)
features['right_knee_angle']     = df.apply(lambda r: angle(r, 'right_hip',  'right_knee',  'right_ankle'), axis=1)
features['left_hip_angle']       = df.apply(lambda r: angle(r, 'left_shoulder',  'left_hip',  'left_knee'),  axis=1)
features['right_hip_angle']      = df.apply(lambda r: angle(r, 'right_shoulder', 'right_hip', 'right_knee'), axis=1)
features['left_ankle_angle']     = df.apply(lambda r: angle(r, 'left_knee',  'left_ankle',  'left_foot_index'),  axis=1)
features['right_ankle_angle']    = df.apply(lambda r: angle(r, 'right_knee', 'right_ankle', 'right_foot_index'), axis=1)

# Туловище и голова
features['trunk_lean'] = df.apply(lambda r: np.degrees(np.arctan2(
    abs((r['x_left_shoulder'] + r['x_right_shoulder']) / 2 - (r['x_left_hip'] + r['x_right_hip']) / 2),
    abs((r['y_left_shoulder'] + r['y_right_shoulder']) / 2 - (r['y_left_hip'] + r['y_right_hip']) / 2) + 1e-6)), axis=1)

features['head_tilt'] = df.apply(lambda r: np.degrees(np.arctan2(
    abs(r['x_nose'] - (r['x_left_shoulder'] + r['x_right_shoulder']) / 2),
    abs(r['y_nose'] - (r['y_left_shoulder'] + r['y_right_shoulder']) / 2) + 1e-6)), axis=1)

# Симметрия и геометрия
features['knee_symmetry']      = abs(features['left_knee_angle'] - features['right_knee_angle'])
features['elbow_symmetry']     = abs(features['left_elbow_angle'] - features['right_elbow_angle'])
features['shoulder_symmetry']  = abs(features['left_shoulder_angle'] - features['right_shoulder_angle'])
features['hip_symmetry']       = abs(features['left_hip_angle'] - features['right_hip_angle'])

features['feet_width']             = df.apply(lambda r: abs(r['x_left_ankle'] - r['x_right_ankle']), axis=1)
features['shoulder_width']         = df.apply(lambda r: abs(r['x_left_shoulder'] - r['x_right_shoulder']), axis=1)
features['feet_to_shoulder_ratio'] = features['feet_width'] / (features['shoulder_width'] + 1e-6)
features['hip_height']             = df.apply(lambda r: (r['y_left_hip'] + r['y_right_hip']) / 2 - (r['y_left_shoulder'] + r['y_right_shoulder']) / 2, axis=1)


print(f'Загружено строк: {len(features)}')
print(f'Статических признаков: 20')



angle_cols = [
    'left_elbow_angle', 'right_elbow_angle',
    'left_shoulder_angle', 'right_shoulder_angle',
    'left_knee_angle', 'right_knee_angle',
    'left_hip_angle', 'right_hip_angle',
    'trunk_lean', 'hip_height'
]

for col in angle_cols:
    vel_col = f'vel_{col}'
    diff = features[col].diff()
    class_changed = features['pose'] != features['pose'].shift(1)
    diff[class_changed] = 0.0
    features[vel_col] = diff

print(f'Временных признаков добавлено: 10')
print(f'Итого признаков: {len(features.columns) - 2}')

X = features.drop(['pose_id', 'pose'], axis=1)
Y = features['pose']

# Кодируем метки
le = LabelEncoder()
Y_encoded = le.fit_transform(Y)
joblib.dump(le, 'label_encoder.pkl')

# Нормализация
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'angle_scaler_v4.pkl')

X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=Y_encoded
)



model = XGBClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=50,          
    use_label_encoder=False,
    n_jobs=-1
)

print("\nОбучаем XGBoost...")

model.fit(
    X_train, Y_train,
    eval_set=[(X_test, Y_test)],
    verbose=False
)


Y_pred = model.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

print(f'Accuracy: {acc:.4f} ({acc*100:.1f}%)')
print(classification_report(Y_test, Y_pred, target_names=le.classes_))

joblib.dump(model, 'exercise_classifier_xgboost.pkl')
print('\nМодель сохранена: exercise_classifier_xgboost.pkl')


# Важность признаков
importances = pd.Series(model.feature_importances_, index=X.columns)
print('\nТоп-15 важных признаков:')
print(importances.sort_values(ascending=False).head(15).round(4))

print('\nВажность временных признаков (vel_*):')
vel_importance = importances[importances.index.str.startswith('vel_')]
print(vel_importance.sort_values(ascending=False).round(4))