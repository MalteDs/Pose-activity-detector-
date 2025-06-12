import cv2
import mediapipe as mp
import time
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib  # si quieres guardar y cargar tu scaler
import math

# === CARGAR MODELO Y SCALER ===
model = load_model("./models/movimiento_classifier.h5")
scaler = joblib.load("./models/scaler.pkl")  # Asumiendo que lo guardaste antes

# === FUNCIONES DE FEATURES ===
def get_angle(a, b, c):
    """Calcula el ángulo entre tres puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    angle = np.arccos(np.clip(np.dot(ba, bc) /
                 (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    return np.degrees(angle)

def extract_features(landmarks):
    try:
        # Coordenadas clave
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        r_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
        r_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        r_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        shoulder_l = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        shoulder_r = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

        # Features
        angle_knee_left = get_angle(l_hip, l_knee, l_ankle)
        angle_knee_right = get_angle(r_hip, r_knee, r_ankle)
        angle_hip_left = get_angle(shoulder_l, l_hip, l_knee)
        angle_hip_right = get_angle(shoulder_r, r_hip, r_knee)
        trunk_inclination = get_angle(shoulder_l, l_hip, shoulder_r)
        shoulder_dist = math.dist(shoulder_l, shoulder_r)
        hip_dist = math.dist(l_hip, r_hip)

        return [angle_knee_left, angle_knee_right, angle_hip_left,
                angle_hip_right, trunk_inclination, shoulder_dist, hip_dist]
    except:
        return None
    
# === CONFIGURAR MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

# === CAPTURAR VIDEO ===
cap = cv2.VideoCapture(0)
start_time = time.time()
duration = 5  # segundos

features = []

print("🎥 Grabando... Realiza tu movimiento frente a la cámara.")
while time.time() - start_time < duration:
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        f = extract_features(results.pose_landmarks.landmark)
        if f:
            features.append(f)

    cv2.imshow("Grabando", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === PROMEDIAR FEATURES Y PREDECIR ===
if features:
    mean_features = np.mean(features, axis=0).reshape(1, -1)
    scaled_features = scaler.transform(mean_features)
    prediction = model.predict(scaled_features)
    predicted_class = np.argmax(prediction)

    print("✅ Movimiento detectado:", predicted_class)
else:
    print("⚠️ No se detectaron suficientes datos para clasificar.")