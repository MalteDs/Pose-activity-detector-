import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import joblib
import os
import pandas as pd
import sys
import traceback
from xgboost import XGBClassifier


base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "xgboost_movimiento.json")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
# Cargar modelo XGBoost y scaler
model = XGBClassifier()
model.load_model(model_path)
scaler = joblib.load(scaler_path)

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def extract_features(lm):
    left_shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    left_hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    right_shoulder = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    right_hip = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    right_knee = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    right_ankle = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    angle_knee_left = calc_angle(left_hip, left_knee, left_ankle)
    angle_knee_right = calc_angle(right_hip, right_knee, right_ankle)
    angle_hip_left = calc_angle(left_shoulder, left_hip, left_knee)
    angle_hip_right = calc_angle(right_shoulder, right_hip, right_knee)
    trunk_inclination = calc_angle(left_hip, left_shoulder, right_shoulder)
    shoulder_dist = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    hip_dist = np.linalg.norm(np.array(left_hip) - np.array(right_hip))

    return [angle_knee_left, angle_knee_right, angle_hip_left, angle_hip_right, trunk_inclination, shoulder_dist, hip_dist]

def get_predicted_class(n):
    if n == 0:
        return "CAMINAR HACIA ADELANTE"
    elif n == 1:
        return "CAMINAR HACIA ATRÁS"
    elif n == 2:
        return "LEVANTARSE"
    elif n == 3:
        return "SENTARSE"
    elif n == 4:
        return "VUELTA"
    else:
        return "DESCONOCIDO"

class RealTimePoseApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Features de Pose en Tiempo Real")

        self.video_label = Label(window)
        self.video_label.pack()

        self.features_label = Label(window, text="Features:", font=("Arial", 12), justify="left")
        self.features_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.update_video()

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        features_text = "Sin detección"
        predicted_class = "N/A"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            features = extract_features(lm)

            feature_names = ['angle_knee_left', 'angle_knee_right', 'angle_hip_left',
                 'angle_hip_right', 'trunk_inclination', 'shoulder_dist', 'hip_dist']
            features_df = pd.DataFrame([features], columns=feature_names)
            features_scaled = scaler.transform(features_df)

            prediction = model.predict(features_scaled)
            predicted_label = int(prediction[0])
            predicted_class = get_predicted_class(predicted_label)

            features_text = (
                f"angle_knee_left: {features[0]:.2f}\n"
                f"angle_knee_right: {features[1]:.2f}\n"
                f"angle_hip_left: {features[2]:.2f}\n"
                f"angle_hip_right: {features[3]:.2f}\n"
                f"trunk_inclination: {features[4]:.2f}\n"
                f"shoulder_dist: {features[5]:.4f}\n"
                f"hip_dist: {features[6]:.4f}\n"
                f"🔍 Predicted Movement: {predicted_class}"
            )
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.putText(frame,
                        f"{predicted_class}",
                        (50, 400),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        4,
                        cv2.LINE_AA)

        # Mostrar en GUI
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.features_label.config(text=features_text)

        self.window.after(10, self.update_video)

    def stop(self):
        self.running = False
        self.cap.release()
        self.window.destroy()

def run_gui():
    root = tk.Tk()
    app = RealTimePoseApp(root)
    root.protocol("WM_DELETE_WINDOW", app.stop)
    root.mainloop()

def excepthook(type, value, tb):
    print("Ocurrió un error no manejado:")
    traceback.print_exception(type, value, tb)
    input("Presiona Enter para salir...")

sys.excepthook = excepthook

try:
    run_gui()
except Exception as e:
    import traceback
    print("Ocurrió un error:", e)
    traceback.print_exc()
    input("Presiona Enter para salir...")