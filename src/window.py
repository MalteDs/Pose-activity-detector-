import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import math
from tensorflow.keras.models import load_model
import joblib
import os
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "Neural-Network_classifier.h5")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")

if not os.path.exists(model_path):
    print(f"❌ El modelo no se encuentra en la ruta: {model_path}")
    exit()
if not os.path.exists(scaler_path):
    print(f"❌ El scaler no se encuentra en la ruta: {scaler_path}")
    exit()

model = load_model(model_path)
scaler = joblib.load(scaler_path)

# MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle

def extract_features(lm):
    ls = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    lh = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    lk = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    la = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    rs = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    rh = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    rk = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    ra = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

    return [
        calc_angle(lh, lk, la),
        calc_angle(rh, rk, ra),
        calc_angle(ls, lh, lk),
        calc_angle(rs, rh, rk),
        calc_angle(lh, ls, rs),
        np.linalg.norm(np.array(ls) - np.array(rs)),
        np.linalg.norm(np.array(lh) - np.array(rh)),
    ]

def get_predicted_class(n):
    return ["CAMINAR HACIA ADELANTE", "CAMINAR HACIA ATRÁS", "LEVANTARSE", "SENTARSE", "VUELTA"][n] if n < 5 else "DESCONOCIDO"

class RealTimePoseApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Detección de Movimiento en Tiempo Real")
        self.window.configure(bg="black")

        self.video_label = Label(window, bg="black")
        self.video_label.pack()

        self.movement_label = Label(window, text="Detectando movimiento...", font=("Helvetica", 24, "bold"),
                                    fg="white", bg="black")
        self.movement_label.pack(pady=10)

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

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        predicted_class = "Sin detección"
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            features = extract_features(lm)
            features_df = pd.DataFrame([features], columns=[
                'angle_knee_left', 'angle_knee_right', 'angle_hip_left',
                'angle_hip_right', 'trunk_inclination', 'shoulder_dist', 'hip_dist'
            ])
            scaled_features = scaler.transform(features_df)
            prediction = model.predict(scaled_features,verbose=0)
            predicted_label = np.argmax(prediction)
            predicted_class = get_predicted_class(predicted_label)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Mostrar movimiento en la interfaz
        self.movement_label.config(text=predicted_class)

        # Mostrar imagen en interfaz
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

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

run_gui()
