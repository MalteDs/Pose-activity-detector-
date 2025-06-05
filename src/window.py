import tkinter as tk
from tkinter import Label
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
import math

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calc_angle(a, b, c):
    """Calcula el ángulo entre tres puntos."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

class RealTimePoseApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Análisis de Actividad en Tiempo Real")

        self.video_label = Label(window)
        self.video_label.pack()

        self.activity_label = Label(window, text="Actividad: ", font=("Arial", 16))
        self.activity_label.pack()

        self.angles_label = Label(window, text="Ángulos:", font=("Arial", 12), justify="left")
        self.angles_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.running = True

        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.update_video()

    def infer_activity(self, knee_angle, hip_angle, inclination):
        """Clasifica la actividad basada en reglas simples (puedes reemplazar con modelo)."""
        if knee_angle < 100 and hip_angle < 100:
            return "Sentadilla"
        elif inclination > 15:
            return "Inclinación lateral"
        else:
            return "De pie"

    def update_video(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)

        knee_angle = hip_angle = inclination = 0

        if results.pose_landmarks:
            # Obtener puntos clave
            lm = results.pose_landmarks.landmark

            # Coordenadas
            shoulder = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            hip = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Ángulos
            knee_angle = calc_angle(hip, knee, ankle)
            hip_angle = calc_angle(shoulder, hip, knee)

            # Inclinación del tronco
            shoulder_right = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            inclination = abs(shoulder[0] - shoulder_right[0]) * 100  # Aproximación de inclinación lateral

            # Dibujar puntos
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Clasificación
        activity = self.infer_activity(knee_angle, hip_angle, inclination)

        # Mostrar en GUI
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.activity_label.config(text=f"Actividad detectada: {activity}")
        self.angles_label.config(text=f"Ángulo rodilla: {knee_angle:.1f}°\nÁngulo cadera: {hip_angle:.1f}°\nInclinación: {inclination:.1f}")

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
