import cv2
import mediapipe as mp
import pandas as pd
import os

# Rutas
INPUT_DIR = 'data/original_videos'
OUTPUT_DIR = 'data/skeleton_csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def extract_pose(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    landmarks_data = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            row = [frame_idx]
            for lm in results.pose_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z, lm.visibility])
            landmarks_data.append(row)

        frame_idx += 1

    cap.release()

    if not landmarks_data:
        print(f"⚠️ Sin landmarks detectados en {video_path}")
        return

    # Crear encabezado
    cols = ['frame']
    for i in range(33):  # 33 puntos en MediaPipe Pose
        cols += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

    df = pd.DataFrame(landmarks_data, columns=cols)
    df.to_csv(output_csv, index=False)
    print(f"✅ CSV guardado: {output_csv}")

# Procesamiento por lote
for filename in os.listdir(INPUT_DIR):
    if filename.endswith('.mp4'):
        input_path = os.path.join(INPUT_DIR, filename)
        output_csv = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + '.csv')
        extract_pose(input_path, output_csv)

pose.close()
