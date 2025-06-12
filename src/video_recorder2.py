import cv2
import time
import os

# Crear carpeta si no existe
output_folder = "src/videos"
os.makedirs(output_folder, exist_ok=True)

# Parámetros de video
video_name = "grabacion_5_segundos.mp4"  # extensión mp4
output_path = os.path.join(output_folder, video_name)
duration_seconds = 5
fps = 20.0
frame_width = 640
frame_height = 480

# Inicializar cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

# Codec para .mp4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

print("Grabando video...")
start_time = time.time()

while time.time() - start_time < duration_seconds:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

print(f"Video guardado en: {output_path}")

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
