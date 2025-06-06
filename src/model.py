import pandas as pd
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten

# Carpeta de CSVs
csv_dir = 'data/skeleton_csv'
X = []
y = []

with open('data/metadata/class_map.json', 'r') as f:
    class_map = json.load(f)

for class_name, label in class_map.items():
    class_dir = os.path.join(csv_dir, class_name)
    for csv_file in os.listdir(class_dir):
        df = pd.read_csv(os.path.join(class_dir, csv_file))
        # Ejemplo: solo usa (x, y, z) de cada landmark
        frame_data = df.iloc[:, 1:].values.reshape(-1, 33, 4)[:, :, :3]  # (frames, 33, 3)
        # Opcional: recorta o pad a 30 frames
        max_frames = 30
        if frame_data.shape[0] < max_frames:
            pad = np.zeros((max_frames - frame_data.shape[0], 33, 3))
            frame_data = np.concatenate([frame_data, pad])
        else:
            frame_data = frame_data[:max_frames]
        X.append(frame_data)
        y.append(label)

X = np.array(X)  # (videos, frames, landmarks, features)
y = np.array(y)




model = Sequential([
    LSTM(64, input_shape=(30, 33*3)),  # aplana landmarks
    Dense(32, activation='relu'),
    Dense(len(class_map), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Entrena el modelo
X_train = X.reshape(X.shape[0], 30, 33*3)
model.fit(X_train, y, epochs=20, batch_size=8, validation_split=0.2)
model.save('mi_modelo_lstm.h5')
