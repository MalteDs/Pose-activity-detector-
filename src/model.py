import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Cargar dataset
df = pd.read_csv("./data/dataset_ready/features.csv")
print("✅ Dataset cargado:", df.shape)

# Features y etiquetas
X = df[['angle_knee_left', 'angle_knee_right', 'angle_hip_left', 'angle_hip_right',
        'trunk_inclination', 'shoulder_dist', 'hip_dist']]
y = df['label']

# Normalizar (opcional, pero mejora el entrenamiento)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en train y test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print("Tamaño de entrenamiento:", X_train.shape)
print("Tamaño de prueba:", X_test.shape)

# Convertir etiquetas a one-hot encoding (opcional, pero útil)
num_classes = len(df['label'].unique())
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Definir modelo
model = Sequential()
model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
