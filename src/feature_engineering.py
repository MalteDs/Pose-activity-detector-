import pandas as pd
import numpy as np
import os

def calc_angle(a, b, c):
    """
    Calcula el ángulo entre tres puntos: a (proximal), b (vértice), c (distal)
    Devuelve el ángulo en grados.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def extract_features_from_row(row):
    """
    Extrae características biomecánicas de una fila con coordenadas de pose.
    """
    features = {}

    # Puntos clave
    hip_left = [row['x23'], row['y23']]
    knee_left = [row['x25'], row['y25']]
    ankle_left = [row['x27'], row['y27']]

    hip_right = [row['x24'], row['y24']]
    knee_right = [row['x26'], row['y26']]
    ankle_right = [row['x28'], row['y28']]

    shoulder_left = [row['x11'], row['y11']]
    shoulder_right = [row['x12'], row['y12']]

    # Ángulos
    features['angle_knee_left'] = calc_angle(hip_left, knee_left, ankle_left)
    features['angle_knee_right'] = calc_angle(hip_right, knee_right, ankle_right)
    features['angle_hip_left'] = calc_angle(shoulder_left, hip_left, knee_left)
    features['angle_hip_right'] = calc_angle(shoulder_right, hip_right, knee_right)

    # Inclinación del tronco (hombros vs caderas)
    trunk_vector = np.array(shoulder_right) + np.array(shoulder_left) - np.array(hip_right) - np.array(hip_left)
    features['trunk_inclination'] = np.arctan2(trunk_vector[1], trunk_vector[0]) * 180 / np.pi

    # Distancias
    features['shoulder_dist'] = np.linalg.norm(np.array(shoulder_left) - np.array(shoulder_right))
    features['hip_dist'] = np.linalg.norm(np.array(hip_left) - np.array(hip_right))

    return features


def generate_feature_dataset(input_path="./data/dataset_ready/movimientos_limpio.csv", save_path="./data/dataset_ready/features.csv"):
    """
    Genera un DataFrame de características a partir de un CSV limpio.
    """
    df = pd.read_csv(input_path)
    print("✅ Dataset limpio cargado:", df.shape)

    feature_rows = []
    for idx, row in df.iterrows():
        feats = extract_features_from_row(row)
        feats['label'] = row['label']
        feats['class_name'] = row['class_name']
        feature_rows.append(feats)

    df_features = pd.DataFrame(feature_rows)
    print("✅ Dataset de características generado:", df_features.shape)

    df_features.to_csv(save_path, index=False)
    print(f"Características guardadas en: {save_path}")

    return df_features
