# eda_utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    print("✅ Dataset cargado con shape:", df.shape)
    return df

def load_class_map(json_path):
    with open(json_path) as f:
        class_map = json.load(f)
    inv_class_map = {v: k for k, v in class_map.items()}
    return class_map, inv_class_map

def plot_class_distribution(df):
    df['class_name'].value_counts().plot(kind='bar', title="Distribución de clases")
    plt.xlabel("Clase")
    plt.ylabel("Cantidad de frames")
    plt.show()

def clean_dataset(df, vis_threshold=0.3):
    print("➡️  Eliminando valores nulos...")
    df_clean = df.dropna()
    print("Filas eliminadas por valores nulos:", len(df) - len(df_clean))

    # Calcular visibilidad promedio por fila
    vis_cols = [f'v{i}' for i in range(33)]
    df_clean['v_mean'] = df_clean[vis_cols].mean(axis=1)

    # Filtrar frames con baja visibilidad
    df_clean = df_clean[df_clean['v_mean'] >= vis_threshold]
    print("Filas eliminadas por visibilidad < 0.3:", len(df) - len(df_clean))

    # Filtro por coordenadas fuera de [0, 1]
    coord_cols = [col for col in df_clean.columns if col.startswith(('x', 'y'))]
    before = len(df_clean)
    df_clean = df_clean[(df_clean[coord_cols] >= 0.0).all(axis=1) & (df_clean[coord_cols] <= 1.0).all(axis=1)]
    after = len(df_clean)
    print(f"Filas eliminadas por coordenadas fuera de rango: {before - after}")

    return df_clean

def plot_shoulder_coordinates(df):
    sns.scatterplot(x=df['x11'], y=df['y11'], alpha=0.5, label='Hombro Izquierdo')
    sns.scatterplot(x=df['x12'], y=df['y12'], alpha=0.5, label='Hombro Derecho')
    plt.title("Distribución de coordenadas de hombros")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

def plot_landmarks(df, frame_index):
    coords = []
    row = df.iloc[frame_index]
    for i in range(33):
        x, y = row[f'x{i}'], row[f'y{i}']
        coords.append((x, y))
    coords = np.array(coords)

    plt.figure(figsize=(6, 6))
    plt.scatter(coords[:, 0], coords[:, 1])
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=9)
    plt.gca().invert_yaxis()
    plt.title(f"Landmarks del frame {frame_index} ({row['class_name']})")
    plt.grid(True)
    plt.show()

def save_clean_dataset(df, output_path):
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset limpio guardado en: {output_path}")
