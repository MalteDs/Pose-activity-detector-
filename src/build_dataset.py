import os
import pandas as pd
import json

# Rutas
INPUT_DIR = 'data/skeleton_csv'
CLASS_MAP_PATH = 'data/metadata/class_map.json'
OUTPUT_PATH = 'data/dataset_ready/movimientos.csv'
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cargar el mapeo de clases
with open(CLASS_MAP_PATH, 'r') as f:
    class_map = json.load(f)

# Lista para guardar todos los DataFrames
all_dataframes = []

# Recorrer cada clase (subcarpetas)
for class_name in sorted(os.listdir(INPUT_DIR)):
    class_path = os.path.join(INPUT_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    class_label = class_map.get(class_name)
    if class_label is None:
        print(f"⚠️ Clase {class_name} no está en class_map.json, ignorada.")
        continue

    for filename in os.listdir(class_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(class_path, filename)
            df = pd.read_csv(file_path)

            # Añadir columna de etiqueta y nombre de clase
            df['label'] = class_label
            df['class_name'] = class_name

            all_dataframes.append(df)

print(f"📊 Se encontraron {len(all_dataframes)} archivos CSV para unificar.")

# Unir todos los dataframes
final_df = pd.concat(all_dataframes, ignore_index=True)

# Guardar CSV final
final_df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Dataset final guardado en: {OUTPUT_PATH}")
