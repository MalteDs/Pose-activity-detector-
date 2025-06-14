# Pose Activity Detector

Sistema para detección y clasificación de actividades humanas basado en extracción de poses.

---

## 🔍 Descripción

Este proyecto detecta y clasifica actividades humanas en videos mediante extracción de poses y modelos de machine learning. Permite:

- Archivo principal de ejecución (`window.py`)
- Procesar múltiples videos para extraer secuencias de pose (`batch-pose-extractor.py`)
- Crear un conjunto de datos a partir de los datos de pose (`build_dataset.py`)
- Entrenar y evaluar modelos como redes neuronales, Random Forest y XGBoost
- Clasificar actividades (e.g. caminar hacia adelante, caminar hacia atrás, sentarse, levantarse, dar una vuelta) usando modelos guardados en `models/`

---

## 🛠️ Requisitos

- Python 3.7+
- pip

Paquetes principales (incluídos en `requirements.txt`):

---

## Integrantes

- Jose Alejandro Muñoz Cerón
- David Santiago Malte Puetate
- Samuel Ibarra Cano