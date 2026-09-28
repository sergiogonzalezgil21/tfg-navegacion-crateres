"""
Paso 3. Ajuste fino sobre los renders de Blender.

No se entrena desde cero: se parte del modelo que ya reconoce crateres en
imagenes reales de la Luna y de Marte, y se le ensena el dominio nuevo. La
forma de un crater ya la sabe; lo que no ha visto nunca es un render.

Los volteos y las rotaciones van desactivados a proposito. Con sol rasante la
sombra de un crater cae siempre al mismo lado, y esa asimetria es justo la
senal que usa el detector: voltear la imagen crea una iluminacion que no
existe en ningun frame. Para un detector robusto a la direccion del sol habria
que subir fliplr y degrees y renderizar con varios azimuts.
"""

import os
import multiprocessing
from pathlib import Path

# =============================================================================

# --- Rutas relativas al propio fichero ------------------------------------
# El repositorio se puede clonar en cualquier carpeta, asi que ninguna ruta
# esta escrita a mano. __file__ no existe cuando el codigo se pega dentro de
# un editor (el de Blender, por ejemplo); en ese caso se usa el directorio de
# trabajo.
_AQUI = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.path.abspath(os.getcwd()))

RUTA_IA = _AQUI                                        # carpeta ia/
DATA_YAML = os.path.join(RUTA_IA, "dataset_blender", "data.yaml")

# Pesos de la etapa 1, el entrenamiento sobre imagenes reales de crateres. No
# viajan en el repositorio por tamano: se obtienen ejecutando los scripts de
# etapa1/, o se indica donde estan con la variable de entorno
# TFG_PESOS_ETAPA1. Si no aparecen, se parte de los pesos publicos de
# Ultralytics, que es la ultima entrada de la lista.
RUTA_ETAPA1 = os.environ.get("TFG_PESOS_ETAPA1",
                             os.path.join(_AQUI, "pesos_etapa1"))

# Modelo de partida. El primero que exista de la lista.
MODELOS_BASE = [
    os.path.join(RUTA_ETAPA1, "yolov8_crater-10", "weights", "best.pt"),
    os.path.join(RUTA_ETAPA1, "yolov8_crater-8",  "weights", "best.pt"),
    "yolov8s.pt",
]

EPOCAS = 60
IMGSZ = 1024      # 1920x1080 reescalado. A 640 los crateres pequenos se
                  # quedan en 4 px y son indetectables; a 1024 pasan a 6-7 y
                  # los de referencia rondan los 145 px.
BATCH = 4         # con imgsz 1024 no cabe mucho mas en una GPU normal
PACIENCIA = 20

# =============================================================================


def elegir_base():
    for m in MODELOS_BASE:
        if m.endswith(".pt") and (os.path.isfile(m) or "/" not in m and "\\" not in m):
            if os.path.isfile(m):
                print(f"Modelo de partida: {m}")
                return m
    print(f"Ninguno de los modelos entrenados existe. Uso {MODELOS_BASE[-1]}")
    return MODELOS_BASE[-1]


def main():
    from ultralytics import YOLO

    if not os.path.isfile(DATA_YAML):
        raise RuntimeError(f"Falta {DATA_YAML}. Ejecuta antes 02_preparar_dataset.py")

    model = YOLO(elegir_base())

    model.train(
        data=DATA_YAML,
        epochs=EPOCAS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        workers=0,
        patience=PACIENCIA,
        project=os.path.join(RUTA_IA, "runs"),
        name="finetune_blender",
        exist_ok=True,
        save=True,
        single_cls=True,

        # --- aumentacion ---
        fliplr=0.0,        # ver la nota de arriba sobre la direccion del sol
        flipud=0.0,
        degrees=0.0,
        translate=0.10,
        scale=0.40,        # simula variacion de altura
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,         # las imagenes son en escala de grises
        hsv_s=0.0,
        hsv_v=0.25,        # variacion de exposicion
        mosaic=1.0,
        close_mosaic=10,
        erasing=0.0,
    )

    print("=" * 70)
    print("Entrenamiento terminado.")
    print(f"Pesos: {os.path.join(RUTA_IA, 'runs', 'finetune_blender', 'weights', 'best.pt')}")
    print("Siguiente: python 04_evaluar.py")
    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
