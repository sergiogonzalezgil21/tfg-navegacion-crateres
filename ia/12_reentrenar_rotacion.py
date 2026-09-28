"""
Paso 12. Reentrenamiento para que el detector aguante la camara girada.

Sobre el vuelo de descenso, el detector entrenado con degrees = 0.0 se
desploma con el giro: 13.1 cajas por frame entre 20 y 45 grados, 0.89 entre
135 y 180, y un 36 % de frames sin ninguna deteccion cuando las etiquetas
dicen que hay mas de 400 crateres visibles. La correlacion entre giro y numero
de detecciones es -0.693. Controlando por giro, el efecto de la altura es
mucho mas suave, asi que el fallo dominante es la rotacion y mientras siga ahi
el limite por densidad de landmarks queda tapado.

La causa es el sol fijo en el mundo: al girar la camara la sombra cambia de
lado dentro de la imagen, y un crater con la sombra al lado equivocado parece
un monticulo.

La correccion es degrees = 180, que gira cada imagen de entrenamiento un
angulo al azar y obliga a la red a apoyarse en la forma en vez de en una
direccion de luz fija; se suben tambien fliplr y flipud. Los datos no cambian:
sigue siendo el conjunto del vuelo de crucero, asi que el descenso queda como
prueba ajena y la mejora es atribuible solo a la aumentacion. Se parte del
modelo ya afinado con Blender.

El recall a 0 grados puede bajar un poco; lo que tiene que subir es el de 90 y
180. Se comprueba con 10_robustez_rotacion.py antes y despues.
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
    # se parte del modelo ya afinado con Blender: solo hay que ensenarle
    # invariancia a la rotacion, no a reconocer crateres desde cero
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "runs", "finetune_blender", "weights", "best.pt"),
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
        name="finetune_rotacion",
        exist_ok=True,
        save=True,
        single_cls=True,

        # --- aumentacion ---
        # LA CORRECCION DE LA FASE 4. Antes estaban los tres a 0 para no
        # romper la coherencia con la direccion del sol. Ese razonamiento era
        # correcto para un vuelo sin guinada y equivocado en cuanto la camara
        # gira: lo que hacia falta era justo lo contrario.
        fliplr=0.5,
        flipud=0.5,
        degrees=180.0,
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
    print(f"Pesos: {os.path.join(RUTA_IA, 'runs', 'finetune_rotacion', 'weights', 'best.pt')}")
    print("Siguiente: python 04_evaluar.py")
    print("=" * 70)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
