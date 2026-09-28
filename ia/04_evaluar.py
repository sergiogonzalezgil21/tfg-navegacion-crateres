"""
Paso 4. Compara el modelo de partida con el afinado, sobre el mismo conjunto
de validacion, separado del de entrenamiento por 200 km de terreno.

Ademas del mAP, que es la metrica estandar y hay que darla, mide el porcentaje
de frames con tres o mas crateres detectados: el emparejamiento por tripletas
necesita tres, y sin ellos la geometria ni llega a intervenir.
"""

import os
import glob

import numpy as np

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
DIR_VAL_IMG = os.path.join(RUTA_IA, "dataset_blender", "val", "images")
DIR_VAL_LAB = os.path.join(RUTA_IA, "dataset_blender", "val", "labels")

# Pesos de la etapa 1, el entrenamiento sobre imagenes reales de crateres. No
# viajan en el repositorio por tamano: se obtienen ejecutando los scripts de
# etapa1/, o se indica donde estan con la variable de entorno
# TFG_PESOS_ETAPA1. Si no aparecen, se parte de los pesos publicos de
# Ultralytics, que es la ultima entrada de la lista.
RUTA_ETAPA1 = os.environ.get("TFG_PESOS_ETAPA1",
                             os.path.join(_AQUI, "pesos_etapa1"))

MODELOS = {
    "viejo (imagenes reales)":
        os.path.join(RUTA_ETAPA1, "yolov8_crater-10", "weights", "best.pt"),
    "viejo (segunda variante)":
        os.path.join(RUTA_ETAPA1, "yolov8_crater-8",  "weights", "best.pt"),
    "afinado con Blender":
        os.path.join(RUTA_IA, "runs", "finetune_blender", "weights", "best.pt"),
}

CONF = 0.25
IMGSZ = 1024

# =============================================================================


def n_reales():
    cuentas = {}
    for t in glob.glob(os.path.join(DIR_VAL_LAB, "*.txt")):
        base = os.path.splitext(os.path.basename(t))[0]
        with open(t, encoding="utf-8") as f:
            cuentas[base] = sum(1 for l in f if l.strip())
    return cuentas


def evaluar(nombre, pesos, reales):
    from ultralytics import YOLO

    if not os.path.isfile(pesos):
        print(f"\n[{nombre}] no existe: {pesos}")
        return None

    print(f"\n[{nombre}]")
    model = YOLO(pesos)

    # --- metricas estandar ---
    try:
        m = model.val(data=DATA_YAML, imgsz=IMGSZ, split="val",
                      verbose=False, plots=False)
        mAP50 = float(m.box.map50)
        mAP = float(m.box.map)
        rec = float(m.box.mr)
        pre = float(m.box.mp)
        print(f"  mAP50 {mAP50:.3f}   mAP50-95 {mAP:.3f}   "
              f"precision {pre:.3f}   recall {rec:.3f}")
    except Exception as e:
        mAP50 = mAP = rec = pre = float("nan")
        print(f"  (no he podido calcular el mAP: {e})")

    # --- la metrica que de verdad importa ---
    imgs = sorted(glob.glob(os.path.join(DIR_VAL_IMG, "*.png")))
    detectados = []
    for i in range(0, len(imgs), 16):
        lote = imgs[i:i + 16]
        for r in model.predict(lote, imgsz=IMGSZ, conf=CONF, verbose=False):
            detectados.append(len(r.boxes))

    det = np.array(detectados)
    gt = np.array([reales.get(os.path.splitext(os.path.basename(p))[0], 0)
                   for p in imgs])

    con3 = 100.0 * (det >= 3).mean()
    print(f"  crateres detectados por frame: media {det.mean():.1f} "
          f"(verdad: {gt.mean():.1f})")
    print(f"  frames con 3 o mas crateres: {con3:.1f} %")
    print(f"  frames con 0 crateres: {100.0*(det==0).mean():.1f} %")

    return dict(nombre=nombre, mAP50=mAP50, mAP=mAP, precision=pre, recall=rec,
                media_det=float(det.mean()), media_gt=float(gt.mean()),
                pct_con3=con3)


def main():
    reales = n_reales()
    if not reales:
        raise RuntimeError(f"No hay etiquetas en {DIR_VAL_LAB}")
    print(f"Validacion: {len(reales)} frames, "
          f"{np.mean(list(reales.values())):.1f} crateres reales por frame")

    filas = [r for r in (evaluar(n, p, reales) for n, p in MODELOS.items()) if r]

    print("\n" + "=" * 78)
    print(f"{'modelo':<26} {'mAP50':>7} {'recall':>7} {'det/frame':>10} {'>=3 crat':>10}")
    print("-" * 78)
    for r in filas:
        print(f"{r['nombre']:<26} {r['mAP50']:>7.3f} {r['recall']:>7.3f} "
              f"{r['media_det']:>10.1f} {r['pct_con3']:>9.1f}%")
    if filas:
        print(f"{'verdad (Blender)':<26} {'-':>7} {'-':>7} "
              f"{filas[0]['media_gt']:>10.1f} {'100.0%':>10}")
    print("=" * 78)

    csv_out = os.path.join(RUTA_IA, "comparativa_modelos.csv")
    import csv as _csv
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"\nTabla guardada en {csv_out}")


if __name__ == "__main__":
    main()
