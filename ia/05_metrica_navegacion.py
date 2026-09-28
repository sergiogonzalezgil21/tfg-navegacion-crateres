"""
Paso 5. La metrica que decide de verdad.

La columna ">=3 crateres" del paso 4 se satura: sale al 100 % para todos los
modelos. El motivo es que se etiquetan todos los crateres visibles, unos 423
por frame y la mayoria de menos de 2 km, y encontrar 3 de 423 lo consigue
cualquiera. Pero al emparejador no le sirven esos: el mapa embarcado solo
lleva los crateres de D >= 16 km.

Lo que hay que medir es cuantos crateres de escala de mapa detecta por frame,
y en que porcentaje de frames llega a tres.

Para pasar una caja a kilometros: con FOV horizontal de 60 grados la huella a
altura h mide 2*h*tan(30), asi que D_km = ancho_normalizado * 2 * h * tan(30).
La altura verdadera sale de trayectoria_real.csv. En vuelo habria que
estimarla, pero para evaluar el detector usar la verdad es lo correcto: separa
su error del de la geometria.
"""

import os
import csv
import glob
import math

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
RUTA_BLENDER = os.path.join(os.path.dirname(_AQUI), "blender")

DIR_VAL_IMG = os.path.join(RUTA_IA, "dataset_blender", "val", "images")
DIR_VAL_LAB = os.path.join(RUTA_IA, "dataset_blender", "val", "labels")
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")

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

D_MIN_MAPA_KM = 16.0     # el mismo umbral con el que se construye mapa_yolo.txt
FOV_HORIZONTAL = 60.0
IMGSZ = 1024

# max_det por defecto en Ultralytics son 300, y con 423 cajas reales por frame
# eso limita el recall a 300/423 = 0.709 por construccion. Se sube para que la
# medida sea del modelo y no del tope.
MAX_DET = 1500

CONFS = [0.25, 0.10, 0.05]    # se prueba mas de un punto de operacion

# =============================================================================


def alturas():
    d = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[f"{int(r['frame']):04d}"] = float(r["altura_km"])
    return d


def gt_grandes(base, alt):
    """Crateres reales de escala de mapa en un frame: (cx, cy, D_km)."""
    p = os.path.join(DIR_VAL_LAB, base + ".txt")
    fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    out = []
    if not os.path.isfile(p):
        return out
    with open(p, encoding="utf-8") as f:
        for l in f:
            v = l.split()
            if len(v) < 5:
                continue
            cx, cy, bw = float(v[1]), float(v[2]), float(v[3])
            D = bw * fac
            if D >= D_MIN_MAPA_KM:
                out.append((cx, cy, D))
    return out


TOL_NORM = 0.03     # 3 % del ancho de imagen: unos 58 px en 1920


def emparejar(det, gt):
    """Cuenta cuantos crateres reales quedan cubiertos por alguna deteccion.

    Se empareja por cercania del centro con una tolerancia fija, no por IoU.
    Para lo que aqui importa (saber si el crater se ha visto y donde esta) es
    mas informativo: el IoU penaliza que la caja sea algo mas grande o mas
    pequena, y al matching por tripletas eso le da igual, porque solo usa los
    centros.

    Cada deteccion se puede usar una sola vez, para que dos crateres reales
    juntos no se den por buenos con una unica deteccion.
    """
    usadas = set()
    aciertos = 0
    for (gx, gy, _gD) in gt:
        mejor, mejor_d = None, 1e9
        for i, (dx, dy, _dD) in enumerate(det):
            if i in usadas:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < mejor_d:
                mejor_d, mejor = d, i
        if mejor is not None and mejor_d < TOL_NORM:
            usadas.add(mejor)
            aciertos += 1
    return aciertos


def evaluar(nombre, pesos, alts, conf):
    from ultralytics import YOLO
    if not os.path.isfile(pesos):
        return None

    model = YOLO(pesos)
    imgs = sorted(glob.glob(os.path.join(DIR_VAL_IMG, "*.png")))

    n_gt, n_det, n_ok, con3 = [], [], [], []

    for i in range(0, len(imgs), 8):
        lote = imgs[i:i + 8]
        res = model.predict(lote, imgsz=IMGSZ, conf=conf,
                            max_det=MAX_DET, verbose=False)
        for p, r in zip(lote, res):
            base = os.path.splitext(os.path.basename(p))[0]
            alt = alts.get(base)
            if alt is None:
                continue
            fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))

            gt = gt_grandes(base, alt)

            det = []
            if r.boxes is not None and len(r.boxes):
                xywhn = r.boxes.xywhn.cpu().numpy()
                for (cx, cy, bw, bh) in xywhn:
                    if bw * fac >= D_MIN_MAPA_KM:
                        det.append((float(cx), float(cy), float(bw * fac)))

            n_gt.append(len(gt))
            n_det.append(len(det))
            n_ok.append(emparejar(det, gt))
            con3.append(len(det) >= 3)

    if not n_gt:
        return None

    n_gt = np.array(n_gt); n_det = np.array(n_det); n_ok = np.array(n_ok)
    return dict(
        nombre=nombre, conf=conf,
        gt_frame=float(n_gt.mean()),
        det_frame=float(n_det.mean()),
        recall=float(n_ok.sum() / max(n_gt.sum(), 1)),
        pct3=100.0 * float(np.mean(con3)),
        pct0=100.0 * float(np.mean(n_det == 0)),
    )


def main():
    alts = alturas()
    print(f"Umbral de mapa: D >= {D_MIN_MAPA_KM} km   max_det = {MAX_DET}")
    print("=" * 82)
    print(f"{'modelo':<26} {'conf':>5} {'reales':>8} {'detect':>8} "
          f"{'recall':>8} {'>=3':>8} {'0 crat':>8}")
    print("-" * 82)

    filas = []
    for nombre, pesos in MODELOS.items():
        for c in CONFS:
            r = evaluar(nombre, pesos, alts, c)
            if r is None:
                continue
            filas.append(r)
            print(f"{r['nombre']:<26} {r['conf']:>5.2f} {r['gt_frame']:>8.1f} "
                  f"{r['det_frame']:>8.1f} {r['recall']:>8.3f} "
                  f"{r['pct3']:>7.1f}% {r['pct0']:>7.1f}%")
    print("=" * 82)

    if filas:
        out = os.path.join(RUTA_IA, "metrica_navegacion.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
