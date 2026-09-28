"""
Paso 7. La medida buena, sin el recorte silencioso del NMS.

Los avisos "NMS time limit exceeded" no eran cosmeticos. Ultralytics le pone
al NMS un presupuesto de 2.0 + 0.05*lote segundos y, cuando se agota, corta el
bucle de imagenes del lote: las que quedaban detras se devuelven VACIAS sin
lanzar ningun error. De ahi que el 46.7 % de los frames saliera con cero
crateres de mapa, y que bajar conf redujera las detecciones (mas candidatos ->
NMS mas lento -> mas frames descartados).

El modelo afinado no tiene sesgo contra los crateres grandes: los detecta con
confianza media 0.838 frente a 0.509 de los pequenos.

Parte A, demuestra el efecto: mismo modelo y mismos frames, lote de 8 frente a
lote de 1. Parte B, la tabla de navegacion ya sin el artefacto, sin lotes y
barriendo umbrales hacia arriba.
"""

import os
import csv
import glob
import math
import time

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

D_MIN_MAPA_KM = 16.0
FOV_HORIZONTAL = 60.0
IMGSZ = 1024
MAX_DET = 3000
TOL_NORM = 0.03

# se barre HACIA ARRIBA, que es donde esta el punto de operacion bueno
CONFS = [0.05, 0.25, 0.50, 0.70]

N_FRAMES_PARTE_A = 60

# =============================================================================


def alturas():
    d = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[f"{int(r['frame']):04d}"] = float(r["altura_km"])
    return d


def gt_grandes(base, alt):
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
            if bw * fac >= D_MIN_MAPA_KM:
                out.append((cx, cy, bw * fac))
    return out


def emparejar(det, gt):
    """Cuantos crateres reales quedan cubiertos. Cada deteccion se usa una vez."""
    usadas = set()
    aciertos = 0
    for (gx, gy, _d) in gt:
        mejor, mejor_d = None, 1e9
        for i, (dx, dy, _dd) in enumerate(det):
            if i in usadas:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < mejor_d:
                mejor_d, mejor = d, i
        if mejor is not None and mejor_d < TOL_NORM:
            usadas.add(mejor)
            aciertos += 1
    return aciertos


def cajas_grandes(r, fac):
    if r.boxes is None or len(r.boxes) == 0:
        return [], 0
    x = r.boxes.xywhn.cpu().numpy()
    D = x[:, 2] * fac
    sel = D >= D_MIN_MAPA_KM
    det = [(float(a), float(b), float(c))
           for a, b, c in zip(x[sel, 0], x[sel, 1], D[sel])]
    return det, int(len(x))


# ---------------------------------------------------------------- parte A

def parte_a(alts):
    from ultralytics import YOLO
    pesos = MODELOS["afinado con Blender"]
    if not os.path.isfile(pesos):
        print("  [!] faltan los pesos del modelo afinado")
        return

    model = YOLO(pesos)
    imgs = sorted(glob.glob(os.path.join(DIR_VAL_IMG, "*.png")))[:N_FRAMES_PARTE_A]

    print("PARTE A - el tamano de lote decide cuantos frames sobreviven")
    print(f"  modelo afinado, conf=0.05, {len(imgs)} frames")
    print(f"  {'lote':>6} {'cajas/frame':>13} {'grandes/frame':>15} "
          f"{'frames vacios':>15} {'seg/frame':>11}")
    print("  " + "-" * 66)

    for lote_n in (8, 1):
        tot, gra, vacios = [], [], 0
        t0 = time.time()
        for i in range(0, len(imgs), lote_n):
            lote = imgs[i:i + lote_n]
            res = model.predict(lote, imgsz=IMGSZ, conf=0.05,
                                max_det=MAX_DET, verbose=False)
            for p, r in zip(lote, res):
                base = os.path.splitext(os.path.basename(p))[0]
                alt = alts.get(base)
                if alt is None:
                    continue
                fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
                det, n = cajas_grandes(r, fac)
                tot.append(n); gra.append(len(det))
                if n == 0:
                    vacios += 1
        dt = (time.time() - t0) / max(len(tot), 1)
        print(f"  {lote_n:>6} {np.mean(tot):>13.1f} {np.mean(gra):>15.2f} "
              f"{100.0*vacios/max(len(tot),1):>14.1f}% {dt:>11.2f}")

    print("\n  Un frame vacio con lote de 8 significa que el NMS se quedo sin")
    print("  tiempo antes de llegar a el. Con lote de 1 cada frame tiene su")
    print("  propio presupuesto y ya no se pierde ninguno.\n")


# ---------------------------------------------------------------- parte B

def parte_b(alts):
    from ultralytics import YOLO

    imgs = sorted(glob.glob(os.path.join(DIR_VAL_IMG, "*.png")))
    print("PARTE B - tabla de navegacion, sin lotes, todos los frames de val")
    print(f"  {len(imgs)} frames   D_mapa >= {D_MIN_MAPA_KM:.0f} km   "
          f"max_det = {MAX_DET}")
    print("=" * 88)
    print(f"{'modelo':<26} {'conf':>5} {'cajas':>8} {'reales':>7} {'detect':>7} "
          f"{'recall':>7} {'>=3':>7} {'0 crat':>7} {'s/frm':>6}")
    print("-" * 88)

    filas = []
    for nombre, pesos in MODELOS.items():
        if not os.path.isfile(pesos):
            continue
        model = YOLO(pesos)
        for c in CONFS:
            n_gt, n_det, n_ok, n_tot, con3 = [], [], [], [], []
            t0 = time.time()
            for p in imgs:
                base = os.path.splitext(os.path.basename(p))[0]
                alt = alts.get(base)
                if alt is None:
                    continue
                fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
                r = model.predict(p, imgsz=IMGSZ, conf=c,
                                  max_det=MAX_DET, verbose=False)[0]
                det, n = cajas_grandes(r, fac)
                gt = gt_grandes(base, alt)
                n_gt.append(len(gt)); n_det.append(len(det))
                n_ok.append(emparejar(det, gt)); n_tot.append(n)
                con3.append(len(det) >= 3)
            if not n_gt:
                continue
            dt = (time.time() - t0) / len(n_gt)
            a = np.array
            f = dict(modelo=nombre, conf=c,
                     cajas_frame=float(a(n_tot).mean()),
                     gt_frame=float(a(n_gt).mean()),
                     det_frame=float(a(n_det).mean()),
                     recall=float(a(n_ok).sum() / max(a(n_gt).sum(), 1)),
                     pct3=100.0 * float(np.mean(con3)),
                     pct0=100.0 * float(np.mean(a(n_det) == 0)),
                     seg_frame=dt)
            filas.append(f)
            print(f"{nombre:<26} {c:>5.2f} {f['cajas_frame']:>8.1f} "
                  f"{f['gt_frame']:>7.2f} {f['det_frame']:>7.2f} "
                  f"{f['recall']:>7.3f} {f['pct3']:>6.1f}% "
                  f"{f['pct0']:>6.1f}% {dt:>6.2f}")
    print("=" * 88)

    if filas:
        out = os.path.join(RUTA_IA, "deteccion_limpia.csv")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        print(f"Guardado en {out}")

        mejor = max(filas, key=lambda r: (r["pct3"], r["recall"]))
        print(f"\nMejor punto de operacion: {mejor['modelo']} con conf="
              f"{mejor['conf']:.2f}")
        print(f"  {mejor['det_frame']:.2f} crateres de mapa por frame, "
              f"{mejor['pct3']:.1f}% de frames con los 3 que hacen falta")


def main():
    alts = alturas()
    print("=" * 88)
    parte_a(alts)
    parte_b(alts)


if __name__ == "__main__":
    main()
