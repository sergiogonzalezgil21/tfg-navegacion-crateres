"""
Paso 11. Detecciones del vuelo de descenso.

Lo mismo que 08_exportar_detecciones.py pero sobre el segundo vuelo, el del
descenso parabolico con guinada. Mismo punto de operacion: conf = 0.70,
imgsz = 1024 y lote de 1.

El lote de 1 no es una preferencia. Ultralytics le pone al NMS un limite de
2.0 + 0.05*lote segundos y, al agotarse, descarta en silencio las imagenes que
quedaban del lote.

La salida va en pixeles, sin altitud ni kilometros: la altitud es justo lo que
el navegador tiene que estimar.
"""

import os
import csv
import glob
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

DIR_RENDER = os.path.join(RUTA_BLENDER, "Output_descenso")
SALIDA = os.path.join(RUTA_IA, "detecciones_descenso.csv")

# modelo reentrenado con aumentacion por rotacion (paso 12). El anterior
# no detectaba nada por encima de 45 grados de giro y este vuelo llega a 180.
PESOS = os.path.join(RUTA_IA, "runs", "finetune_rotacion", "weights", "best.pt")

CONF = 0.70
IMGSZ = 1024
MAX_DET = 3000
BW_MIN = 0.03        # mismo filtro de tamano angular que en el vuelo original

# =============================================================================


def main():
    from ultralytics import YOLO

    if not os.path.isfile(PESOS):
        print(f"[!] no existen los pesos: {PESOS}")
        return

    imgs = sorted(glob.glob(os.path.join(DIR_RENDER, "*.png")))
    if not imgs:
        print(f"[!] no hay renders en {DIR_RENDER}")
        print("    Has corrido ya Render -> Render Animation con CamaraDescenso?")
        return

    model = YOLO(PESOS)
    print("=" * 76)
    print(f"DESCENSO   {len(imgs)} frames   conf={CONF}   imgsz={IMGSZ}   lote=1")
    print("=" * 76)

    filas = []
    vacios = 0
    t0 = time.time()

    for k, p in enumerate(imgs):
        base = os.path.splitext(os.path.basename(p))[0]

        r = model.predict(p, imgsz=IMGSZ, conf=CONF,
                          max_det=MAX_DET, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            vacios += 1
            continue

        xywhn = r.boxes.xywhn.cpu().numpy()
        xywh = r.boxes.xywh.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()

        sel = xywhn[:, 2] >= BW_MIN
        xywhn, xywh, cf = xywhn[sel], xywh[sel], cf[sel]
        if len(cf) == 0:
            vacios += 1
            continue

        for (cxn, cyn, bwn, bhn), (cxp, cyp, bwp, bhp), c in zip(xywhn, xywh, cf):
            filas.append(dict(
                frame=int(base),
                cx_px=round(float(cxp), 2), cy_px=round(float(cyp), 2),
                w_px=round(float(bwp), 2), h_px=round(float(bhp), 2),
                cx_norm=round(float(cxn), 6), cy_norm=round(float(cyn), 6),
                w_norm=round(float(bwn), 6), h_norm=round(float(bhn), 6),
                conf=round(float(c), 4)))

        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(imgs)}   {(time.time()-t0)/(k+1):.2f} s/frame")

    if not filas:
        print("[!] el modelo no ha devuelto ni una caja. Algo va mal.")
        return

    with open(SALIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader()
        w.writerows(filas)

    print()
    print(f"{len(filas)} detecciones de {len(imgs)} frames -> {SALIDA}")
    print(f"media {len(filas)/len(imgs):.1f} cajas por frame, "
          f"{vacios} frames sin ninguna")

    # cuantas cajas por frame a lo largo del vuelo: al bajar deberia haber
    # menos crateres en el encuadre, pero mas grandes
    n = {}
    for r in filas:
        n[r["frame"]] = n.get(r["frame"], 0) + 1
    fs = sorted(n)
    print()
    print("  cajas por frame a lo largo del descenso:")
    for a, b in ((1, 100), (101, 250), (251, 400), (401, 550), (551, 700)):
        v = [n[f] for f in fs if a <= f <= b]
        if v:
            print(f"    frames {a:>3}-{b:<3}  media {np.mean(v):5.1f}   "
                  f"min {min(v):3d}   max {max(v):3d}")


if __name__ == "__main__":
    main()
