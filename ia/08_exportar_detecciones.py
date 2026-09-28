"""
Paso 8. Exporta las detecciones para MATLAB y elige el umbral del mapa.

Se infiere de un frame en uno, siempre, para que el NMS no descarte frames
enteros al agotar su presupuesto de tiempo. El punto de operacion es
conf = 0.70: mismo recall sobre los crateres de escala de mapa (0.994) y la
mitad de cajas basura, porque el detector los ve con confianza media 0.838.

Los pasos anteriores convertian las cajas a kilometros con la altitud
verdadera. Para medir el detector es lo correcto; para alimentar a MATLAB no,
porque la altitud es justo lo que hay que estimar. Asi que la salida va en
PIXELES, y los crateres de mapa se seleccionan de forma invariante a escala:
las K cajas mas grandes del frame, que por construccion son los landmarks.

Produce detecciones.csv con los 700 frames en pixeles, mas dos barridos: el de
D_min, que dice hasta que diametro se puede bajar el mapa, y el de K, que dice
cuantas de las K cajas mas grandes son landmarks de verdad.
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

DIR_RENDER = os.path.join(RUTA_BLENDER, "Output")          # los 700 frames
DIR_VAL_LAB = os.path.join(RUTA_IA, "dataset_blender", "val", "labels")
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")

PESOS = os.path.join(RUTA_IA, "runs", "finetune_blender", "weights", "best.pt")

CONF = 0.70          # punto de operacion elegido en el paso 7
IMGSZ = 1024
MAX_DET = 3000
FOV_HORIZONTAL = 60.0
TOL_NORM = 0.03

# solo se exportan cajas de cierto tamano angular: por debajo son crateres
# pequenos que el mapa no contiene y solo estorban. 0.03 normalizado son unos
# 58 px en 1920, y a 170 km de altura equivale a ~5.9 km de diametro.
BW_MIN_EXPORT = 0.03

D_MINS = [6, 8, 10, 12, 14, 16, 20]
KS = [3, 4, 5, 6, 8, 10, 12]

# =============================================================================


def alturas():
    d = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[f"{int(r['frame']):04d}"] = float(r["altura_km"])
    return d


def gt_km(base, alt, dmin):
    """Crateres reales del frame con D >= dmin, en coordenadas normalizadas."""
    p = os.path.join(DIR_VAL_LAB, base + ".txt")
    if not os.path.isfile(p):
        return None
    fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            v = l.split()
            if len(v) < 5:
                continue
            cx, cy, bw = float(v[1]), float(v[2]), float(v[3])
            if bw * fac >= dmin:
                out.append((cx, cy))
    return out


def emparejar(det_xy, gt_xy):
    usadas = set()
    ac = 0
    for (gx, gy) in gt_xy:
        mejor, md = None, 1e9
        for i, (dx, dy) in enumerate(det_xy):
            if i in usadas:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < md:
                md, mejor = d, i
        if mejor is not None and md < TOL_NORM:
            usadas.add(mejor)
            ac += 1
    return ac


# =============================================================================

def main():
    from ultralytics import YOLO

    if not os.path.isfile(PESOS):
        print(f"[!] no existen los pesos: {PESOS}")
        return

    imgs = sorted(glob.glob(os.path.join(DIR_RENDER, "*.png")))
    if not imgs:
        print(f"[!] no hay renders en {DIR_RENDER}")
        return

    alts = alturas()
    model = YOLO(PESOS)

    print("=" * 84)
    print(f"EXPORTANDO   {len(imgs)} frames   conf={CONF}   imgsz={IMGSZ}   "
          f"lote=1 (obligatorio)")
    print("=" * 84)

    # cache: para cada frame guardamos las cajas una sola vez y luego se
    # barren los umbrales sin volver a inferir
    cache = {}
    filas_csv = []
    t0 = time.time()

    for k, p in enumerate(imgs):
        base = os.path.splitext(os.path.basename(p))[0]
        # UNO A UNO. Nunca en lote: ver la cabecera de este fichero.
        r = model.predict(p, imgsz=IMGSZ, conf=CONF,
                          max_det=MAX_DET, verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            cache[base] = (np.zeros((0, 4)), np.zeros(0))
            continue

        xywhn = r.boxes.xywhn.cpu().numpy()
        xywh = r.boxes.xywh.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy()

        sel = xywhn[:, 2] >= BW_MIN_EXPORT
        xywhn, xywh, cf = xywhn[sel], xywh[sel], cf[sel]
        cache[base] = (xywhn, cf)

        for (cxn, cyn, bwn, bhn), (cxp, cyp, bwp, bhp), c in zip(xywhn, xywh, cf):
            filas_csv.append(dict(
                frame=int(base),
                cx_px=round(float(cxp), 2), cy_px=round(float(cyp), 2),
                w_px=round(float(bwp), 2), h_px=round(float(bhp), 2),
                cx_norm=round(float(cxn), 6), cy_norm=round(float(cyn), 6),
                w_norm=round(float(bwn), 6), h_norm=round(float(bhn), 6),
                conf=round(float(c), 4)))

        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(imgs)} frames   "
                  f"{(time.time()-t0)/(k+1):.2f} s/frame")

    if not filas_csv:
        print("[!] el modelo no ha devuelto ni una caja, algo va mal")
        return

    out = os.path.join(RUTA_IA, "detecciones.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas_csv[0].keys()))
        w.writeheader(); w.writerows(filas_csv)
    n_frames_vacios = sum(1 for b in cache if len(cache[b][0]) == 0)
    print(f"\n{len(filas_csv)} detecciones de {len(imgs)} frames -> {out}")
    print(f"media {len(filas_csv)/len(imgs):.1f} cajas por frame, "
          f"{n_frames_vacios} frames vacios")
    print("  (el fichero va en pixeles y sin altitud: MATLAB estima la escala)")

    # ---------------------------------------------------------------- barridos
    bases_val = sorted(os.path.splitext(os.path.basename(x))[0]
                       for x in glob.glob(os.path.join(DIR_VAL_LAB, "*.txt")))
    bases_val = [b for b in bases_val if b in cache and b in alts]
    if not bases_val:
        print("\n[!] sin etiquetas de validacion, no se puede barrer")
        return

    print("\n" + "=" * 84)
    print(f"BARRIDO 1 - hasta que diametro aguanta el detector "
          f"({len(bases_val)} frames de val)")
    print(f"{'D_min km':>9} {'reales/frm':>11} {'detect/frm':>11} "
          f"{'recall':>8} {'>=3':>8} {'>=4':>8}")
    print("-" * 60)

    for dmin in D_MINS:
        ngt, ndet, nok, c3, c4 = [], [], [], [], []
        for b in bases_val:
            alt = alts[b]
            fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
            gt = gt_km(b, alt, dmin)
            if gt is None:
                continue
            xywhn, _cf = cache[b]
            m = (xywhn[:, 2] * fac) >= dmin if len(xywhn) else np.zeros(0, bool)
            det = [(float(a), float(c)) for a, c in
                   zip(xywhn[m, 0], xywhn[m, 1])] if len(xywhn) else []
            ngt.append(len(gt)); ndet.append(len(det))
            nok.append(emparejar(det, gt))
            c3.append(len(det) >= 3); c4.append(len(det) >= 4)
        if not ngt:
            continue
        a = np.array
        print(f"{dmin:>9} {a(ngt).mean():>11.2f} {a(ndet).mean():>11.2f} "
              f"{a(nok).sum()/max(a(ngt).sum(),1):>8.3f} "
              f"{100*np.mean(c3):>7.1f}% {100*np.mean(c4):>7.1f}%")

    print("\n" + "=" * 84)
    print("BARRIDO 2 - regla invariante a escala: quedarse con las K mas grandes")
    print("  (esto es lo que hara MATLAB, que no conoce la altitud)")
    print(f"{'K':>4} {'aciertos':>10} {'precision':>11} {'>=3 buenas':>12} "
          f"{'D_min equiv':>13}")
    print("-" * 56)

    D_REF = 16.0
    for K in KS:
        ok_tot, k_tot, con3, dmins = 0, 0, [], []
        for b in bases_val:
            alt = alts[b]
            fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
            gt = gt_km(b, alt, D_REF)
            if gt is None:
                continue
            xywhn, _cf = cache[b]
            if len(xywhn) == 0:
                con3.append(False)
                continue
            orden = np.argsort(-xywhn[:, 2])[:K]
            top = xywhn[orden]
            det = [(float(x), float(y)) for x, y in zip(top[:, 0], top[:, 1])]
            ok = emparejar(det, gt)
            ok_tot += ok; k_tot += len(det); con3.append(ok >= 3)
            dmins.append(float(top[:, 2].min() * fac))
        if k_tot == 0:
            continue
        print(f"{K:>4} {ok_tot/max(len(con3),1):>10.2f} "
              f"{ok_tot/k_tot:>11.3f} {100*np.mean(con3):>11.1f}% "
              f"{np.mean(dmins):>12.1f} km")

    print("\n  aciertos   = crateres de mapa correctos por frame")
    print("  precision  = fraccion de las K cajas que son crateres de mapa")
    print(f"  >=3 buenas = frames con al menos 3 aciertos (D>={D_REF:.0f} km)")
    print("  D_min equiv= diametro medio de la caja mas pequena de las K")
    print("=" * 84)


if __name__ == "__main__":
    main()
