"""
Paso 6. Diagnostico: el modelo afinado detecta MENOS crateres de escala de
mapa cuando se BAJA el umbral de confianza (3.2 por frame a conf 0.25, 2.1 a
conf 0.05). Eso es imposible: bajar conf solo puede anadir cajas.

La sospecha es el cupo de max_det. Ultralytics, despues del NMS, ordena por
confianza y se queda con las primeras max_det cajas. Al bajar conf entran
miles de cajas pequenas de confianza media, se llena el cupo, y las pocas
cajas grandes se caen del corte.

Mide tres cosas: cajas totales por frame (si sale pegado a max_det, el cupo
esta lleno), cajas de escala de mapa con max_det 300 / 1500 / 6000 (si al
subir el cupo reaparecen, queda demostrado), y la confianza de las cajas
grandes frente a las pequenas con su posicion en el ranking.
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
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")

PESOS_AFINADO = os.path.join(RUTA_IA, "runs", "finetune_blender",
                             "weights", "best.pt")
# Pesos de la etapa 1, el entrenamiento sobre imagenes reales de crateres. No
# viajan en el repositorio por tamano: se obtienen ejecutando los scripts de
# etapa1/, o se indica donde estan con la variable de entorno
# TFG_PESOS_ETAPA1. Si no aparecen, se parte de los pesos publicos de
# Ultralytics, que es la ultima entrada de la lista.
RUTA_ETAPA1 = os.environ.get("TFG_PESOS_ETAPA1",
                             os.path.join(_AQUI, "pesos_etapa1"))
PESOS_VIEJO = os.path.join(RUTA_ETAPA1, "yolov8_crater-10", "weights", "best.pt")

D_MIN_MAPA_KM = 16.0
FOV_HORIZONTAL = 60.0
IMGSZ = 1024

CONF = 0.05                      # el punto donde mas se nota el problema
CUPOS = [300, 1500, 6000]        # max_det a comparar

N_FRAMES = 60                    # con 60 frames sobra para el diagnostico

# =============================================================================


def alturas():
    d = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[f"{int(r['frame']):04d}"] = float(r["altura_km"])
    return d


def diagnostico(nombre, pesos, alts, filas):
    from ultralytics import YOLO
    if not os.path.isfile(pesos):
        print(f"  [!] no existe: {pesos}")
        return

    model = YOLO(pesos)
    imgs = sorted(glob.glob(os.path.join(DIR_VAL_IMG, "*.png")))[:N_FRAMES]
    if not imgs:
        print("  [!] no hay imagenes en val")
        return

    print(f"\n{nombre}")
    print(f"  {'max_det':>8} {'cajas/frame':>13} {'grandes/frame':>15} "
          f"{'frames con 0':>14} {'cupo lleno':>12}")
    print("  " + "-" * 68)

    guardado = {}

    for cupo in CUPOS:
        tot, gra, lleno = [], [], []
        conf_gra, conf_peq, rank_gra = [], [], []

        for i in range(0, len(imgs), 8):
            lote = imgs[i:i + 8]
            res = model.predict(lote, imgsz=IMGSZ, conf=CONF,
                                max_det=cupo, verbose=False)
            for p, r in zip(lote, res):
                base = os.path.splitext(os.path.basename(p))[0]
                alt = alts.get(base)
                if alt is None:
                    continue
                fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))

                if r.boxes is None or len(r.boxes) == 0:
                    tot.append(0); gra.append(0); lleno.append(False)
                    continue

                bw = r.boxes.xywhn.cpu().numpy()[:, 2]
                cf = r.boxes.conf.cpu().numpy()
                D = bw * fac
                es_grande = D >= D_MIN_MAPA_KM

                tot.append(int(len(bw)))
                gra.append(int(es_grande.sum()))
                lleno.append(len(bw) >= cupo)

                if cupo == CUPOS[-1]:
                    # el ranking de Ultralytics ya viene ordenado por confianza
                    orden = np.argsort(-cf)
                    pos = np.empty_like(orden)
                    pos[orden] = np.arange(len(cf))
                    if es_grande.any():
                        conf_gra.extend(cf[es_grande].tolist())
                        rank_gra.extend((pos[es_grande] / len(cf)).tolist())
                    if (~es_grande).any():
                        conf_peq.extend(cf[~es_grande].tolist())

        tot = np.array(tot); gra = np.array(gra); lleno = np.array(lleno)
        print(f"  {cupo:>8} {tot.mean():>13.1f} {gra.mean():>15.2f} "
              f"{100.0*np.mean(gra == 0):>13.1f}% "
              f"{100.0*np.mean(lleno):>11.1f}%")
        guardado[cupo] = (tot.mean(), gra.mean())
        filas.append(dict(
            modelo=nombre, conf=CONF, max_det=cupo,
            cajas_frame=float(tot.mean()),
            grandes_frame=float(gra.mean()),
            pct_frames_sin_grandes=100.0 * float(np.mean(gra == 0)),
            pct_cupo_lleno=100.0 * float(np.mean(lleno)),
            conf_media_grandes="", conf_media_pequenas="",
            rank_medio_grandes=""))

        if cupo == CUPOS[-1] and conf_gra and conf_peq:
            print("\n  Sesgo de tamano del modelo:")
            print(f"    confianza media de las cajas >= {D_MIN_MAPA_KM:.0f} km : "
                  f"{np.mean(conf_gra):.3f}   (n={len(conf_gra)})")
            print(f"    confianza media de las cajas pequenas       : "
                  f"{np.mean(conf_peq):.3f}   (n={len(conf_peq)})")
            print(f"    posicion media de una caja grande en el ranking: "
                  f"{100.0*np.mean(rank_gra):.1f}% de la lista")
            print("    (si esa posicion esta en la cola, un max_det bajo las")
            print("     borra justo a ellas y a ninguna otra)")
            filas[-1]["conf_media_grandes"] = round(float(np.mean(conf_gra)), 4)
            filas[-1]["conf_media_pequenas"] = round(float(np.mean(conf_peq)), 4)
            filas[-1]["rank_medio_grandes"] = round(100.0 * float(np.mean(rank_gra)), 2)

    a, b = guardado[CUPOS[0]][1], guardado[CUPOS[-1]][1]
    if b > a * 1.15:
        print(f"\n  >> CONFIRMADO: subiendo el cupo de {CUPOS[0]} a {CUPOS[-1]} "
              f"los crateres de mapa pasan de {a:.2f} a {b:.2f} por frame.")
        print("     El problema era el recorte por max_det, no el detector.")
    else:
        print(f"\n  >> El cupo NO era el problema ({a:.2f} -> {b:.2f}).")
        print("     El modelo simplemente no ve los crateres grandes.")


def main():
    alts = alturas()
    print("=" * 78)
    print(f"DIAGNOSTICO   conf={CONF}   imgsz={IMGSZ}   "
          f"frames={N_FRAMES}   D_mapa>={D_MIN_MAPA_KM:.0f} km")
    print("=" * 78)
    filas = []
    diagnostico("afinado con Blender", PESOS_AFINADO, alts, filas)
    diagnostico("viejo (imagenes reales)  [control]", PESOS_VIEJO, alts, filas)
    print("\n" + "=" * 78)

    if filas:
        out = os.path.join(RUTA_IA, "diagnostico_deteccion.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
