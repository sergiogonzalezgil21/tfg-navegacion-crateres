"""
Paso 9. Elige a la vez el umbral del mapa y cuantas cajas coge MATLAB.

Barrer las dos cosas por separado engana: al medir la precision de "quedarse
con las K cajas mas grandes" contra un mapa de D >= 16 km, toda caja de entre
8 y 16 km cuenta como fallo, pero si el mapa se construye a D >= 10 km esas
mismas cajas son aciertos. Las dos decisiones estan acopladas.

  D_mapa   hasta que diametro entran crateres en el mapa embarcado. Cuanto
           mas bajo, mas landmarks por frame, pero las tripletas crecen con
           el cubo.
  K        cuantas de las cajas mas grandes se queda MATLAB. Es invariante a
           escala: no necesita conocer la altitud.

Columnas que importan: precision, la fraccion de las K cajas que son landmarks
de verdad, que es la tasa de outliers que tendra que tragarse el emparejador;
y >=4, el porcentaje de frames con cuatro o mas aciertos, porque tres son el
minimo para la tripleta y el cuarto es el que permite verificarla.

Lee detecciones.csv, no infiere nada: tarda un par de segundos.
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

CSV_DET = os.path.join(RUTA_IA, "detecciones.csv")
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")
DIR_VAL_LAB = os.path.join(RUTA_IA, "dataset_blender", "val", "labels")

FOV_HORIZONTAL = 60.0
TOL_NORM = 0.03

D_MAPAS = [8, 10, 12, 14, 16]
KS = [4, 5, 6, 7, 8, 10]

# el mapa se construye con estos mismos criterios en blender_exportar_etiquetas
ANCHO_KM, ALTO_KM = 840.0, 630.0
MIN_FRESCURA = 0.30

# =============================================================================


def cargar():
    alt = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            alt[int(r["frame"])] = float(r["altura_km"])

    det = {}
    with open(CSV_DET, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            det.setdefault(int(r["frame"]), []).append(
                (float(r["cx_norm"]), float(r["cy_norm"]), float(r["w_norm"])))

    frames = sorted(int(os.path.basename(p)[:-4])
                    for p in glob.glob(os.path.join(DIR_VAL_LAB, "*.txt")))
    frames = [f for f in frames if f in alt]
    return alt, det, frames


def gt(f, alt, dmin):
    fac = 2.0 * alt[f] * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    out = []
    with open(os.path.join(DIR_VAL_LAB, f"{f:04d}.txt"), encoding="utf-8") as fh:
        for l in fh:
            v = l.split()
            if len(v) < 5:
                continue
            if float(v[3]) * fac >= dmin:
                out.append((float(v[1]), float(v[2])))
    return out


def emparejar(d, g):
    usadas = set()
    ok = 0
    for gx, gy in g:
        mejor, md = None, 1e9
        for i, (dx, dy) in enumerate(d):
            if i in usadas:
                continue
            dd = math.hypot(dx - gx, dy - gy)
            if dd < md:
                md, mejor = dd, i
        if mejor is not None and md < TOL_NORM:
            usadas.add(mejor)
            ok += 1
    return ok


def tam_mapa(dmin):
    """Cuantos crateres tendria el mapa embarcado, y cuantas tripletas."""
    p = os.path.join(RUTA_BLENDER, "crateres_catalogo.csv")
    if not os.path.isfile(p):
        return None, None
    n = 0
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (float(r["D_km"]) >= dmin
                    and float(r["frescura"]) >= MIN_FRESCURA):
                n += 1
    return n, n * (n - 1) * (n - 2) // 6


def main():
    for p in (CSV_DET, CSV_TRAY):
        if not os.path.isfile(p):
            print(f"[!] falta {p} - ejecuta antes 08_exportar_detecciones.py")
            return

    alt, det, frames = cargar()
    print("=" * 78)
    print(f"BARRIDO CONJUNTO   {len(frames)} frames de validacion   "
          f"tolerancia {TOL_NORM}")
    print("  mapa a D >= D_mapa, y MATLAB se queda con las K cajas mas grandes")
    print("=" * 78)

    filas = []
    for dmap in D_MAPAS:
        n_map, trip = tam_mapa(dmap)
        cab = f"D_mapa = {dmap} km"
        if n_map:
            cab += f"   ->   {n_map} crateres en el mapa, {trip:,} tripletas"
        print(f"\n{cab}")
        print(f"  {'K':>3} {'reales/frm':>11} {'aciertos':>9} {'precision':>10} "
              f"{'recall':>8} {'>=3':>7} {'>=4':>7}")
        print("  " + "-" * 60)

        for K in KS:
            A = T = G = 0
            c3, c4 = [], []
            for f in frames:
                g = gt(f, alt, dmap)
                G += len(g)
                d = sorted(det.get(f, []), key=lambda x: -x[2])[:K]
                ok = emparejar([(a, b) for a, b, _ in d], g)
                A += ok; T += len(d)
                c3.append(ok >= 3); c4.append(ok >= 4)
            n = len(frames)
            fila = dict(D_mapa=dmap, K=K, n_mapa=n_map, tripletas=trip,
                        reales_frame=G / n, aciertos_frame=A / n,
                        precision=A / max(T, 1), recall=A / max(G, 1),
                        pct3=100 * float(np.mean(c3)),
                        pct4=100 * float(np.mean(c4)))
            filas.append(fila)
            print(f"  {K:>3} {fila['reales_frame']:>11.2f} "
                  f"{fila['aciertos_frame']:>9.2f} {fila['precision']:>10.3f} "
                  f"{fila['recall']:>8.3f} {fila['pct3']:>6.1f}% "
                  f"{fila['pct4']:>6.1f}%")

    out = os.path.join(RUTA_IA, "barrido_conjunto.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        w.writeheader(); w.writerows(filas)
    print(f"\nGuardado en {out}")

    # el mejor: exige redundancia (>=4 en todos los frames) y luego maximiza
    # precision, que es lo que le baja el trabajo al emparejamiento
    val = [f for f in filas if f["pct4"] >= 99.9]
    if val:
        b = max(val, key=lambda r: r["precision"])
        print("\n" + "=" * 78)
        print(f"RECOMENDADO:  D_mapa = {b['D_mapa']} km,  K = {b['K']}")
        print(f"  {b['aciertos_frame']:.2f} landmarks correctos por frame "
              f"de {b['K']} cajas  (precision {b['precision']:.3f})")
        print(f"  100 % de frames con 4 o mas: hay redundancia para validar")
        if b["n_mapa"]:
            print(f"  mapa embarcado: {b['n_mapa']} crateres, "
                  f"{b['tripletas']:,} tripletas")
        print("=" * 78)
    else:
        print("\n[!] ninguna combinacion llega al 100 % de frames con >=4.")
        print("    Sin redundancia no se puede validar el emparejamiento:")
        print("    habria que regenerar el terreno con mas landmarks.")


if __name__ == "__main__":
    main()
