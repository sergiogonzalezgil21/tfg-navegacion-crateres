"""
Comprueba, sin imprimir nada, que las piezas del mapa cierran y encajan.

  1. Cada pieza es un solido cerrado: toda arista pertenece a dos triangulos,
     Euler V-E+F = 2 y el volumen sale positivo. Si falla algo de esto, el
     laminador "repara" la pieza por su cuenta y sin avisar.
  2. Todas apoyan en el mismo plano, z = 0, que es la condicion para que la
     maqueta montada no cojee.
  3. Las costuras cuadran: el borde derecho de cada pieza tiene los mismos
     vertices que el izquierdo de su vecina, uno a uno y con la misma cota.
     Se da el desajuste maximo en milimetros.
  4. Reconstruye el mapa entero a partir de los STL ya escritos y lo saca en
     PNG con luz rasante: una costura desplazada se ve como una raya recta.

    python comprobar_piezas.py [carpeta]      (por defecto: impresion3d)

Deja tambien un comprobacion.txt con el informe completo, util cuando se
lanza desde el editor de Blender y la consola del sistema no esta abierta.
"""

import os
import sys
import csv
import math

import numpy as np

try:
    AQUI = os.path.dirname(os.path.abspath(__file__))
except NameError:
    AQUI = os.path.abspath(os.getcwd())

try:                      # dentro de Blender sys.argv son los de Blender
    import bpy            # noqa: F401
    _EN_BLENDER = True
except ImportError:
    _EN_BLENDER = False

if not _EN_BLENDER and len(sys.argv) > 1:
    CARPETA = sys.argv[1]
else:
    CARPETA = os.path.join(AQUI, "impresion3d")
TOL_MM = 1e-4          # por debajo de esto es ruido del float32 del STL

_DT = np.dtype({"names": ["n", "v", "a"],
                "formats": ["<3f4", "<9f4", "<u2"],
                "offsets": [0, 12, 48], "itemsize": 50})


def leer_stl(ruta):
    raw = open(ruta, "rb").read()
    n = int(np.frombuffer(raw[80:84], "<u4")[0])
    t = np.frombuffer(raw[84:84 + 50 * n], _DT, count=n)
    return t["v"].reshape(-1, 3, 3).astype(np.float64)


def analizar(V):
    """Topologia y volumen de una malla (N,3,3)."""
    plano = np.ascontiguousarray(V.reshape(-1, 3).astype(np.float32))
    clave = plano.view([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]).ravel()
    unicos, idx = np.unique(clave, return_inverse=True)
    F = idx.reshape(-1, 3)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    _, cuenta = np.unique(E, axis=0, return_counts=True)
    vol = np.einsum("ij,ij->i", V[:, 0], np.cross(V[:, 1], V[:, 2])).sum() / 6.0
    return dict(tri=len(V), vert=len(unicos),
                sueltas=int((cuenta == 1).sum()), multiples=int((cuenta > 2).sum()),
                euler=len(unicos) - len(cuenta) + len(V), vol=vol / 1000.0)


def borde(V, eje, valor):
    """Vertices unicos del borde {eje == valor}, ordenados por la otra
    coordenada del plano. Devuelve (transversal, z)."""
    P = V.reshape(-1, 3)
    m = np.abs(P[:, eje] - valor) < 1e-4
    otro = 1 - eje
    B = np.unique(np.round(P[m][:, [otro, 2]], 4), axis=0)
    # se quita la falda vertical: en cada transversal quedan la cota del
    # terreno y la del fondo (z=0). Nos interesa el perfil de arriba.
    orden = np.lexsort((-B[:, 1], B[:, 0]))
    B = B[orden]
    _, primero = np.unique(B[:, 0], return_index=True)
    return B[primero]


def rejilla(V):
    """Reconstruye la superficie superior de una pieza como matriz de cotas."""
    P = np.unique(np.round(V.reshape(-1, 3), 4), axis=0)
    P = P[P[:, 2] > 0.0]                       # fuera el fondo plano
    xs = np.unique(P[:, 0]); ys = np.unique(P[:, 1])
    Z = np.zeros((len(ys), len(xs)))
    jx = np.searchsorted(xs, P[:, 0]); iy = np.searchsorted(ys, P[:, 1])
    Z[iy, jx] = P[:, 2]                        # cada (x,y) tiene una sola cota
    if not np.all(Z > 0):
        print(f"  aviso: {int((Z <= 0).sum())} nodos sin cota al reconstruir")
    return Z[::-1], xs, ys                     # fila 0 = y mayor = norte


class _Tee:
    def __init__(self, ruta):
        self.f = open(ruta, "w", encoding="utf-8")
        self.orig = sys.stdout
        sys.stdout = self

    def write(self, t):
        self.orig.write(t); self.f.write(t)

    def flush(self):
        self.orig.flush(); self.f.flush()

    def cerrar(self):
        sys.stdout = self.orig
        self.f.close()


def main():
    if not os.path.isdir(CARPETA):
        print(f"No encuentro la carpeta {CARPETA}"); return
    reg = _Tee(os.path.join(CARPETA, "comprobacion.txt"))
    try:
        _main()
    finally:
        reg.cerrar()


def _main():
    ruta_csv = os.path.join(CARPETA, "mapa_lunar_piezas.csv")
    if not os.path.isfile(ruta_csv):
        print(f"No encuentro {ruta_csv}"); return
    piezas = list(csv.DictReader(open(ruta_csv, newline="")))
    rej = {}
    fallos = 0

    print("=" * 72)
    print("1) Cada pieza por separado")
    print("=" * 72)
    print(f"{'pieza':<18}{'triangulos':>11}{'aristas sueltas':>17}"
          f"{'Euler':>7}{'volumen':>11}{'apoyo':>9}")
    for p in piezas:
        f = os.path.join(CARPETA, p["pieza"] + ".stl")
        V = leer_stl(f)
        a = analizar(V)
        z0 = V[:, :, 2].min()
        ok = (a["sueltas"] == 0 and a["multiples"] == 0 and a["euler"] == 2
              and a["vol"] > 0 and abs(z0) < TOL_MM)
        fallos += (not ok)
        print(f"{p['pieza']:<18}{a['tri']:>11,}{a['sueltas']:>17}"
              f"{a['euler']:>7}{a['vol']:>9.1f} cm3{z0:>7.3f} mm"
              f"   {'OK' if ok else '<-- MAL'}")
        rej[(p["fila"], int(p["col"]))] = V

    print("\n" + "=" * 72)
    print("2) Costuras entre vecinas")
    print("=" * 72)
    filas = sorted({p["fila"] for p in piezas})
    cols = sorted({int(p["col"]) for p in piezas})
    peor = 0.0

    for fi in filas:                                   # vecinas de izq/dcha
        for c in cols[:-1]:
            A, B = rej.get((fi, c)), rej.get((fi, c + 1))
            if A is None or B is None:
                continue
            ba = borde(A, 0, A[:, :, 0].max())          # borde derecho de A
            bb = borde(B, 0, B[:, :, 0].min())          # borde izquierdo de B
            estado, d = comparar(ba, bb)
            peor = max(peor, d)
            fallos += (estado != "OK")
            print(f"  {fi}{c} | {fi}{c+1}   {len(ba):>5} vertices   "
                  f"desajuste max {d:.6f} mm   {estado}")

    for k in range(len(filas) - 1):                    # vecinas arriba/abajo
        fa, fb = filas[k], filas[k + 1]
        for c in cols:
            A, B = rej.get((fa, c)), rej.get((fb, c))
            if A is None or B is None:
                continue
            ba = borde(A, 1, A[:, :, 1].min())          # borde sur de A
            bb = borde(B, 1, B[:, :, 1].max())          # borde norte de B
            estado, d = comparar(ba, bb)
            peor = max(peor, d)
            fallos += (estado != "OK")
            print(f"  {fa}{c} / {fb}{c}   {len(ba):>5} vertices   "
                  f"desajuste max {d:.6f} mm   {estado}")

    print("\n" + "=" * 72)
    if fallos == 0:
        print(f"TODO CORRECTO. Desajuste maximo en cualquier costura: "
              f"{peor:.6f} mm")
        print("Las 12 piezas son solidos cerrados, apoyan en el mismo plano y")
        print("comparten los vertices de sus bordes uno a uno.")
    else:
        print(f"{fallos} comprobaciones han fallado. Ver las lineas marcadas.")
    print("=" * 72)

    montar(rej, filas, cols, os.path.join(CARPETA, "mapa_lunar_montado.png"))


def comparar(a, b):
    if a.shape != b.shape:
        return f"<-- distinto numero de vertices ({len(a)} vs {len(b)})", float("inf")
    d = float(np.abs(a - b).max())
    return ("OK" if d < TOL_MM else "<-- NO CUADRA"), d


def montar(rej, filas, cols, ruta):
    """Pega las cotas de las 12 piezas y saca un PNG con luz rasante."""
    try:
        from PIL import Image
    except ImportError:
        print("(sin Pillow: me salto el montaje visual)"); return

    bloques = []
    for fi in filas:
        fila = []
        for c in cols:
            V = rej.get((fi, c))
            if V is None:
                return
            Z, xs, ys = rejilla(V)
            # los bordes compartidos se quitan de todas menos de la primera
            if c != cols[0]:
                Z = Z[:, 1:]
            if fi != filas[0]:
                Z = Z[1:, :]
            fila.append(Z)
        bloques.append(np.hstack(fila))
    Z = np.vstack(bloques)

    paso = float(np.diff(np.unique(np.round(xs, 4)))[0])
    gy, gx = np.gradient(Z.astype(np.float32), paso)
    az, el = math.radians(315.0), math.radians(30.0)
    pend = np.arctan(np.hypot(gx, gy)); asp = np.arctan2(gy, -gx)
    sh = np.clip(np.sin(el) * np.cos(pend) +
                 np.cos(el) * np.sin(pend) * np.cos(az - asp), 0, 1)
    Image.fromarray((255 * (0.08 + 0.92 * sh)).astype(np.uint8)).save(ruta)
    print(f"\nMontaje reconstruido desde los STL: {Z.shape[1]} x {Z.shape[0]} "
          f"muestras -> {ruta}")
    print("Si una costura no cuadrara, se veria como una raya recta "
          "cruzando el relieve.")


if __name__ == "__main__":
    main()
