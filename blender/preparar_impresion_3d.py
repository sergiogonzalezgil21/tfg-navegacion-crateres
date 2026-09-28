"""
Convierte el terreno en una maqueta imprimible en 3D, escalada 1:1 000 000
(840 x 630 km -> 840 x 630 mm), con una base solida de espesor constante por
debajo del punto mas bajo de la superficie.

El heightmap se remuestrea por promediado de area, no por interpolacion: lo
que no cabe en una linea de extrusion se funde con sus vecinos en vez de
aparecer o desaparecer segun donde caiga la muestra.

La cota del fondo se calcula una sola vez para todo el mapa y no por trozos.
De ahi salen las dos propiedades que interesan: ningun crater perfora la base,
y todas las piezas tienen la misma altura total, asi que la maqueta apoya
plana. Si el mapa no cabe en la cama, se parte en piezas que casan a tope.

Al terminar informa del material bajo el punto mas bajo, de los crateres que
se pierden por debajo de la altura de capa y del filamento estimado.

    python preparar_impresion_3d.py

Le basta numpy y terreno_lunar.py al lado. Tambien se puede lanzar desde el
editor de Blender, que trae numpy; en ese caso los print() salen en la consola
del sistema, y por eso deja ademas un registro.txt en la carpeta de salida.
Las rutas se resuelven respecto a la carpeta de este archivo.
"""

import os
import sys
import csv
import math
import time
import struct

import numpy as np

# Carpeta de este archivo. Blender ejecuta los scripts con el directorio de
# trabajo puesto en donde se arranco Blender, asi que "." no sirve.
try:
    AQUI = os.path.dirname(os.path.abspath(__file__))
except NameError:                 # pegado a mano en una consola interactiva
    AQUI = os.path.abspath(os.getcwd())


# =========================================================
# CONFIGURACION
# =========================================================

# --- de donde sale el terreno ---
CARPETA_FUENTE = AQUI             # donde esta terreno_lunar.py
CACHE_NPY = "H_8192.npy"          # cache del heightmap en metros
RES_FUENTE_X = 8192               # resolucion con la que se genera si no hay cache
SEED = 20260818                   # semilla del TFG; otro valor da otro terreno

# --- escala ---
# 840 km -> 840 mm.  1 km = 1 mm = 1/1e6.
ESCALA = 1_000_000.0
EXAGERACION_Z = 1.0               # 1.0 = escala real. Con 2 o 3 se aprecian
                                  # los crateres pequenos, a costa de dejar
                                  # de estar a escala (el script lo avisa)

# Recorte opcional del mapa, en km sobre el sistema del TFG (origen abajo a la
# izquierda, x hacia la derecha, y hacia arriba). None = mapa entero.
# Ejemplo, los 200x200 km del centro:  RECORTE_KM = (320, 215, 520, 415)
RECORTE_KM = None

# --- resolucion de la malla impresa ---
PASO_MM = 0.5                     # separacion entre muestras en XY.
                                  # Con boquilla de 0.4 mm no compensa bajar
                                  # de aqui: el extrusor no puede dibujar un
                                  # detalle mas fino que su propio cordon, y
                                  # cada muestra de mas multiplica el tamano
                                  # del STL. 0.4 apura el detalle; 0.8 alivia
                                  # el tamano del fichero.

# --- base ---
ESPESOR_BASE_MM = 3.0             # material macizo BAJO EL PUNTO MAS BAJO
MIN_MATERIAL_MM = 1.2             # umbral de aviso (3 perimetros a 0.4 mm)

# --- troceado ---
CAMA_X_MM = 220.0                 # cama de la impresora; con None, None no
CAMA_Y_MM = 220.0                 # se trocea y sale una unica pieza
MARGEN_CAMA_MM = 5.0              # margen por lado dentro de la cama
ORIGEN_LOCAL = True               # cada pieza con su esquina en (0,0)

# Rejilla de piezas impuesta a mano en lugar de deducida de la cama:
# columnas y filas. 4 x 3 son las 12 piezas de 210x210 mm del TFG.
# Con None se calcula sola a partir de CAMA_X_MM / CAMA_Y_MM.
PIEZAS_X = 4
PIEZAS_Y = 3

# --- impresion ---
ALTURA_CAPA_MM = 0.2              # solo para los avisos de resolucion

# --- salida ---
CARPETA_SALIDA = os.path.join(AQUI, "impresion3d")
PREFIJO = "mapa_lunar"
GUARDAR_PREVIA = True             # PNG con el sombreado y las lineas de corte
GUARDAR_CSV_PIEZAS = True


# =========================================================
# 1. HEIGHTMAP
# =========================================================

def cargar_heightmap():
    """Devuelve H en metros, con fila 0 = borde superior (y = ALTO_KM)."""
    ruta = os.path.join(CARPETA_FUENTE, CACHE_NPY)
    if os.path.isfile(ruta):
        print(f"[1] Cache encontrada: {CACHE_NPY}")
        H = np.load(ruta)
        print(f"    {H.shape[1]}x{H.shape[0]} muestras")
        return H.astype(np.float32)

    print("[1] No hay cache. Generando el terreno (esto tarda unos minutos)...")
    sys.path.insert(0, os.path.abspath(CARPETA_FUENTE))
    import terreno_lunar as TL
    res_y = int(round(RES_FUENTE_X * TL.ALTO_KM / TL.ANCHO_KM))
    t0 = time.time()
    H, _A, _mare, _gt, _kmpx, _cat = TL.generar(RES_FUENTE_X, res_y, seed=SEED)
    print(f"    generado en {time.time() - t0:.0f} s")
    np.save(ruta, H)
    print(f"    cache guardada en {ruta}")
    return H.astype(np.float32)


def dimensiones_km():
    """Lee ANCHO_KM/ALTO_KM de terreno_lunar si esta; si no, los del TFG."""
    try:
        sys.path.insert(0, os.path.abspath(CARPETA_FUENTE))
        import terreno_lunar as TL
        return float(TL.ANCHO_KM), float(TL.ALTO_KM)
    except Exception:
        return 840.0, 630.0


# =========================================================
# 2. REMUESTREO POR PROMEDIADO DE AREA
#
# Resamplear con un filtro de caja exacto en cada eje. Vale para reducir y
# para ampliar, y no introduce el aliasing que daria coger un pixel de cada N.
# =========================================================

def _remuestrear_eje(A, n_out, eje):
    A = np.moveaxis(A, eje, 0)
    n_in = A.shape[0]
    if n_in == n_out:
        return np.moveaxis(A, 0, eje)

    # integral acumulada con un cero delante: C[k] = suma de las k primeras
    C = np.zeros((n_in + 1,) + A.shape[1:], np.float64)
    np.cumsum(A, axis=0, out=C[1:])

    bordes = np.linspace(0.0, n_in, n_out + 1)
    lo, hi = bordes[:-1], bordes[1:]

    def integral(t):
        """Integral de 0 a t, con t en unidades de muestra (fraccionario)."""
        k = np.clip(np.floor(t).astype(np.int64), 0, n_in - 1)
        f = (t - k).reshape((-1,) + (1,) * (A.ndim - 1))
        return C[k] + f * (C[k + 1] - C[k])

    ancho = (hi - lo).reshape((-1,) + (1,) * (A.ndim - 1))
    R = (integral(hi) - integral(lo)) / ancho
    return np.moveaxis(R.astype(np.float32), 0, eje)


def remuestrear(H, ny, nx):
    return _remuestrear_eje(_remuestrear_eje(H, ny, 0), nx, 1)


# =========================================================
# 3. ESCRITOR DE STL BINARIO
# =========================================================

_DT_TRI = np.dtype({
    "names":   ["n", "v", "attr"],
    "formats": ["<3f4", "<9f4", "<u2"],
    "offsets": [0, 12, 48],
    "itemsize": 50,
})


class EscritorSTL:
    """Escribe STL binario a trozos sin tener toda la malla en memoria."""

    def __init__(self, ruta, nombre="malla"):
        self.f = open(ruta, "wb")
        self.f.write(nombre.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        self.f.write(struct.pack("<I", 0))   # se rellena al cerrar
        self.n = 0
        self.ruta = ruta

    def anadir(self, tri, hacia):
        """tri: (N,3,3) float. hacia: (3,) direccion exterior esperada.

        Cada triangulo cuyo normal apunte al lado contrario se da la vuelta.
        Asi no hay que razonar sobre el sentido de giro en ningun sitio: se
        declara hacia donde mira la cara y el codigo se encarga.
        """
        if len(tri) == 0:
            return
        tri = np.ascontiguousarray(tri, np.float64)
        nor = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ln = np.linalg.norm(nor, axis=1)
        bueno = ln > 1e-12
        tri, nor, ln = tri[bueno], nor[bueno], ln[bueno]
        if len(tri) == 0:
            return
        nor /= ln[:, None]
        invertir = nor @ np.asarray(hacia, np.float64) < 0.0
        tri[invertir] = tri[invertir][:, ::-1]
        nor[invertir] *= -1.0

        buf = np.zeros(len(tri), _DT_TRI)
        buf["n"] = nor.astype(np.float32)
        buf["v"] = tri.reshape(-1, 9).astype(np.float32)
        self.f.write(buf.tobytes())
        self.n += len(tri)

    def cerrar(self):
        self.f.seek(80)
        self.f.write(struct.pack("<I", self.n))
        self.f.close()
        return self.n


# =========================================================
# 4. CONSTRUCCION DEL SOLIDO
# =========================================================

def _malla_vertices(z, x, y):
    """(ny,nx,3) con las coordenadas de cada vertice de la superficie."""
    ny, nx = z.shape
    V = np.empty((ny, nx, 3), np.float64)
    V[:, :, 0] = x[None, :]
    V[:, :, 1] = y[:, None]
    V[:, :, 2] = z
    return V


def _tris_superficie(V):
    """Dos triangulos por celda, con la diagonal alternada en tablero de
    ajedrez para que las laderas no queden peinadas siempre en la misma
    direccion (se nota mucho en un relieve tan plano como este)."""
    a = V[:-1, :-1]; b = V[:-1, 1:]; c = V[1:, 1:]; d = V[1:, :-1]
    ny, nx = a.shape[0], a.shape[1]
    par = ((np.arange(ny)[:, None] + np.arange(nx)[None, :]) % 2 == 0)

    t1 = np.where(par[..., None, None],
                  np.stack([a, b, c], axis=-2),
                  np.stack([a, b, d], axis=-2))
    t2 = np.where(par[..., None, None],
                  np.stack([a, c, d], axis=-2),
                  np.stack([b, c, d], axis=-2))
    return np.concatenate([t1.reshape(-1, 3, 3), t2.reshape(-1, 3, 3)], axis=0)


def _tris_pared(borde, z0):
    """Faldon vertical desde una polilinea de borde (N,3) hasta z = z0."""
    arriba = borde
    abajo = borde.copy()
    abajo[:, 2] = z0
    a = arriba[:-1]; b = arriba[1:]; c = abajo[1:]; d = abajo[:-1]
    return np.concatenate([np.stack([a, b, c], axis=-2),
                           np.stack([a, c, d], axis=-2)], axis=0)


def _tris_fondo(contorno, z0):
    """Abanico desde el centro hacia el contorno. Usa los MISMOS vertices que
    la pared, asi que no quedan uniones en T y el solido cierra.

    El vertice central va en el centro del rectangulo y no en una esquina: si
    se pone en una esquina, todos los triangulos del tramo que sale de ella
    son de area nula, al quitarlos quedan aristas sueltas y el laminador ve
    la pieza como abierta."""
    P = contorno.copy()
    P[:, 2] = z0
    c = np.array([0.5 * (P[:, 0].min() + P[:, 0].max()),
                  0.5 * (P[:, 1].min() + P[:, 1].max()), z0])
    a = np.repeat(c[None, :], len(P) - 1, axis=0)
    return np.stack([a, P[:-1], P[1:]], axis=-2)


def construir_pieza(ruta, z, x, y, z0, nombre):
    V = _malla_vertices(z, x, y)
    ny, nx = z.shape

    # contorno cerrado recorriendo el borde de la rejilla
    contorno = np.concatenate([
        V[0, :],            # fila superior, de izquierda a derecha
        V[1:, -1],          # columna derecha hacia abajo
        V[-1, -2::-1],      # fila inferior de vuelta
        V[-2::-1, 0],       # columna izquierda hacia arriba
    ], axis=0)

    w = EscritorSTL(ruta, nombre)
    w.anadir(_tris_superficie(V), (0, 0, 1))

    # el faldon se emite por tramos para poder darle a cada uno su exterior
    i = 0
    for tramo, hacia in ((V[0, :], (0, 1, 0)),
                         (V[:, -1], (1, 0, 0)),
                         (V[-1, ::-1], (0, -1, 0)),
                         (V[::-1, 0], (-1, 0, 0))):
        w.anadir(_tris_pared(np.ascontiguousarray(tramo), z0), hacia)
        i += 1

    w.anadir(_tris_fondo(contorno, z0), (0, 0, -1))
    n = w.cerrar()
    return n, os.path.getsize(ruta)


# =========================================================
# 5. VERIFICACION
# =========================================================

def verificar(z, z0, paso_mm, km_por_mm):
    print("\n[5] Verificacion")
    espesor_min = float(z.min() - z0)
    print(f"    Material bajo el punto mas bajo del mapa : {espesor_min:.2f} mm")
    if espesor_min < MIN_MATERIAL_MM:
        print(f"    !! POR DEBAJO de {MIN_MATERIAL_MM} mm: sube ESPESOR_BASE_MM")
    else:
        print(f"    OK (umbral {MIN_MATERIAL_MM} mm)")

    # la superficie nunca baja de z.min(), asi que ningun crater toca el fondo
    print(f"    Altura total de la maqueta               : {z.max() - z0:.2f} mm")
    print(f"    Relieve (de valle a cumbre)              : {z.max() - z.min():.2f} mm")

    n_capas = (z.max() - z.min()) / ALTURA_CAPA_MM
    print(f"    Capas de relieve a {ALTURA_CAPA_MM} mm               : {n_capas:.0f}")

    # que crateres del catalogo sobreviven a la resolucion de la impresora
    csv_cat = os.path.join(CARPETA_FUENTE, "crateres_catalogo.csv")
    if os.path.isfile(csv_cat):
        D, fr = [], []
        with open(csv_cat, newline="") as fh:
            for r in csv.DictReader(fh):
                D.append(float(r["D_km"])); fr.append(float(r["frescura"]))
        D = np.array(D); fr = np.array(fr)
        # profundidad de Pike, atenuada por frescura como en terreno_lunar
        d0 = np.where(D < 18.0, 0.196 * D ** 1.010, 1.044 * D ** 0.301)
        prof_mm = d0 * (0.25 + 0.75 * fr) * 1000.0 / ESCALA * 1000.0 * EXAGERACION_Z
        diam_mm = D * 1000.0 / ESCALA * 1000.0
        visible = (prof_mm >= ALTURA_CAPA_MM) & (diam_mm >= 3 * paso_mm)
        print(f"    Crateres del catalogo                    : {len(D)}")
        print(f"    ... que la impresora puede marcar        : {int(visible.sum())} "
              f"({100.0 * visible.mean():.1f} %)")
        print(f"    ... el mas profundo mide                 : {prof_mm.max():.2f} mm")
        if visible.mean() < 0.5:
            print("    !! Mas de la mitad se pierden. Para que se aprecien hay")
            print("       que subir EXAGERACION_Z (2-3) o imprimir un recorte.")


# =========================================================
# 6. PREVIA
# =========================================================

def guardar_previa(z, cortes_x, cortes_y, x, y, ruta):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("    (sin Pillow: me salto la previa)")
        return
    gy, gx = np.gradient(z.astype(np.float32), PASO_MM)
    az, el = math.radians(315.0), math.radians(30.0)
    pend = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(gy, -gx)
    sh = np.clip(np.sin(el) * np.cos(pend) +
                 np.cos(el) * np.sin(pend) * np.cos(az - asp), 0, 1)
    img = (255 * (0.08 + 0.92 * sh)).astype(np.uint8)
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    for j in cortes_x[1:-1]:
        d.line([(j, 0), (j, z.shape[0] - 1)], fill=(255, 60, 60), width=3)
    for i in cortes_y[1:-1]:
        d.line([(0, i), (z.shape[1] - 1, i)], fill=(255, 60, 60), width=3)

    # nombre de cada pieza en su centro, para montarlo sin dudar
    try:
        fnt = ImageFont.truetype("DejaVuSans-Bold.ttf", 46)
    except Exception:
        fnt = ImageFont.load_default()
    for ti in range(len(cortes_y) - 1):
        for tj in range(len(cortes_x) - 1):
            cx = 0.5 * (cortes_x[tj] + cortes_x[tj + 1])
            cy = 0.5 * (cortes_y[ti] + cortes_y[ti + 1])
            d.text((cx, cy), f"{chr(ord('A') + ti)}{tj + 1}", font=fnt,
                   fill=(255, 40, 40), anchor="mm",
                   stroke_width=3, stroke_fill=(255, 255, 255))
    im.save(ruta)
    print(f"    previa -> {ruta}")


# =========================================================
# PRINCIPAL
# =========================================================

class _Tee:
    """Duplica la salida en pantalla y en un archivo, de modo que el
    resultado quede legible aunque se haya lanzado desde Blender sin la
    consola del sistema abierta."""

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
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    reg = _Tee(os.path.join(CARPETA_SALIDA, "registro.txt"))
    try:
        _main()
    finally:
        reg.cerrar()


def _main():

    ancho_km, alto_km = dimensiones_km()
    mm_por_km = 1000.0 * 1000.0 / ESCALA      # km -> m -> mm -> /escala
    ancho_mm = ancho_km * mm_por_km
    alto_mm = alto_km * mm_por_km

    print("=" * 62)
    print(f"Terreno  : {ancho_km:.0f} x {alto_km:.0f} km")
    print(f"Escala   : 1:{ESCALA:,.0f}   ->  {ancho_mm:.0f} x {alto_mm:.0f} mm")
    print(f"Exag. Z  : {EXAGERACION_Z:g}x")
    print("=" * 62)

    H = cargar_heightmap()

    # --- 1b. recorte opcional ---
    if RECORTE_KM is not None:
        x0k, y0k, x1k, y1k = RECORTE_KM
        km_px = ancho_km / H.shape[1]
        j0 = int(round(x0k / km_px)); j1 = int(round(x1k / km_px))
        i0 = int(round((alto_km - y1k) / km_px))
        i1 = int(round((alto_km - y0k) / km_px))
        j0, j1 = max(0, j0), min(H.shape[1], j1)
        i0, i1 = max(0, i0), min(H.shape[0], i1)
        H = H[i0:i1, j0:j1]
        ancho_km = (j1 - j0) * km_px
        alto_km = (i1 - i0) * km_px
        ancho_mm = ancho_km * mm_por_km
        alto_mm = alto_km * mm_por_km
        print(f"    Recorte: x {x0k}-{x1k} km, y {y0k}-{y1k} km  ->  "
              f"{ancho_km:.1f} x {alto_km:.1f} km = {ancho_mm:.0f} x {alto_mm:.0f} mm")

    # --- 2. remuestreo ---
    nx = int(round(ancho_mm / PASO_MM)) + 1
    ny = int(round(alto_mm / PASO_MM)) + 1
    print(f"\n[2] Remuestreo por promediado de area -> {nx} x {ny} muestras "
          f"({PASO_MM} mm/muestra)")
    Hp = remuestrear(H, ny, nx)
    del H

    # --- 3. escala vertical ---
    # H esta en metros: m -> mm es x1000, y luego /ESCALA.
    z = Hp.astype(np.float64) * 1000.0 / ESCALA * EXAGERACION_Z
    km_por_mm = 1.0 / mm_por_km
    print(f"[3] Z: {Hp.min():.0f} .. {Hp.max():.0f} m  ->  "
          f"{z.min():.2f} .. {z.max():.2f} mm")

    # --- 4. base ---
    # Un unico desplazamiento para TODO el mapa. El minimo global queda a
    # ESPESOR_BASE_MM y el fondo a cero.
    z = z - z.min() + ESPESOR_BASE_MM
    z0 = 0.0
    print(f"[4] Base: fondo plano en z=0, punto mas bajo del terreno a "
          f"{ESPESOR_BASE_MM} mm")

    verificar(z, z0, PASO_MM, km_por_mm)

    # coordenadas. Fila 0 del array = borde superior del mapa (y = alto)
    x = np.arange(nx) * PASO_MM
    y = alto_mm - np.arange(ny) * PASO_MM

    # --- troceado ---
    util_x = (CAMA_X_MM - 2 * MARGEN_CAMA_MM) if CAMA_X_MM else None
    util_y = (CAMA_Y_MM - 2 * MARGEN_CAMA_MM) if CAMA_Y_MM else None
    if PIEZAS_X and PIEZAS_Y:
        ntx, nty = int(PIEZAS_X), int(PIEZAS_Y)
    elif util_x and util_y:
        ntx = int(math.ceil(ancho_mm / util_x))
        nty = int(math.ceil(alto_mm / util_y))
    else:
        ntx = nty = 1
    cortes_x = [int(round(k * (nx - 1) / ntx)) for k in range(ntx + 1)]
    cortes_y = [int(round(k * (ny - 1) / nty)) for k in range(nty + 1)]

    pieza_x, pieza_y = ancho_mm / ntx, alto_mm / nty
    print(f"\n[6] Troceado: {ntx} columnas x {nty} filas = {ntx * nty} piezas "
          f"de {pieza_x:.1f} x {pieza_y:.1f} mm")
    if util_x and util_y:
        if pieza_x <= util_x and pieza_y <= util_y:
            print(f"    Caben en la cama de {CAMA_X_MM:.0f} x {CAMA_Y_MM:.0f} mm "
                  f"({MARGEN_CAMA_MM:.0f} mm de margen por lado): OK")
        else:
            print(f"    !! NO CABEN en {util_x:.0f} x {util_y:.0f} mm utiles: "
                  f"hay que subir PIEZAS_X / PIEZAS_Y, o ponerlas a None.")

    if GUARDAR_PREVIA:
        guardar_previa(z, cortes_x, cortes_y, x, y,
                       os.path.join(CARPETA_SALIDA, f"{PREFIJO}_previa.png"))

    filas = []
    tot_tri = tot_bytes = 0
    for ti in range(nty):
        for tj in range(ntx):
            i0, i1 = cortes_y[ti], cortes_y[ti + 1]
            j0, j1 = cortes_x[tj], cortes_x[tj + 1]
            zt = z[i0:i1 + 1, j0:j1 + 1]
            xt = x[j0:j1 + 1].copy()
            yt = y[i0:i1 + 1].copy()
            if ORIGEN_LOCAL:
                xt -= xt.min()
                yt -= yt.min()
            # fila 1 = la de arriba del mapa -> nomenclatura A1, B1...
            nombre = f"{PREFIJO}_{chr(ord('A') + ti)}{tj + 1}"
            ruta = os.path.join(CARPETA_SALIDA, nombre + ".stl")
            n, b = construir_pieza(ruta, zt, xt, yt, z0, nombre)
            tot_tri += n; tot_bytes += b
            ancho_p = xt.max() - xt.min()
            alto_p = yt.max() - yt.min()
            print(f"    {nombre:<22} {ancho_p:6.1f} x {alto_p:6.1f} x "
                  f"{zt.max() - z0:5.2f} mm   {n:>9,} tri   {b / 2**20:6.1f} MB")
            filas.append(dict(pieza=nombre, col=tj + 1, fila=chr(ord('A') + ti),
                              x0_km=round(j0 * PASO_MM * km_por_mm, 2),
                              y0_km=round((alto_mm - i1 * PASO_MM) * km_por_mm, 2),
                              ancho_mm=round(ancho_p, 2), alto_mm=round(alto_p, 2),
                              espesor_mm=round(float(zt.max() - z0), 2),
                              triangulos=n))

    if GUARDAR_CSV_PIEZAS and filas:
        ruta = os.path.join(CARPETA_SALIDA, f"{PREFIJO}_piezas.csv")
        with open(ruta, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader(); w.writerows(filas)
        print(f"    indice -> {ruta}")

    # volumen aproximado: base maciza + relieve por encima
    area_mm2 = ancho_mm * alto_mm
    vol_mm3 = float((z - z0).mean()) * area_mm2
    print(f"\n[7] Total: {tot_tri:,} triangulos, {tot_bytes / 2**20:.0f} MB de STL")
    print(f"    Volumen macizo: {vol_mm3 / 1000.0:.0f} cm3  "
          f"({vol_mm3 * 1.24 / 1000.0:.0f} g de PLA al 100 %, "
          f"~{vol_mm3 * 1.24 * 0.25 / 1000.0:.0f} g al 25 % de relleno)")
    print("    Imprime con el fondo plano sobre la cama: no necesita soportes.")
    print("\n    Todo esto queda tambien en "
          + os.path.join(CARPETA_SALIDA, "registro.txt"))


if __name__ == "__main__":
    main()
