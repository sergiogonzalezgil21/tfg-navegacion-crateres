"""
Generador de superficie lunar. Devuelve, en numpy y sin dependencias exoticas,
el heightmap en metros, el mapa de albedo y la verdad de referencia de los
crateres en km. Se usa igual desde Blender que desde Python normal.

Lo que modela:
  * ley de potencias para los tamanos de crater,  N(>D) ~ D^-b
  * crateres simples (cuenco) y complejos (fondo plano, terrazas, pico
    central), con el cambio de morfologia hacia los 18 km como en la Luna
  * perfil profundidad/diametro de Pike, borde elevado y manto de eyecta
  * degradacion por edad: los viejos mas someros, los jovenes encima
  * mares basalticos, crestas de arruga y rugosidad de regolito multiescala
  * albedo con mares oscuros, rayos de eyecta fresca y moteado
"""

import numpy as np

# =========================================================
# CONFIGURACION
# =========================================================

# El tamano del terreno se puede cambiar desde fuera antes de llamar a
# generar():   TL.ANCHO_KM = 840.0 ; TL.ALTO_KM = 630.0
ANCHO_KM = 840.0          # eje X del terreno
ALTO_KM = 630.0           # eje Y del terreno
# 8192 / 840 km = 102 m/pixel.
# Podria generarse a 12288 (68 m/px, igual que el mapa pequeno), pero entonces
# detalle.png tendria 113 Mpx y Blender guarda las texturas como float RGBA:
# 1.8 GB solo para el bump, otro tanto para el albedo. Con 500 frames por
# renderizar eso no compensa. A 102 m/px el bump sigue siendo mas fino que la
# malla (280 m/vertice) y va parejo con lo que resuelve la camara (78-115 m/px
# entre 110 y 160 km de altura).
RES_X = 8192
RES_Y = 6144

SEED = 20260818

# Poblacion de crateres de fondo (los que dan realismo, no se usan de landmark)
D_MIN_KM = 0.28           # diametro minimo
D_MAX_KM = 45.0           # diametro maximo del fondo
EXPONENTE_SFD = 2.1       # N(>D) ~ D^-b ; 2 aprox lunar
# 0.75 crateres/km2 sobre los 529 200 km2 del terreno.
N_CRATERES_FONDO = 400000

# Solo entran al catalogo los crateres que la camara puede llegar a resolver.
# A 130 km de altura un pixel son 78 m, y una caja de menos de 12 px no es
# aprendible: eso son 0.94 km. Guardar los cientos de miles de crateres mas
# pequenos solo hincharia el CSV y la memoria.
D_MIN_CATALOGO_KM = 0.8

D_TRANSICION_KM = 18.0    # simple -> complejo (Luna: 15-20 km)

# Crateres de referencia (landmark). Con LANDMARKS_AUTO se reparten por todo
# el terreno respetando una separacion minima, en vez de fijarlos a mano.
LANDMARKS_AUTO = True
N_LANDMARKS = 180
LANDMARK_SEP_MIN_KM = 34.0    # separacion minima entre landmarks
LANDMARK_MARGEN_KM = 25.0     # distancia minima al borde del terreno
LANDMARK_D_MIN_KM = 16.0
LANDMARK_D_MAX_KM = 32.0

# Procesado por bandas de filas. A 12288x9216 cada array float32 ocupa 453 MB,
# asi que las operaciones que crean temporales de tamano completo hay que
# hacerlas a trozos o la generacion se come toda la RAM.
BANDA_FILAS = 1024

# Crateres landmark: los 18 del TFG, en km, sobre el sistema (0..280, 0..210)
LANDMARKS = [
    ("c1",   25.0, 170.0, 18.0), ("c2",   70.0, 170.0, 23.0),
    ("c3",   30.0, 115.0, 20.0), ("c4",   95.0, 130.0, 21.0),
    ("c5",   60.0,  70.0, 26.0), ("c6",   22.0,  30.0, 16.0),
    ("c7",   95.0,  55.0, 23.0), ("c8",  125.0,  25.0, 20.0),
    ("c9",  150.0,  65.0, 24.5), ("c10", 180.0,  28.0, 17.0),
    ("c11", 170.0, 110.0, 20.5), ("c12", 225.0,  55.0, 21.0),
    ("c13", 215.0, 110.0, 28.0), ("c14", 248.0, 125.0, 20.0),
    ("c15", 225.0, 175.0, 30.0), ("c16", 175.0, 168.0, 18.0),
    ("c17", 110.0, 195.0, 18.5), ("c18", 140.0, 195.0, 22.0),
]


# =========================================================
# RUIDO FRACTAL (fBm) POR BANDAS
#
# El ruido se evalua por bandas de filas y no sobre la imagen entera.
# Evaluarlo de golpe mantiene vivos siete arrays del tamano completo a la vez
# (las cuatro esquinas de la interpolacion, los dos parciales y el resultado);
# a 12288x9216 son 453 MB cada uno, o sea 3.2 GB para una sola octava. Por
# bandas, el pico de memoria depende de BANDA_FILAS y no del tamano del
# terreno.
# =========================================================

def _capas_fbm(rng, ny, nx, celdas_base, octavas, lacunaridad, ganancia):
    """Sortea las rejillas de cada octava. Son pequenas: caben de sobra."""
    capas = []
    amp = 1.0
    total = 0.0
    cy = celdas_base
    cx = max(1, int(round(celdas_base * nx / ny)))
    for _ in range(octavas):
        if cy > ny or cx > nx:
            break
        capas.append((rng.random((cy + 1, cx + 1)).astype(np.float32), cy, cx, amp))
        total += amp
        amp *= ganancia
        cy = int(cy * lacunaridad)
        cx = int(cx * lacunaridad)
    return capas, total


def _fbm_banda(capas, total, ny, nx, y0, y1, paso_x=1):
    """Evalua el fBm en las filas [y0, y1). Con paso_x > 1 submuestrea en X."""
    xs = np.arange(0, nx, paso_x, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    acc = np.zeros((len(ys), len(xs)), np.float32)

    for (g, cy, cx, amp) in capas:
        xi = xs * (cx / nx)
        ix = np.floor(xi).astype(np.int32)
        tx = xi - ix
        tx = tx * tx * (3.0 - 2.0 * tx)      # smoothstep, para no ver la rejilla

        yi = ys * (cy / ny)
        iy = np.floor(yi).astype(np.int32)
        ty = yi - iy
        ty = ty * ty * (3.0 - 2.0 * ty)

        f0 = g[iy][:, ix]
        f1 = g[iy][:, ix + 1]
        f2 = g[iy + 1][:, ix]
        f3 = g[iy + 1][:, ix + 1]

        a = f0 + (f1 - f0) * tx[None, :]
        b = f2 + (f3 - f2) * tx[None, :]
        acc += amp * (a + (b - a) * ty[:, None])

    acc /= total
    return acc


def sumar_fbm(dest, rng, celdas_base, octavas, amplitud,
              lacunaridad=2.0, ganancia=0.5, banda=None):
    """Suma a 'dest' un fBm centrado en cero y de recorrido 'amplitud'.

    Se normaliza estirando a [0,1] igual que hacia la version antigua, pero el
    minimo y el maximo se estiman sobre un submuestreo (una de cada 8 columnas
    y bandas salteadas). Para un simple estiramiento de contraste eso es
    indistinguible del calculo exacto y evita una segunda pasada completa.
    """
    ny, nx = dest.shape
    if banda is None:
        banda = BANDA_FILAS

    capas, total = _capas_fbm(rng, ny, nx, celdas_base, octavas,
                              lacunaridad, ganancia)

    lo, hi = np.inf, -np.inf
    for y0 in range(0, ny, banda * 4):
        y1 = min(ny, y0 + 64)
        m = _fbm_banda(capas, total, ny, nx, y0, y1, paso_x=8)
        lo = min(lo, float(m.min()))
        hi = max(hi, float(m.max()))
    rango = max(hi - lo, 1e-6)

    for y0 in range(0, ny, banda):
        y1 = min(ny, y0 + banda)
        m = _fbm_banda(capas, total, ny, nx, y0, y1)
        m -= lo
        m /= rango
        m -= 0.5
        m *= amplitud
        dest[y0:y1] += m


def fbm(rng, ny, nx, celdas_base=4, octavas=9, lacunaridad=2.0, ganancia=0.5):
    """Version que devuelve el array entero. Solo para tamanos pequenos."""
    out = np.zeros((ny, nx), np.float32)
    sumar_fbm(out, rng, celdas_base, octavas, 1.0, lacunaridad, ganancia)
    out += 0.5
    return np.clip(out, 0.0, 1.0)


# =========================================================
# CRATERES DE REFERENCIA (LANDMARK)
# =========================================================

def generar_landmarks(rng, ancho_km, alto_km, n_objetivo, sep_min_km,
                      margen_km, d_min, d_max):
    """Reparte n crateres de referencia por todo el terreno.

    Dos cosas importan aqui:

    - Separacion minima. Si dos landmarks se solapan, el detector los ve como
      uno y el matching se rompe.
    - Que NO formen una rejilla regular. El algoritmo del TFG empareja
      tripletas por su forma; en una rejilla hay miles de triangulos
      congruentes y el emparejamiento se vuelve ambiguo. Por eso se colocan
      por lanzamiento de dardos con rechazo, que da separacion garantizada
      pero posiciones irregulares.
    """
    x_min, x_max = margen_km, ancho_km - margen_km
    y_min, y_max = margen_km, alto_km - margen_km

    puntos = []
    sep = sep_min_km
    intentos = 0
    limite = n_objetivo * 400

    while len(puntos) < n_objetivo and intentos < limite:
        intentos += 1
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        ok = True
        for (px, py, _) in puntos:
            if (x - px) ** 2 + (y - py) ** 2 < sep * sep:
                ok = False
                break
        if ok:
            puntos.append((x, y, float(rng.uniform(d_min, d_max))))

        # si cuesta demasiado, se relaja la separacion en vez de fallar
        if intentos % (n_objetivo * 40) == 0 and len(puntos) < n_objetivo:
            sep *= 0.92

    return [(f"c{i+1}", x, y, d) for i, (x, y, d) in enumerate(puntos)]


# =========================================================
# PERFILES DE CRATER
# =========================================================

def _perfil_crater(r, D_km, frescura):
    """
    Perfil radial de un crater. r es distancia normalizada al radio del borde.
    Devuelve altura en metros (negativa dentro, positiva en borde y eyecta).

    frescura: 1.0 = recien formado y nitido, 0.0 = casi borrado.
    """
    D_m = D_km * 1000.0

    # Profundidad segun las relaciones morfometricas de Pike (1977) para la
    # Luna. La regla rapida d/D = 0.2 da 4 km de hondo para un crater de 20 km,
    # el doble de lo real; con Pike sale 2.6 km y las sombras dejan de ser
    # agujeros negros.
    if D_km < D_TRANSICION_KM:
        prof = 0.196 * (D_km ** 1.010) * 1000.0        # simples
    else:
        prof = 1.044 * (D_km ** 0.301) * 1000.0        # complejos

    h_borde = 0.032 * D_m          # borde elevado, ~4% del diametro
    prof *= (0.25 + 0.75 * frescura)
    h_borde *= (0.15 + 0.85 * frescura)

    z = np.zeros_like(r)

    dentro = r <= 1.0
    fuera = ~dentro

    if D_km < D_TRANSICION_KM:
        # ---- CRATER SIMPLE: cuenco parabolico ----
        rr = r[dentro]
        z[dentro] = -prof + (prof + h_borde) * rr * rr
    else:
        # ---- CRATER COMPLEJO: fondo plano + terrazas + pico central ----
        rr = r[dentro]
        r_fondo = 0.40 + 0.10 * (1.0 - frescura)   # los viejos se rellenan mas

        # Transicion fondo->pared con smootherstep (6t^5-15t^4+10t^3). La
        # smoothstep normal deja discontinua la segunda derivada y eso salia
        # en el render como un circulo nitido en el borde del fondo.
        t = np.clip((rr - r_fondo) / (1.0 - r_fondo), 0.0, 1.0)
        s5 = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
        zz = -prof + (prof + h_borde) * s5

        # el fondo no es un plano perfecto: leve concavidad hacia el centro
        zz -= 0.05 * prof * np.clip(1.0 - rr / r_fondo, 0.0, 1.0) ** 2

        # terrazas: escalones concentricos en la pared, solo si esta fresco
        if frescura > 0.45:
            zz += 0.010 * prof * frescura * np.sin(2.0 * np.pi * 2.0 * t) * (t > 0)

        # Pico central: en la Luna solo aparece por encima de ~25 km.
        # Perfil coseno elevado en vez de cono, para que no salga un pezon.
        if D_km > 25.0 and frescura > 0.40:
            r_pico = 0.22
            u = np.clip(1.0 - rr / r_pico, 0.0, 1.0)
            zz += 0.30 * prof * frescura * (u * u * (3.0 - 2.0 * u))

        z[dentro] = zz

    # ---- MANTO DE EYECTA: decae como r^-3, con corte suave ----
    rr = r[fuera]
    z[fuera] = h_borde * np.power(rr, -3.8)

    return z


def estampar_crater(H, x_px, y_px, R_px, D_km, frescura, rng, alcance=2.2):
    """Suma el perfil de un crater al heightmap H, trabajando en una ventana local."""
    ny, nx = H.shape
    rad = int(np.ceil(R_px * alcance))
    if rad < 2:
        return None

    x0 = max(0, int(x_px) - rad); x1 = min(nx, int(x_px) + rad + 1)
    y0 = max(0, int(y_px) - rad); y1 = min(ny, int(y_px) + rad + 1)
    if x1 <= x0 or y1 <= y0:
        return None

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = (xx - x_px).astype(np.float32)
    dy = (yy - y_px).astype(np.float32)

    # elipticidad leve + rotacion: ningun crater real es un circulo perfecto
    ang = rng.uniform(0, np.pi)
    e = rng.uniform(0.92, 1.08)
    ca, sa = np.cos(ang), np.sin(ang)
    xr = dx * ca + dy * sa
    yr = (-dx * sa + dy * ca) / e

    r = np.sqrt(xr * xr + yr * yr) / R_px

    # irregularidad del borde: el radio efectivo ondula con el azimut
    if R_px > 6:
        theta = np.arctan2(yr, xr)
        # suma de armonicos con caida 1/k: contorno irregular pero no poligonal
        onda = np.zeros_like(theta)
        for k in (5, 7, 9, 13, 17, 23):
            onda += (0.016 / np.sqrt(k)) * np.sin(k * theta + rng.uniform(0, 6.28))
        # ventana estrecha en r=1: solo el borde es irregular, el fondo es liso
        w = np.exp(-((r - 1.0) ** 2) / (2 * 0.16 ** 2))
        r = r / (1.0 + w * onda)

    r = np.maximum(r, 1e-3)
    z = _perfil_crater(r, D_km, frescura)

    # La eyecta real sale en estrias radiales, no como un domo liso. Sin esto
    # cada crater quedaba rodeado de un anillo brillante tipo huevo frito.
    if R_px > 4:
        th2 = np.arctan2(yr, xr)
        amp = np.zeros_like(th2)
        for kk in rng.choice(np.arange(4, 20), size=4, replace=False):
            amp += (0.055 / np.sqrt(kk)) * np.sin(kk * th2 + rng.uniform(0, 6.28))
        # entra progresivamente a partir de r=1 para no crear un salto en el borde
        rampa = np.clip((r - 1.0) / 0.35, 0.0, 1.0)
        m_ej = r > 1.0
        z[m_ej] *= (1.0 + amp[m_ej] * rampa[m_ej])

    # Apagado SUAVE de la eyecta hacia el borde de la ventana. Cortarla en
    # seco dejaba un escalon circular de decenas de metros que el hillshade
    # delataba como una red de lineas finas por todo el mapa.
    fade = np.clip((alcance - r) / (alcance - 1.0), 0.0, 1.0)
    fade = fade ** 1.5
    fade = fade * fade * (3.0 - 2.0 * fade)
    fuera_m = r > 1.0
    z[fuera_m] *= fade[fuera_m]

    H[y0:y1, x0:x1] += z
    return (x0, x1, y0, y1, r)


# =========================================================
# GENERACION COMPLETA
# =========================================================

def generar(res_x=None, res_y=None, seed=SEED, n_fondo=None,
            verbose=True, banda=None):
    rng = np.random.default_rng(seed)

    if res_x is None:
        res_x = RES_X
    if res_y is None:
        res_y = int(round(res_x * ALTO_KM / ANCHO_KM))
    if n_fondo is None:
        n_fondo = N_CRATERES_FONDO
    if banda is None:
        banda = BANDA_FILAS

    km_por_px = ANCHO_KM / res_x
    px_por_km = 1.0 / km_por_px

    if verbose:
        print(f"Terreno: {ANCHO_KM:.0f} x {ALTO_KM:.0f} km")
        print(f"Resolucion: {res_x}x{res_y}  ->  {km_por_px*1000:.1f} m/pixel")
        print(f"Memoria por array: {res_x*res_y*4/2**20:.0f} MB")

    H = np.zeros((res_y, res_x), np.float32)      # altura en metros

    # ---------------------------------------------------------
    # 1) TOPOGRAFIA REGIONAL: ondulaciones largas de las tierras altas
    # ---------------------------------------------------------
    if verbose: print("Topografia regional...")
    sumar_fbm(H, rng, celdas_base=3, octavas=7, amplitud=2200, banda=banda)

    # ---------------------------------------------------------
    # 2) MARES: cuencas amplias, planas y hundidas
    # ---------------------------------------------------------
    if verbose: print("Mares...")
    mare = np.zeros((res_y, res_x), np.float32)

    # el numero de mares escala con el area del terreno
    area_rel = (ANCHO_KM * ALTO_KM) / (280.0 * 210.0)
    n_mares = int(rng.integers(3, 5) * max(1.0, area_rel ** 0.5))

    params = []
    for _ in range(n_mares):
        params.append(dict(
            cx=rng.uniform(0.12, 0.88) * res_x,
            cy=rng.uniform(0.12, 0.88) * res_y,
            rx=rng.uniform(0.07, 0.13) * res_x * max(1.0, area_rel ** 0.25),
            ry=rng.uniform(0.07, 0.13) * res_y * max(1.0, area_rel ** 0.25),
            ang=rng.uniform(0, np.pi),
            f3=rng.uniform(0, 6.28), f7=rng.uniform(0, 6.28)))

    xs = np.arange(res_x, dtype=np.float32)
    for y0 in range(0, res_y, banda):
        y1 = min(res_y, y0 + banda)
        ys = np.arange(y0, y1, dtype=np.float32)[:, None]
        m_b = np.zeros((y1 - y0, res_x), np.float32)

        for p in params:
            dx = xs[None, :] - p["cx"]
            dy = ys - p["cy"]
            ca, sa = np.cos(p["ang"]), np.sin(p["ang"])
            xr = (dx * ca + dy * sa) / p["rx"]
            yr = (-dx * sa + dy * ca) / p["ry"]
            d = np.sqrt(xr * xr + yr * yr)

            theta = np.arctan2(yr, xr)
            d /= (1.0 + 0.16 * np.sin(3 * theta + p["f3"])
                      + 0.09 * np.sin(7 * theta + p["f7"]))

            m = np.clip((1.10 - d) / 0.30, 0.0, 1.0)
            m *= m * (3.0 - 2.0 * m)
            np.maximum(m_b, m, out=m_b)

        mare[y0:y1] = m_b

        # aplicar el mar al relieve en la misma banda, para no crear
        # temporales del tamano completo
        hb = H[y0:y1]
        hb -= m_b * 900.0
        hb *= (1.0 - 0.80 * m_b)
        hb -= m_b * 500.0

    # ---------------------------------------------------------
    # 3) CRESTAS DE ARRUGA dentro de los mares
    # ---------------------------------------------------------
    if verbose: print("Crestas de arruga...")
    for _ in range(rng.integers(4, 8)):
        n_pts = rng.integers(5, 10)
        px = rng.uniform(0.1, 0.9) * res_x
        py = rng.uniform(0.1, 0.9) * res_y
        ang = rng.uniform(0, 2 * np.pi)
        pts = [(px, py)]
        for _ in range(n_pts - 1):
            ang += rng.uniform(-0.5, 0.5)
            paso = rng.uniform(0.05, 0.12) * res_x
            px += np.cos(ang) * paso
            py += np.sin(ang) * paso
            pts.append((px, py))

        ancho = rng.uniform(0.010, 0.022) * res_x
        alt = rng.uniform(120.0, 320.0)

        for i in range(len(pts) - 1):
            ax, ay = pts[i]; bx, by = pts[i + 1]
            x0 = int(max(0, min(ax, bx) - ancho)); x1 = int(min(res_x, max(ax, bx) + ancho))
            y0 = int(max(0, min(ay, by) - ancho)); y1 = int(min(res_y, max(ay, by) + ancho))
            if x1 <= x0 or y1 <= y0:
                continue
            wy, wx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            abx = bx - ax; aby = by - ay
            ab2 = abx * abx + aby * aby
            if ab2 == 0:
                continue
            t = np.clip(((wx - ax) * abx + (wy - ay) * aby) / ab2, 0.0, 1.0)
            dd = np.hypot(wx - (ax + t * abx), wy - (ay + t * aby))
            u = np.clip(dd / ancho, 0.0, 1.0)
            perfil = (1.0 - u * u) ** 2
            # solo se elevan donde hay mar
            H[y0:y1, x0:x1] += alt * perfil * mare[y0:y1, x0:x1]

    # ---------------------------------------------------------
    # 4) POBLACION DE CRATERES DE FONDO (ley de potencias, por edad)
    # ---------------------------------------------------------
    if verbose: print(f"Poblacion de crateres ({n_fondo})...")

    # muestreo de la ley de potencias N(>D) ~ D^-b por transformada inversa
    u = rng.random(n_fondo)
    b = EXPONENTE_SFD
    Dmin, Dmax = D_MIN_KM, D_MAX_KM
    Ds = (Dmin ** (-b) - u * (Dmin ** (-b) - Dmax ** (-b))) ** (-1.0 / b)

    # De grande a pequeno NO: queremos orden por EDAD. Los viejos primero,
    # para que los jovenes se superpongan encima. Edad y frescura anticorreladas.
    edades = rng.random(n_fondo)                 # 1 = mas viejo
    orden = np.argsort(-edades)

    xs = rng.random(n_fondo) * res_x
    ys = rng.random(n_fondo) * res_y

    rayos = []      # crateres frescos y grandes que tendran rayos brillantes
    catalogo = []   # TODOS los crateres estampados, para generar etiquetas YOLO

    for idx in orden:
        D = float(Ds[idx])
        R_px = 0.5 * D * px_por_km
        if R_px < 1.2:
            continue

        # los crateres viejos estan mas degradados
        frescura = float(np.clip(1.0 - edades[idx] + rng.normal(0, 0.12), 0.05, 1.0))

        # dentro de los mares hay MUCHOS menos crateres: la lava los borro
        m_local = mare[int(ys[idx]) % res_y, int(xs[idx]) % res_x]
        if m_local > 0.35 and rng.random() < 0.80 * m_local:
            continue

        estampar_crater(H, xs[idx], ys[idx], R_px, D, frescura, rng)
        if D >= D_MIN_CATALOGO_KM:
            catalogo.append(dict(x_km=float(xs[idx]) * km_por_px,
                                 y_km=ALTO_KM - float(ys[idx]) * km_por_px,
                                 D_km=D, frescura=frescura, tipo="fondo"))

        if frescura > 0.90 and D > 6.0:
            rayos.append((xs[idx], ys[idx], R_px, frescura))

    # ---------------------------------------------------------
    # 5) CRATERES LANDMARK: los ultimos, frescos y bien marcados
    # ---------------------------------------------------------
    if LANDMARKS_AUTO:
        landmarks = generar_landmarks(
            rng, ANCHO_KM, ALTO_KM, N_LANDMARKS, LANDMARK_SEP_MIN_KM,
            LANDMARK_MARGEN_KM, LANDMARK_D_MIN_KM, LANDMARK_D_MAX_KM)
    else:
        landmarks = LANDMARKS

    if verbose:
        print(f"Crateres landmark... ({len(landmarks)})")
    gt = []
    for nombre, x_km, y_km, D_km in landmarks:
        x_px = x_km * px_por_km
        y_px = (ALTO_KM - y_km) * px_por_km      # imagen: Y hacia abajo
        R_px = 0.5 * D_km * px_por_km
        frescura = float(rng.uniform(0.86, 1.0))
        estampar_crater(H, x_px, y_px, R_px, D_km, frescura, rng)
        rayos.append((x_px, y_px, R_px, frescura))
        gt.append(dict(nombre=nombre, x_km=x_km, y_km=y_km, D_km=D_km,
                       x_px=x_px, y_px=y_px, R_px=R_px))
        catalogo.append(dict(x_km=x_km, y_km=y_km, D_km=D_km,
                             frescura=frescura, tipo="landmark", nombre=nombre))

    # ---------------------------------------------------------
    # 5b) ULTIMA LLUVIA DE CRATERES PEQUENOS
    # Se estampan DESPUES de los landmark para que tambien caigan dentro de
    # ellos. Un crater grande con el fondo perfectamente liso canta a CGI.
    # ---------------------------------------------------------
    if verbose: print("Crateres pequenos superpuestos...")
    n_peq = int(n_fondo * 0.45)
    u2 = rng.random(n_peq)
    Ds2 = (0.28 ** (-b) - u2 * (0.28 ** (-b) - 2.5 ** (-b))) ** (-1.0 / b)
    xs2 = rng.random(n_peq) * res_x
    ys2 = rng.random(n_peq) * res_y
    for i in range(n_peq):
        R2 = 0.5 * float(Ds2[i]) * px_por_km
        if R2 < 1.2:
            continue
        fr2 = float(rng.uniform(0.35, 1.0))
        estampar_crater(H, xs2[i], ys2[i], R2, float(Ds2[i]), fr2, rng)
        if Ds2[i] >= D_MIN_CATALOGO_KM:
            catalogo.append(dict(x_km=float(xs2[i]) * km_por_px,
                                 y_km=ALTO_KM - float(ys2[i]) * km_por_px,
                                 D_km=float(Ds2[i]), frescura=fr2, tipo="fondo"))

    # ---------------------------------------------------------
    # 6) RUGOSIDAD DE REGOLITO a escala fina
    # ---------------------------------------------------------
    if verbose: print("Rugosidad de regolito...")
    esc = max(1.0, ANCHO_KM / 280.0)     # mantener la escala fisica del grano
    sumar_fbm(H, rng, celdas_base=int(64 * esc), octavas=6,
              amplitud=130.0, banda=banda)
    sumar_fbm(H, rng, celdas_base=int(256 * esc), octavas=4,
              amplitud=35.0, banda=banda)

    # ---------------------------------------------------------
    # 7) ALBEDO
    # ---------------------------------------------------------
    if verbose: print("Mapa de albedo...")
    A = np.full((res_y, res_x), 0.42, np.float32)          # tierras altas
    for y0 in range(0, res_y, banda):                       # mares oscuros
        y1 = min(res_y, y0 + banda)
        A[y0:y1] -= mare[y0:y1] * 0.27
    sumar_fbm(A, rng, celdas_base=int(16 * esc), octavas=7,
              amplitud=0.10, banda=banda)
    sumar_fbm(A, rng, celdas_base=int(200 * esc), octavas=4,
              amplitud=0.05, banda=banda)

    # Rayos y halos de eyecta fresca.
    # BUG que tenia: el patron radial de rayos se aplicaba desde d=0, y como su
    # peso era maximo en el centro, cada crater salia con una estrella de radios
    # de bicicleta dibujada dentro del cuenco. Los rayos lunares son un
    # fenomeno de eyecta: empiezan FUERA del borde. Ademas eran un solo armonico
    # senoidal, demasiado regular; los rayos reales son estrechos y discontinuos.
    for (cx, cy, R_px, fr) in rayos:
        if fr < 0.90:
            continue
        alcance = R_px * rng.uniform(4.0, 9.0)
        rad = int(alcance)
        x0 = max(0, int(cx) - rad); x1 = min(res_x, int(cx) + rad + 1)
        y0 = max(0, int(cy) - rad); y1 = min(res_y, int(cy) + rad + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        wy, wx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        dx = wx - cx; dy = wy - cy
        d = np.hypot(dx, dy)
        th = np.arctan2(dy, dx)

        # halo continuo de eyecta fresca justo alrededor del borde
        halo = np.clip((R_px * 2.2 - d) / (R_px * 1.2), 0.0, 1.0) ** 2

        # rayos: suma de armonicos altos, recortada para que salgan estrechos
        ray = np.zeros_like(th)
        for k in rng.choice(np.arange(6, 26), size=3, replace=False):
            ray += np.sin(k * th + rng.uniform(0, 6.28)) / np.sqrt(k)
        ray = np.clip(ray * 1.6, 0.0, 1.0) ** 2
        caida = np.clip(1.0 - d / alcance, 0.0, 1.0) ** 1.5

        # ambos se apagan dentro del crater
        mascara_fuera = np.clip((d - R_px) / (0.35 * R_px), 0.0, 1.0)

        A[y0:y1, x0:x1] += 0.13 * fr * mascara_fuera * (0.55 * halo + 0.75 * ray * caida)

    A = np.clip(A, 0.05, 1.0)

    # cota del terreno en cada crater (util para proyectar las etiquetas YOLO
    # con precision: un crater no esta a z=0, esta sobre el relieve)
    for c in catalogo:
        cxp = int(np.clip(c["x_km"] * px_por_km, 0, res_x - 1))
        cyp = int(np.clip((ALTO_KM - c["y_km"]) * px_por_km, 0, res_y - 1))
        c["z_km"] = float(H[cyp, cxp]) / 1000.0

    if verbose:
        print(f"Catalogo: {len(catalogo)} crateres estampados")
    return H, A, mare, gt, km_por_px, catalogo


# =========================================================
# HILLSHADE para vista previa (simula el sol rasante)
# =========================================================

def hillshade(H, km_por_px, azimut_deg=315.0, elevacion_deg=22.0,
              exageracion=1.0):
    dx_m = km_por_px * 1000.0
    gy, gx = np.gradient(H.astype(np.float32) * exageracion, dx_m)

    az = np.deg2rad(360.0 - azimut_deg + 90.0)
    el = np.deg2rad(elevacion_deg)

    pendiente = np.arctan(np.hypot(gx, gy))
    aspecto = np.arctan2(gy, -gx)

    sh = (np.sin(el) * np.cos(pendiente) +
          np.cos(el) * np.sin(pendiente) * np.cos(az - aspecto))
    sh = np.clip(sh, 0.0, 1.0)
    # luz ambiente: en orbita real la sombra recibe algo de luz rebotada,
    # nunca es negro absoluto
    return 0.06 + 0.94 * sh


if __name__ == "__main__":
    import os
    import sys
    from PIL import Image

    res_x = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
    res_y = int(res_x * ALTO_KM / ANCHO_KM)

    H, A, mare, gt, kmpx, cat = generar(res_x, res_y)

    print(f"Altura: min {H.min():.0f} m  max {H.max():.0f} m  "
          f"rango {H.max()-H.min():.0f} m")

    sh = hillshade(H, kmpx, elevacion_deg=28.0)
    img = sh * A
    lo, hi = np.percentile(img, [0.5, 99.6])
    img = np.clip((img - lo) / (hi - lo), 0, 1) ** (1 / 1.6)

    salida = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "preview.png")
    Image.fromarray((img * 255).astype(np.uint8)).save(salida)
    print(f"{salida} guardado")
