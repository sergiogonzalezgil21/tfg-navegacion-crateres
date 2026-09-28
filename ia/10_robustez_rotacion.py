"""
Paso 10. Aguanta el detector que la imagen venga girada?

En la fase 4 la camara gira alrededor de su eje optico. El navegador ya se
comprobo exactamente invariante a rotacion, asi que lo que se pone a prueba es
el detector. Y hay motivo para sospechar: el sol esta fijo en el mundo, de
modo que al girar la camara la direccion de las sombras gira dentro de la
imagen, y la red se entreno con degrees = 0.0.

No hace falta re-renderizar. Para una camara nadir sobre un plano, girarla
alrededor de su eje optico da exactamente la imagen original rotada respecto
a su centro. Basta girar los frames que ya hay. La unica diferencia esta en
los bordes, asi que todo se mide dentro del circulo inscrito.

Mide el recall sobre los crateres de escala de mapa en funcion del angulo de
giro. Conviene lanzarlo antes que el render de la fase 4, para no pelear por
la GPU.
"""

import os
import csv
import glob
import math

import cv2
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

DIR_IMG = os.path.join(RUTA_IA, "dataset_blender", "val", "images")
DIR_LAB = os.path.join(RUTA_IA, "dataset_blender", "val", "labels")
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")

MODELOS = {
    "antes  (degrees=0)":   os.path.join(RUTA_IA, "runs", "finetune_blender",
                                         "weights", "best.pt"),
    "despues (degrees=180)": os.path.join(RUTA_IA, "runs", "finetune_rotacion",
                                          "weights", "best.pt"),
}

W, H = 1920.0, 1080.0
FOV_HORIZONTAL = 60.0
D_MIN_MAPA_KM = 10.0        # el umbral del mapa embarcado
CONF = 0.70                 # punto de operacion de las fases 3A y 3B
IMGSZ = 1024
MAX_DET = 3000
TOL_NORM = 0.03

ANGULOS = [0, 15, 30, 45, 60, 90, 135, 180]
PASO_FRAMES = 3             # 1 de cada 3 frames de validacion, sobra

# =============================================================================


def alturas():
    d = {}
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d[int(r["frame"])] = float(r["altura_km"])
    return d


def etiquetas(base, alt):
    """Crateres de escala de mapa del frame, en pixeles: (x, y, D_km)."""
    p = os.path.join(DIR_LAB, base + ".txt")
    if not os.path.isfile(p):
        return []
    fac = 2.0 * alt * math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    out = []
    with open(p, encoding="utf-8") as f:
        for l in f:
            v = l.split()
            if len(v) < 5:
                continue
            D = float(v[3]) * fac
            if D >= D_MIN_MAPA_KM:
                out.append((float(v[1]) * W, float(v[2]) * H, D))
    return out


CX, CY = W / 2.0, H / 2.0
R_UTIL = min(W, H) / 2.0 - 40.0     # circulo inscrito, con un margen


def girar_puntos(P, ang_deg):
    """Gira puntos igual que lo hace cv2.warpAffine con la imagen.

    OJO CON EL SIGNO. La rotacion "de libro" es
        x' = cos*dx - sin*dy
        y' = sin*dx + cos*dy
    pero eso vale en ejes con la Y hacia ARRIBA. En coordenadas de imagen la
    Y va hacia abajo, y cv2.getRotationMatrix2D usa:
        x' =  cos*dx + sin*dy
        y' = -sin*dx + cos*dy

    Con la version de libro las etiquetas giran en sentido CONTRARIO a la
    imagen. El fallo no se nota a 0 ni a 180 grados, porque el seno vale
    cero, y a cualquier otro angulo no empareja absolutamente nada. Es
    exactamente el patron que aparecio en la primera medida.
    """
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for (x, y, D) in P:
        dx, dy = x - CX, y - CY
        out.append((CX + ca * dx + sa * dy, CY - sa * dx + ca * dy, D))
    return out


def dentro(x, y):
    return math.hypot(x - CX, y - CY) <= R_UTIL


def emparejar(det, gt):
    usadas = set()
    ok = 0
    for (gx, gy, _d) in gt:
        mejor, md = None, 1e9
        for i, (dx, dy, _dd) in enumerate(det):
            if i in usadas:
                continue
            d = math.hypot(dx - gx, dy - gy)
            if d < md:
                md, mejor = d, i
        if mejor is not None and md < TOL_NORM * W:
            usadas.add(mejor)
            ok += 1
    return ok


def evaluar(nombre, pesos, alt, imgs):
    from ultralytics import YOLO
    if not os.path.isfile(pesos):
        print(f"  [!] no existen los pesos: {pesos}")
        return None

    model = YOLO(pesos)
    print(f"\n{nombre}")
    print(f"  {'giro':>6} {'reales/frm':>11} {'detect/frm':>11} {'recall':>8} "
          f"{'>=3':>7} {'>=4':>7} {'ancho rel':>10}")
    print("  " + "-" * 68)

    filas = []
    for ang in ANGULOS:
        n_gt = n_det = n_ok = 0
        c3, c4, anchos = [], [], []

        for p in imgs:
            base = os.path.splitext(os.path.basename(p))[0]
            f = int(base)
            if f not in alt:
                continue

            gt = etiquetas(base, alt[f])
            gt = [g for g in girar_puntos(gt, ang) if dentro(g[0], g[1])]

            img = cv2.imread(p)
            if ang != 0:
                M = cv2.getRotationMatrix2D((CX, CY), ang, 1.0)
                img = cv2.warpAffine(img, M, (int(W), int(H)),
                                     flags=cv2.INTER_LINEAR,
                                     borderValue=(0, 0, 0))

            r = model.predict(img, imgsz=IMGSZ, conf=CONF,
                              max_det=MAX_DET, verbose=False)[0]

            det = []
            if r.boxes is not None and len(r.boxes):
                xywh = r.boxes.xywh.cpu().numpy()
                fac = 2.0 * alt[f] * math.tan(math.radians(FOV_HORIZONTAL / 2))
                for (cx, cy, bw, _bh) in xywh:
                    if bw / W * fac >= D_MIN_MAPA_KM and dentro(cx, cy):
                        det.append((float(cx), float(cy), float(bw)))

            # sesgo de tamano: ancho detectado frente al ancho real del crater
            # emparejado. Con degrees=180 YOLO agranda las cajas al girarlas,
            # asi que hay que comprobar si el modelo nuevo las infla.
            for (dx, dy, dw) in det:
                mejor, md = None, 1e9
                for (gx, gy, gD) in gt:
                    d = math.hypot(dx - gx, dy - gy)
                    if d < md:
                        md, mejor = d, gD
                if mejor is not None and md < TOL_NORM * W:
                    fac = 2.0 * alt[f] * math.tan(math.radians(FOV_HORIZONTAL/2))
                    anchos.append((dw / W * fac) / mejor)

            ok = emparejar(det, gt)
            n_gt += len(gt); n_det += len(det); n_ok += ok
            c3.append(ok >= 3); c4.append(ok >= 4)

        nf = max(len(c3), 1)
        rec = n_ok / max(n_gt, 1)
        anc = float(np.median(anchos)) if anchos else float("nan")
        filas.append(dict(modelo=nombre, giro=ang, reales_frame=n_gt/nf,
                          det_frame=n_det/nf, recall=rec,
                          pct3=100*np.mean(c3), pct4=100*np.mean(c4),
                          ancho_rel=anc))
        print(f"  {ang:>6} {n_gt/nf:>11.2f} {n_det/nf:>11.2f} {rec:>8.3f} "
              f"{100*np.mean(c3):>6.1f}% {100*np.mean(c4):>6.1f}% {anc:>10.3f}")
    return filas


def main():
    alt = alturas()
    imgs = sorted(glob.glob(os.path.join(DIR_IMG, "*.png")))[::PASO_FRAMES]
    if not imgs:
        print(f"[!] no hay imagenes en {DIR_IMG}")
        return

    print("=" * 80)
    print(f"ROBUSTEZ A LA ROTACION   {len(imgs)} frames   conf={CONF}   "
          f"D_mapa >= {D_MIN_MAPA_KM:.0f} km")
    print(f"  se mide dentro del circulo inscrito (r = {R_UTIL:.0f} px)")
    print("  'ancho rel' = ancho de caja detectado / diametro real. Deberia ser 1.")
    print("=" * 80)

    todas = []
    for nombre, pesos in MODELOS.items():
        f = evaluar(nombre, pesos, alt, imgs)
        if f:
            todas.extend(f)

    if not todas:
        return

    out = os.path.join(RUTA_IA, "robustez_rotacion.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(todas[0].keys()))
        w.writeheader(); w.writerows(todas)
    print(f"\nGuardado en {out}")

    # comparativa directa
    nombres = list(dict.fromkeys(r["modelo"] for r in todas))
    if len(nombres) == 2:
        a, b = nombres
        print()
        print(f"{'giro':>6} {'recall antes':>14} {'recall despues':>16} {'cambio':>10}")
        print("-" * 50)
        for ang in ANGULOS:
            ra = next(r["recall"] for r in todas if r["modelo"] == a and r["giro"] == ang)
            rb = next(r["recall"] for r in todas if r["modelo"] == b and r["giro"] == ang)
            flecha = "  " if abs(rb-ra) < 0.02 else (" +" if rb > ra else " -")
            print(f"{ang:>6} {ra:>14.3f} {rb:>16.3f} {flecha}{abs(rb-ra):>8.3f}")


if __name__ == "__main__":
    main()
