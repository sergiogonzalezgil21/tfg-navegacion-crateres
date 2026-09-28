"""
Genera las etiquetas YOLO de cada frame del vuelo de crucero y la trayectoria
verdadera, a partir de la escena de Blender.

Blender sabe donde esta cada crater y donde esta la camara en cada frame, asi
que las etiquetas salen de proyectar el catalogo por la camara: verdad de
referencia exacta, sin etiquetar nada a mano, y sin la que no se podria poner
un numero al error de navegacion.

Produce  labels/0001.txt ...     etiquetas YOLO de cada frame
         trayectoria_real.csv    posicion, altura y orientacion verdaderas
         mapa_yolo.txt           el mapa de referencia en formato YOLO
         mapa_referencia.csv     el mismo mapa, en km, para MATLAB

Se ejecuta con el .blend abierto y la camara ya animada; hay que ajustar
FRAME_INI y FRAME_FIN.
"""

import os
import csv
import math

import bpy
import numpy as np
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

# =============================================================================
# CONFIGURACION
# =============================================================================

# --- Rutas relativas al propio fichero ------------------------------------
# El repositorio se puede clonar en cualquier carpeta, asi que ninguna ruta
# esta escrita a mano. __file__ no existe cuando el codigo se pega dentro de
# un editor (el de Blender, por ejemplo); en ese caso se usa el directorio de
# trabajo.
_AQUI = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.path.abspath(os.getcwd()))

RUTA_PROYECTO = _AQUI          # carpeta blender/ del repositorio
NOMBRE_CAMARA = "CamaraNave"

FRAME_INI = 1
FRAME_FIN = 700

ANCHO_KM = 840.0
ALTO_KM = 630.0

# Un crater con la caja de menos de esto no es aprendible y solo mete ruido
MIN_LADO_PX = 12

# Los crateres muy degradados apenas se ven: etiquetarlos ensena a la red a
# "detectar" cosas que no estan
MIN_FRESCURA = 0.30

# Margen: descarta crateres cuyo centro cae fuera del encuadre
MARGEN = 0.02

# Puntos del borde que se proyectan por crater para sacar la caja
N_PUNTOS_BORDE = 12

# Diametro minimo para entrar en el MAPA de referencia.
#
# Antes el mapa lo formaban solo los 180 crateres marcados como "landmark".
# Eso era inconsistente: en el terreno hay ademas 30 crateres de fondo de mas
# de 16 km que el detector va a ver igual, y que no tenian entrada en el mapa.
# El 14 % de las detecciones grandes no habria tenido con que emparejarse, y
# esas detecciones sin pareja son las que generan tripletas espurias.
#
# El mapa se construye por TAMANO, no por etiqueta. Con 16 km salen 211
# crateres y C(211,3) = 1.5 M tripletas, del mismo orden que las 954 000 que
# habia con 180, asi que al MATLAB no le cuesta mas.
D_MIN_MAPA_KM = 16.0

# =============================================================================


def cargar_catalogo():
    """Carga el catalogo a arrays de numpy, en coordenadas de mundo."""
    path = os.path.join(RUTA_PROYECTO, "crateres_catalogo.csv")
    xs, ys, zs, rs = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if float(row["frescura"]) < MIN_FRESCURA:
                continue
            xs.append(float(row["x_km"]) - ANCHO_KM / 2.0)
            ys.append(float(row["y_km"]) - ALTO_KM / 2.0)
            zs.append(float(row["z_km"]))
            rs.append(float(row["D_km"]) / 2.0)

    X = np.array(xs, np.float64)
    Y = np.array(ys, np.float64)
    Z = np.array(zs, np.float64)
    R = np.array(rs, np.float64)
    print(f"Catalogo: {len(X)} crateres con frescura >= {MIN_FRESCURA}")
    return X, Y, Z, R


def puntos_borde(X, Y, Z, R, n=N_PUNTOS_BORDE):
    """Devuelve (N, n, 3): n puntos repartidos por el borde de cada crater."""
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    P = np.empty((len(X), n, 3), np.float64)
    P[:, :, 0] = X[:, None] + R[:, None] * ca[None, :]
    P[:, :, 1] = Y[:, None] + R[:, None] * sa[None, :]
    P[:, :, 2] = Z[:, None]
    return P


def proyectar(P, M_inv, tan_x, tan_y):
    """Proyecta puntos de mundo (N, n, 3) a coordenadas normalizadas.

    Se hace a mano y vectorizado en vez de con world_to_camera_view porque esa
    funcion trabaja punto a punto: con 40 000 crateres, 12 puntos cada uno y
    700 frames serian 340 millones de llamadas a Python. Asi son unas pocas
    operaciones de numpy por frame.

    Devuelve (u, v_desde_arriba, delante) con u,v en [0,1] dentro del encuadre.
    """
    forma = P.shape[:-1]
    Q = P.reshape(-1, 3)

    # a coordenadas de camara (la camara mira hacia -Z en su propio sistema)
    cx = Q @ M_inv[0, :3] + M_inv[0, 3]
    cy = Q @ M_inv[1, :3] + M_inv[1, 3]
    cz = Q @ M_inv[2, :3] + M_inv[2, 3]

    delante = cz < 0.0
    prof = np.where(delante, -cz, 1.0)

    u = 0.5 + 0.5 * (cx / prof) / tan_x
    v = 0.5 - 0.5 * (cy / prof) / tan_y     # YOLO mide Y desde arriba

    return (u.reshape(forma), v.reshape(forma), delante.reshape(forma))


def comprobar_proyeccion(scene, cam, M_inv, tan_x, tan_y, X, Y, Z):
    """Contrasta la proyeccion vectorizada con la de Blender en unos puntos.

    Merece la pena: si la matriz o el FOV estuvieran mal, las etiquetas
    saldrian desplazadas de forma sistematica y el entrenamiento aprenderia
    ese sesgo sin que nada diera error.
    """
    n = min(20, len(X))
    if n == 0:
        return
    idx = np.linspace(0, len(X) - 1, n).astype(int)
    P = np.stack([X[idx], Y[idx], Z[idx]], axis=-1)[:, None, :]
    u, v, _ = proyectar(P, M_inv, tan_x, tan_y)

    peor = 0.0
    for k, i in enumerate(idx):
        co = world_to_camera_view(scene, cam, Vector((X[i], Y[i], Z[i])))
        peor = max(peor, abs(co.x - u[k, 0]), abs((1.0 - co.y) - v[k, 0]))
    print(f"  Contraste con world_to_camera_view: desviacion max {peor:.2e}")
    if peor > 1e-4:
        print("     AVISO: las dos proyecciones no coinciden. Algo va mal.")


def main():
    scene = bpy.context.scene
    cam = bpy.data.objects.get(NOMBRE_CAMARA) or scene.camera
    if cam is None:
        raise RuntimeError("No encuentro la camara. Revisa NOMBRE_CAMARA.")

    W = scene.render.resolution_x
    H = scene.render.resolution_y
    tan_x = math.tan(cam.data.angle_x / 2.0)
    tan_y = tan_x * (H / W)
    print(f"Camara: {cam.name}   render {W}x{H}   "
          f"FOV_h {math.degrees(cam.data.angle_x):.1f} grados")

    X, Y, Z, R = cargar_catalogo()
    if len(X) == 0:
        raise RuntimeError("El catalogo esta vacio.")

    dir_labels = os.path.join(RUTA_PROYECTO, "labels")
    os.makedirs(dir_labels, exist_ok=True)

    filas_tray = []
    total_cajas = 0
    frames_pobres = 0
    frame_guardado = scene.frame_current
    primera = True

    for f in range(FRAME_INI, FRAME_FIN + 1):
        scene.frame_set(f)
        bpy.context.view_layer.update()

        mw = cam.matrix_world
        pos = mw.translation
        rot = mw.to_euler('XYZ')
        M_inv = np.array(mw.inverted(), np.float64)

        # ---- poda espacial ----
        # Sin esto habria que proyectar los 40 000 crateres en cada frame.
        # Solo pueden caer en el encuadre los que estan cerca del punto que la
        # camara tiene debajo; el 1.8 es margen de sobra para las esquinas.
        alcance = abs(pos.z) * tan_x * 1.8
        cerca = ((np.abs(X - pos.x) < alcance + R) &
                 (np.abs(Y - pos.y) < alcance + R))
        idx = np.flatnonzero(cerca)

        lineas = []
        if len(idx):
            P = puntos_borde(X[idx], Y[idx], Z[idx], R[idx])
            u, v, delante = proyectar(P, M_inv, tan_x, tan_y)

            if primera:
                comprobar_proyeccion(scene, cam, M_inv, tan_x, tan_y,
                                     X[idx], Y[idx], Z[idx])
                primera = False

            valido = delante.all(axis=1)
            u0 = u.min(axis=1); u1 = u.max(axis=1)
            v0 = v.min(axis=1); v1 = v.max(axis=1)

            cxs = 0.5 * (u0 + u1)
            cys = 0.5 * (v0 + v1)
            bw = u1 - u0
            bh = v1 - v0

            ok = (valido
                  & (cxs >= -MARGEN) & (cxs <= 1 + MARGEN)
                  & (cys >= -MARGEN) & (cys <= 1 + MARGEN)
                  & (bw * W >= MIN_LADO_PX) & (bh * H >= MIN_LADO_PX))

            # recorte al encuadre
            rx0 = np.clip(cxs - bw / 2, 0.0, 1.0)
            rx1 = np.clip(cxs + bw / 2, 0.0, 1.0)
            ry0 = np.clip(cys - bh / 2, 0.0, 1.0)
            ry1 = np.clip(cys + bh / 2, 0.0, 1.0)
            bw2 = rx1 - rx0
            bh2 = ry1 - ry0

            # si al recortar queda menos de un tercio del crater, no sirve
            area = np.maximum(bw * bh, 1e-12)
            ok &= (bw2 > 0) & (bh2 > 0) & ((bw2 * bh2) >= 0.33 * area)

            for k in np.flatnonzero(ok):
                lineas.append(
                    f"0 {0.5*(rx0[k]+rx1[k]):.6f} {0.5*(ry0[k]+ry1[k]):.6f} "
                    f"{bw2[k]:.6f} {bh2[k]:.6f}")

        with open(os.path.join(dir_labels, f"{f:04d}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("\n".join(lineas))

        total_cajas += len(lineas)
        if len(lineas) < 3:
            frames_pobres += 1

        filas_tray.append([
            f,
            f"{pos.x + ANCHO_KM/2.0:.4f}",     # a coordenadas del mapa
            f"{pos.y + ALTO_KM/2.0:.4f}",
            f"{pos.z:.4f}",                    # altura en km
            f"{math.degrees(rot.x):.4f}",
            f"{math.degrees(rot.y):.4f}",
            f"{math.degrees(rot.z):.4f}",
            len(lineas),
        ])

        if f % 100 == 0 or f == FRAME_INI:
            print(f"  frame {f:4d}: {len(lineas):3d} crateres, "
                  f"altura {pos.z:6.2f} km  (candidatos {len(idx)})")

    scene.frame_set(frame_guardado)

    with open(os.path.join(RUTA_PROYECTO, "trayectoria_real.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "x_km", "y_km", "altura_km",
                    "rot_x_deg", "rot_y_deg", "rot_z_deg", "n_crateres"])
        w.writerows(filas_tray)

    # ---- el mapa de referencia, en YOLO, exacto ----
    # El mapa de referencia sale directamente de la verdad del generador, no
    # de detecciones de la red sobre mapa.png: asi no se propaga el error del
    # detector al mapa embarcado.
    n_map = 0
    with open(os.path.join(RUTA_PROYECTO, "mapa_yolo.txt"), "w",
              encoding="utf-8") as fh, \
         open(os.path.join(RUTA_PROYECTO, "mapa_referencia.csv"), "w",
              newline="", encoding="utf-8") as fc:

        wc = csv.writer(fc)
        wc.writerow(["id", "nombre", "tipo", "x_km", "y_km", "D_km"])

        path = os.path.join(RUTA_PROYECTO, "crateres_catalogo.csv")
        with open(path, newline="", encoding="utf-8") as f2:
            for row in csv.DictReader(f2):
                D = float(row["D_km"])
                if D < D_MIN_MAPA_KM or float(row["frescura"]) < MIN_FRESCURA:
                    continue
                x, y = float(row["x_km"]), float(row["y_km"])
                xn = x / ANCHO_KM
                yn = 1.0 - y / ALTO_KM
                wn = D / ANCHO_KM
                hn = D / ALTO_KM
                fh.write(f"0 {xn:.6f} {yn:.6f} {wn:.6f} {hn:.6f}\n")
                n_map += 1
                wc.writerow([n_map, row.get("nombre", ""), row["tipo"],
                             f"{x:.4f}", f"{y:.4f}", f"{D:.4f}"])

    n = FRAME_FIN - FRAME_INI + 1
    print("=" * 70)
    print(f"Frames procesados: {n}")
    print(f"Cajas totales: {total_cajas}  ->  media {total_cajas/n:.1f} por frame")
    print(f"Frames con menos de 3 crateres: {frames_pobres} "
          f"({100*frames_pobres/n:.1f} %)")
    print(f"labels/  ->  {dir_labels}")
    print(f"trayectoria_real.csv  ->  ground truth para validar el MATLAB")
    trip = n_map * (n_map - 1) * (n_map - 2) // 6
    print(f"mapa_yolo.txt  ->  {n_map} crateres de D >= {D_MIN_MAPA_KM} km")
    print(f"mapa_referencia.csv  ->  los mismos, en km, para MATLAB")
    print(f"   tripletas del mapa: {trip:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
