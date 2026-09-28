"""
Dibuja la trayectoria de la camara como un tubo sobre el terreno, coloca una
camara de tres cuartos que encuadra la escena entera y renderiza
fig_trayectoria3d.png.

Crea una camara nueva, CamaraFigura, y la deja activa solo durante el render:
no toca CamaraNave ni su animacion. Se ejecuta con terreno_lunar.blend
abierto.
"""

import os
import csv
import math

import bpy
from mathutils import Vector

# --- Rutas relativas al propio fichero ------------------------------------
# El repositorio se puede clonar en cualquier carpeta, asi que ninguna ruta
# esta escrita a mano. __file__ no existe cuando el codigo se pega dentro de
# un editor (el de Blender, por ejemplo); en ese caso se usa el directorio de
# trabajo.
_AQUI = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.path.abspath(os.getcwd()))

RUTA = _AQUI          # carpeta blender/ del repositorio
CSV_TRAY = os.path.join(RUTA, "trayectoria_real.csv")
SALIDA = os.path.join(RUTA, "fig_trayectoria3d.png")

ANCHO_KM, ALTO_KM = 840.0, 630.0
RADIO_TUBO = 2.5          # km
MUESTREO = 4              # 1 de cada N fotogramas
RES_X, RES_Y = 2000, 1300
MUESTRAS = 96


def leer():
    pts = []
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i % MUESTREO:
                continue
            # del sistema del mapa al del mundo: el terreno esta centrado
            pts.append((float(r["x_km"]) - ANCHO_KM / 2.0,
                        float(r["y_km"]) - ALTO_KM / 2.0,
                        float(r["altura_km"])))
    return pts


def borrar(nombre):
    o = bpy.data.objects.get(nombre)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)


def curva(pts):
    borrar("Trayectoria")
    cu = bpy.data.curves.new("TrayectoriaCurva", type="CURVE")
    cu.dimensions = "3D"
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, (x, y, z) in enumerate(pts):
        sp.points[i].co = (x, y, z, 1.0)
    cu.bevel_depth = RADIO_TUBO
    cu.bevel_resolution = 4
    ob = bpy.data.objects.new("Trayectoria", cu)
    bpy.context.collection.objects.link(ob)

    mat = bpy.data.materials.new("MatTrayectoria")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0.85, 0.12, 0.12, 1.0)
    em.inputs["Strength"].default_value = 3.0
    nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    ob.data.materials.append(mat)
    return ob


def marcas(pts):
    """Esferas al principio y al final del recorrido."""
    for nombre, p, col in (("IniTray", pts[0], (0.1, 0.5, 1.0)),
                           ("FinTray", pts[-1], (1.0, 0.75, 0.0))):
        borrar(nombre)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=RADIO_TUBO * 3.0, location=p)
        ob = bpy.context.active_object
        ob.name = nombre
        m = bpy.data.materials.new("Mat" + nombre)
        m.use_nodes = True
        nt = m.node_tree
        nt.nodes.clear()
        o = nt.nodes.new("ShaderNodeOutputMaterial")
        e = nt.nodes.new("ShaderNodeEmission")
        e.inputs["Color"].default_value = (*col, 1.0)
        e.inputs["Strength"].default_value = 4.0
        nt.links.new(e.outputs["Emission"], o.inputs["Surface"])
        ob.data.materials.append(m)


def camara_figura():
    borrar("CamaraFigura")
    bpy.ops.object.camera_add(location=(-720, -700, 620))
    cam = bpy.context.active_object
    cam.name = "CamaraFigura"
    cam.data.lens = 42.0
    cam.data.clip_start = 1.0
    cam.data.clip_end = 5000.0
    # apuntar al centro del terreno
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    return cam


def main():
    pts = leer()
    print(f"Trayectoria: {len(pts)} puntos")
    curva(pts)
    marcas(pts)
    cam = camara_figura()

    sc = bpy.context.scene
    cam_previa = sc.camera
    sc.camera = cam
    rx, ry = sc.render.resolution_x, sc.render.resolution_y
    fp = sc.render.filepath
    modo = sc.render.image_settings.color_mode
    try:
        muestras_previas = sc.cycles.samples
    except Exception:
        muestras_previas = None

    sc.render.resolution_x, sc.render.resolution_y = RES_X, RES_Y
    sc.render.image_settings.color_mode = "RGB"
    sc.render.filepath = SALIDA
    if muestras_previas is not None:
        sc.cycles.samples = MUESTRAS

    bpy.ops.render.render(write_still=True)
    print(f"Guardada: {SALIDA}")

    # dejar la escena como estaba
    sc.camera = cam_previa
    sc.render.resolution_x, sc.render.resolution_y = rx, ry
    sc.render.image_settings.color_mode = modo
    sc.render.filepath = fp
    if muestras_previas is not None:
        sc.cycles.samples = muestras_previas


if __name__ == "__main__":
    main()
