"""
Monta la escena completa en Blender: genera el heightmap y el mapa de albedo
llamando a terreno_lunar.py, los guarda como PNG de 16 bits, crea el plano de
840 x 630 km con el relieve metido en la Z de la malla, el material, el sol
rasante y la camara, y exporta el catalogo de crateres a CSV.

Necesita terreno_lunar.py en la misma carpeta. Tarda unos minutos, casi todos
en generar el heightmap.

1 unidad de Blender = 1 km. En metros, un terreno de 840 km con la camara a
120 km mete numeros de seis cifras en el z-buffer y aparecen artefactos de
precision.
"""

import sys
import os
import csv
import math
import zlib
import struct

import bpy
import numpy as np
from mathutils import Vector

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

# ---- TAMANO DEL TERRENO ----
TERRENO_ANCHO_KM = 840.0
TERRENO_ALTO_KM = 630.0

RES_HEIGHTMAP = 8192          # 8192 / 840 km = 102 m/pixel
SEED = 20260818               # semilla del generador; otro valor da otro terreno
N_CRATERES_FONDO = 400000     # 0.75 crateres/km2, la densidad de siempre

# Crateres de referencia repartidos por todo el terreno
N_LANDMARKS = 180
LANDMARK_SEP_MIN_KM = 34.0

# Malla del plano. 3000x2250 = 6.75 M de vertices, unos 280 m por vertice.
# El detalle mas fino que no cabe en la malla entra por bump map.
# Con 2000x1500 el visor va mas suelto: pasa a 420 m por vertice y apenas
# se nota, porque el relieve fino lo pone el bump de todos modos.
SUBDIV_X = 3000
SUBDIV_Y = 2250

# Las texturas se guardan reducidas respecto al heightmap. Blender mantiene
# cada textura en memoria como float RGBA: a 8192x6144 son 805 MB por textura.
# El albedo es suave y no pierde nada al reducirlo; el detalle del bump si
# necesita resolucion, asi que se queda como esta.
REDUCIR_ALBEDO = 2            # 8192 -> 4096
REDUCIR_ALTURA = 2            # altura.png es solo documentacion

# Iluminacion. Un sol RASANTE es lo que hace legibles los crateres: es la
# condicion en la que estan tomadas casi todas las imagenes orbitales reales
# con las que se entreno la red.
SOL_ELEVACION = 28.0          # grados sobre el horizonte
SOL_AZIMUT = 315.0            # grados
SOL_FUERZA = 6.0

# Camara
ALTURA_CAMARA_KM = 120.0
FOV_HORIZONTAL = 60.0
RENDER_W, RENDER_H = 1920, 1080

LIMPIAR_ESCENA = True

# =============================================================================

sys.path.append(RUTA_PROYECTO)
import terreno_lunar as TL                                    # noqa: E402
import importlib                                              # noqa: E402
importlib.reload(TL)

# la configuracion de aqui manda sobre la del modulo
TL.ANCHO_KM = TERRENO_ANCHO_KM
TL.ALTO_KM = TERRENO_ALTO_KM
TL.N_LANDMARKS = N_LANDMARKS
TL.LANDMARK_SEP_MIN_KM = LANDMARK_SEP_MIN_KM

ANCHO_KM = TL.ANCHO_KM
ALTO_KM = TL.ALTO_KM

RES_X = RES_HEIGHTMAP
RES_Y = int(round(RES_HEIGHTMAP * ALTO_KM / ANCHO_KM))


def ruta(nombre):
    return os.path.join(RUTA_PROYECTO, nombre)


# =============================================================================
# 1) GENERAR LOS MAPAS
# =============================================================================

def generar_mapas():
    print("=" * 70)
    print("Generando terreno. Esto tarda unos minutos.")
    H, A, mare, gt, km_por_px, catalogo = TL.generar(
        RES_X, RES_Y, seed=SEED, n_fondo=N_CRATERES_FONDO)

    h_min, h_max = float(H.min()), float(H.max())
    print(f"Relieve: {h_min:.0f} a {h_max:.0f} m  (rango {h_max - h_min:.0f} m)")

    # ---- altura.png : 16 bits, normalizado ----
    # Ya no se usa para nada del render (la geometria sale del array
    # directamente y el bump de detalle.png). Se guarda reducido como
    # documentacion, por si hace falta para la memoria o para MATLAB.
    Hn = (H - h_min) / (h_max - h_min)
    guardar_png16(reducir(Hn, REDUCIR_ALTURA), ruta("altura.png"))

    # ---- detalle.png : residuo de alta frecuencia para el bump ----
    # La malla solo puede representar hasta su propia resolucion. Todo lo que
    # queda por debajo (el grano del regolito, los crateres de menos de 1 km)
    # se mete como bump para que se vea sin cargar 12 M de vertices.
    fy = max(1, RES_Y // SUBDIV_Y)
    fx = max(1, RES_X // SUBDIV_X)
    bajo = suavizar_bloques(H, fy, fx)     # lo que SI puede representar la malla
    detalle = H - bajo
    d_max = float(np.abs(detalle).max()) + 1e-6
    guardar_png16(detalle / (2 * d_max) + 0.5, ruta("detalle.png"))
    print(f"Detalle de alta frecuencia: +-{d_max:.0f} m")

    # ---- albedo.png : 8 bits ----
    guardar_png8(reducir(np.clip(A, 0, 1), REDUCIR_ALBEDO), ruta("albedo.png"))

    # ---- catalogo de crateres ----
    with open(ruta("crateres_catalogo.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nombre", "tipo", "x_km", "y_km", "z_km", "D_km", "frescura"])
        for c in sorted(catalogo, key=lambda c: -c["D_km"]):
            w.writerow([c.get("nombre", ""), c["tipo"],
                        f"{c['x_km']:.4f}", f"{c['y_km']:.4f}",
                        f"{c['z_km']:.4f}", f"{c['D_km']:.4f}",
                        f"{c['frescura']:.3f}"])
    print(f"Catalogo: {len(catalogo)} crateres -> crateres_catalogo.csv")

    # ---- ground truth de los 18 landmark, en km ----
    with open(ruta("mapa_landmarks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nombre", "x_km", "y_km", "D_km"])
        for g in gt:
            w.writerow([g["nombre"], f"{g['x_km']:.3f}",
                        f"{g['y_km']:.3f}", f"{g['D_km']:.3f}"])
    print("Landmarks -> mapa_landmarks.csv")

    return H, bajo, h_min, h_max, d_max


def reducir(arr, f):
    """Reduce por promedio de bloques f x f. Promediar y no decimar evita
    aliasing: con decimacion el ruido fino se convierte en moire."""
    if f <= 1:
        return arr
    ny, nx = arr.shape
    ny2, nx2 = (ny // f) * f, (nx // f) * f
    return arr[:ny2, :nx2].reshape(ny2 // f, f, nx2 // f, f).mean(axis=(1, 3))


def suavizar_bloques(H, fy, fx):
    """Promedio por bloques y vuelta al tamano original. Emula lo que la malla
    puede representar."""
    ny, nx = H.shape
    ny2, nx2 = (ny // fy) * fy, (nx // fx) * fx
    rec = H[:ny2, :nx2].reshape(ny2 // fy, fy, nx2 // fx, fx).mean(axis=(1, 3))
    out = np.repeat(np.repeat(rec, fy, axis=0), fx, axis=1)
    full = np.empty_like(H)
    full[:ny2, :nx2] = out
    if ny2 < ny:
        full[ny2:, :nx2] = out[-1:, :]
    if nx2 < nx:
        full[:, nx2:] = full[:, nx2 - 1:nx2]
    return full


def _escribir_png_gris(path, arr01, bits=16):
    """Escribe un PNG en escala de grises (8 o 16 bits) a pelo.

    Por que no se usa el guardado de imagenes de Blender: la via
    images.new() + pixels.foreach_set() + save_render() escribia archivos EN
    BLANCO sin dar ningun error. Se notaba en que altura.png y detalle.png
    pesaban los dos exactamente 110 KB (un PNG uniforme comprime a casi nada)
    cuando 4096x3072 a 16 bits deberia rondar los 10-20 MB. Ese era el motivo
    real de que el terreno saliera liso. Un PNG es lo bastante simple como
    para escribirlo a mano y quitarse la dependencia.

    Fila 0 del array = fila 0 del PNG = borde SUPERIOR de la imagen, que es
    justo el convenio que ya usa el heightmap (y = +ALTO_KM/2).
    """
    a = np.clip(np.asarray(arr01, np.float64), 0.0, 1.0)
    ny, nx = a.shape

    if bits == 16:
        datos = (a * 65535.0 + 0.5).astype('>u2')
        prof = 16
        ancho_fila = nx * 2
    else:
        datos = (a * 255.0 + 0.5).astype(np.uint8)
        prof = 8
        ancho_fila = nx

    filas = datos.tobytes()
    cruda = bytearray()
    for y in range(ny):
        cruda.append(0)                      # byte de filtro: 0 = sin filtro
        cruda += filas[y * ancho_fila:(y + 1) * ancho_fila]

    def trozo(tipo, contenido):
        c = tipo + contenido
        return (struct.pack('>I', len(contenido)) + c +
                struct.pack('>I', zlib.crc32(c) & 0xFFFFFFFF))

    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(trozo(b'IHDR', struct.pack('>IIBBBBB', nx, ny, prof, 0, 0, 0, 0)))
        f.write(trozo(b'IDAT', zlib.compress(bytes(cruda), 6)))
        f.write(trozo(b'IEND', b''))

    kb = os.path.getsize(path) / 1024.0
    print(f"  {os.path.basename(path)}  {nx}x{ny}  {prof} bits  {kb:.0f} KB")
    if kb < 200:
        print("     AVISO: el archivo es sospechosamente pequeno")


def guardar_png16(arr01, path):
    _escribir_png_gris(path, arr01, 16)


def guardar_png8(arr01, path):
    _escribir_png_gris(path, arr01, 8)


# =============================================================================
# 2) MONTAR LA ESCENA
# =============================================================================

def limpiar():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for bloque in (bpy.data.meshes, bpy.data.materials, bpy.data.textures):
        for d in list(bloque):
            if d.users == 0:
                bloque.remove(d)


def crear_rejilla(bajo):
    """Rejilla regular SUBDIV_X x SUBDIV_Y con la altura YA metida en la Z de
    cada vertice.

    La alternativa habitual es crear un plano liso y poner el relieve con un
    modificador Displace que lee altura.png a traves de las UV. Eso son tres
    piezas que tienen que encajar (PNG guardado bien + UV bien + modificador
    bien) y si falla cualquiera de las tres el resultado es un plano
    perfectamente liso, sin ningun mensaje de error. Muestreando el heightmap
    aqui no hay nada que encajar.

    Se muestrea 'bajo' (el heightmap promediado por bloques al tamano de la
    malla) en vez de H, para no meter aliasing: el detalle mas fino que la
    malla ya va por bump map en el material.
    """
    nx, ny = SUBDIV_X, SUBDIV_Y
    xs = np.linspace(-ANCHO_KM / 2, ANCHO_KM / 2, nx, dtype=np.float32)
    ys = np.linspace(-ALTO_KM / 2, ALTO_KM / 2, ny, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    # indices en el heightmap. Fila 0 del array = borde SUPERIOR (y = +ALTO/2)
    col = np.clip(((gx + ANCHO_KM / 2) / ANCHO_KM * (RES_X - 1)).astype(np.int32),
                  0, RES_X - 1)
    fil = np.clip(((ALTO_KM / 2 - gy) / ALTO_KM * (RES_Y - 1)).astype(np.int32),
                  0, RES_Y - 1)
    z_km = bajo[fil, col] / 1000.0          # metros -> km (1 BU = 1 km)

    verts = np.empty((ny * nx, 3), np.float32)
    verts[:, 0] = gx.ravel()
    verts[:, 1] = gy.ravel()
    verts[:, 2] = z_km.ravel()

    idx = np.arange(ny * nx, dtype=np.int32).reshape(ny, nx)
    quads = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]],
                     axis=-1).reshape(-1, 4)

    me = bpy.data.meshes.new("TerrenoMesh")
    construir_malla(me, verts, quads)

    obj = bpy.data.objects.new("Terreno", me)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # UV: (0,0) abajo izquierda -> (1,1) arriba derecha.
    # Solo las usan las texturas de albedo y de bump del material.
    uv = me.uv_layers.new(name="UVMap")
    loop_v = np.empty(len(me.loops), np.int32)
    me.loops.foreach_get("vertex_index", loop_v)
    u = (verts[loop_v, 0] + ANCHO_KM / 2) / ANCHO_KM
    v = (verts[loop_v, 1] + ALTO_KM / 2) / ALTO_KM
    uv.data.foreach_set("uv", np.stack([u, v], axis=-1).astype(np.float32).ravel())

    for p in me.polygons:
        p.use_smooth = True

    print(f"Malla: {nx}x{ny} = {nx*ny} vertices, {len(quads)} caras")
    print(f"  Z de los vertices: {z_km.min():.3f} a {z_km.max():.3f} km "
          f"(relieve {z_km.max()-z_km.min():.3f} km)")
    if z_km.max() - z_km.min() < 0.01:
        print("  AVISO: el relieve es practicamente nulo. Algo va mal.")
    return obj


def construir_malla(me, verts, quads):
    """Rellena la malla con la API rapida, y si falla cae a from_pydata.

    from_pydata necesita listas de Python: para 6.75 M de vertices eso son
    6.75 M de tuplas de floats, casi 1 GB solo en objetos Python, y tarda
    minutos. La via de foreach_set copia el buffer de numpy tal cual.
    """
    nv = len(verts)
    nq = len(quads)
    try:
        me.vertices.add(nv)
        me.vertices.foreach_set("co", verts.ravel())

        me.loops.add(nq * 4)
        me.loops.foreach_set("vertex_index", quads.ravel())

        me.polygons.add(nq)
        me.polygons.foreach_set("loop_start",
                                np.arange(0, nq * 4, 4, dtype=np.int32))
        # loop_total solo existe hasta Blender 3.x; en 4.x se deduce
        try:
            me.polygons.foreach_set("loop_total",
                                    np.full(nq, 4, dtype=np.int32))
        except Exception:
            pass

        me.update(calc_edges=True)
        me.validate()
        print(f"  malla construida con foreach_set ({nv} vertices)")
    except Exception as e:
        print(f"  foreach_set fallo ({e}), uso from_pydata. Ira mas lento.")
        me.clear_geometry()
        me.from_pydata(verts.tolist(), [], quads.tolist())
        me.update()


def imagen(nombre, datos=False):
    path = ruta(nombre)
    if not os.path.isfile(path):
        print(f"  AVISO: no encuentro {nombre}")
        return None
    img = bpy.data.images.load(path, check_existing=True)
    if datos:
        img.colorspace_settings.name = 'Non-Color'
    print(f"  textura {nombre}: {img.size[0]}x{img.size[1]}")
    return img


def crear_material(obj, d_max):
    mat = bpy.data.materials.new("Regolito")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (600, 0)

    # Regolito: difusor rugoso (Oren-Nayar). Es mucho mejor modelo para un
    # suelo sin atmosfera que un Principled con especular.
    bsdf = nt.nodes.new("ShaderNodeBsdfDiffuse"); bsdf.location = (350, 0)
    bsdf.inputs["Roughness"].default_value = 0.9

    # OJO con el espacio de color. El array de albedo son reflectancias
    # LINEALES (0.42 en tierras altas, 0.15 en mares). Si la textura se carga
    # como sRGB, Blender la lineariza otra vez: 0.42 se convierte en 0.147 y
    # todo el render sale casi tres veces mas oscuro de lo que toca. Cargarla
    # como Non-Color deja el valor tal cual.
    img_alb = imagen("albedo.png", datos=True)
    tex_alb = nt.nodes.new("ShaderNodeTexImage"); tex_alb.location = (-200, 150)
    tex_alb.image = img_alb
    tex_alb.interpolation = 'Cubic'
    tex_alb.extension = 'EXTEND'

    # tinte lunar: gris muy ligeramente calido
    mezcla = nt.nodes.new("ShaderNodeMixRGB"); mezcla.location = (100, 150)
    mezcla.blend_type = 'MULTIPLY'
    mezcla.inputs["Fac"].default_value = 1.0
    mezcla.inputs["Color2"].default_value = (1.00, 0.98, 0.94, 1.0)

    img_det = imagen("detalle.png", datos=True)
    tex_det = nt.nodes.new("ShaderNodeTexImage"); tex_det.location = (-200, -250)
    tex_det.image = img_det
    tex_det.interpolation = 'Cubic'
    tex_det.extension = 'EXTEND'

    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (100, -250)
    # el detalle esta guardado centrado en 0.5 con rango +-d_max metros
    bump.inputs["Distance"].default_value = (2 * d_max) / 1000.0
    bump.inputs["Strength"].default_value = 1.0

    # Si alguna textura no esta, se conecta lo que si haya. El relieve ya vive
    # en la geometria, asi que el material puede fallar sin arruinar la escena.
    if img_alb is not None:
        nt.links.new(tex_alb.outputs["Color"], mezcla.inputs["Color1"])
        nt.links.new(mezcla.outputs["Color"], bsdf.inputs["Color"])
    else:
        bsdf.inputs["Color"].default_value = (0.42, 0.41, 0.39, 1.0)
    if img_det is not None:
        nt.links.new(tex_det.outputs["Color"], bump.inputs["Height"])
        nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    obj.data.materials.append(mat)
    print("Material de regolito creado")


def crear_sol():
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 400))
    sol = bpy.context.active_object
    sol.name = "Sol"
    sol.data.energy = SOL_FUERZA
    sol.data.angle = math.radians(0.53)     # tamano angular real del Sol
    sol.data.color = (1.0, 0.98, 0.95)

    # Rotacion a partir de elevacion y azimut. Un sol a 90 grados apuntaria
    # recto hacia abajo (-Z), que aplana el relieve por completo en el render.
    el = math.radians(SOL_ELEVACION)
    az = math.radians(SOL_AZIMUT)
    sol.rotation_euler = (math.radians(90.0) - el, 0.0, az)
    print(f"Sol: elevacion {SOL_ELEVACION} deg, azimut {SOL_AZIMUT} deg")
    return sol


def crear_camara():
    bpy.ops.object.camera_add(location=(0, 0, ALTURA_CAMARA_KM))
    cam = bpy.context.active_object
    cam.name = "CamaraNave"
    cam.rotation_euler = (0.0, 0.0, 0.0)      # nadir

    d = cam.data
    d.sensor_fit = 'HORIZONTAL'
    d.sensor_width = 36.0
    d.lens = (d.sensor_width / 2.0) / math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    d.clip_start = 1.0
    d.clip_end = 2000.0

    bpy.context.scene.camera = cam
    huella = 2 * ALTURA_CAMARA_KM * math.tan(math.radians(FOV_HORIZONTAL / 2))
    print(f"Camara: f={d.lens:.2f} mm  FOV_h={FOV_HORIZONTAL} deg  "
          f"huella {huella:.1f} x {huella*RENDER_H/RENDER_W:.1f} km")
    return cam


def crear_camara_mapa():
    """Camara ORTOGRAFICA que encuadra el terreno exacto, para renderizar el
    mapa de referencia.

    El encuadre tiene que ser el terreno exacto, sin margenes. Si el render
    sale en 16:9 quedan franjas vacias a los lados, y entonces la escala en
    kilometros que el codigo de MATLAB deduce del ancho de la imagen queda
    infraestimada. Con esta camara el encuadre es exactamente 840 x 630 km en
    4:3, y entonces
        km_por_pixel = 840 / mapWidth = 630 / mapHeight
    es correcto y ademas isotropo.
    """
    bpy.ops.object.camera_add(location=(0, 0, 400))
    cam = bpy.context.active_object
    cam.name = "CamaraMapa"
    cam.rotation_euler = (0.0, 0.0, 0.0)
    d = cam.data
    d.type = 'ORTHO'
    d.ortho_scale = ANCHO_KM          # encuadra justo los 840 km de ancho
    d.clip_start = 1.0
    d.clip_end = 2000.0
    print(f"CamaraMapa ortografica: {ANCHO_KM} x {ALTO_KM} km exactos, 4:3")
    print("  renderiza con ella a 2048x1536 para obtener mapa.png")
    return cam


def configurar_render():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 128
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.diffuse_bounces = 4       # la luz rebotada llena las sombras
    sc.render.resolution_x = RENDER_W
    sc.render.resolution_y = RENDER_H
    sc.render.resolution_percentage = 100
    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'BW'
    sc.render.filepath = os.path.join(RUTA_PROYECTO, "Output", "")

    # espacio: fondo negro, sin luz ambiente falsa
    world = bpy.data.worlds.get("Espacio") or bpy.data.worlds.new("Espacio")
    sc.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs["Color"].default_value = (0, 0, 0, 1)
        bg.inputs["Strength"].default_value = 0.0

    sc.unit_settings.system = 'METRIC'
    sc.unit_settings.scale_length = 1000.0     # 1 unidad Blender = 1 km
    sc.unit_settings.length_unit = 'KILOMETERS'
    print("Render: Cycles, 1920x1080, escala 1 BU = 1 km")


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(RUTA_PROYECTO, exist_ok=True)
    os.makedirs(os.path.join(RUTA_PROYECTO, "Output"), exist_ok=True)

    H, bajo, h_min, h_max, d_max = generar_mapas()

    if LIMPIAR_ESCENA:
        limpiar()

    configurar_render()
    obj = crear_rejilla(bajo)
    crear_material(obj, d_max)
    crear_sol()
    crear_camara()
    crear_camara_mapa()

    print("=" * 70)
    print("LISTO.")
    print("Siguiente paso: anima la camara y ejecuta")
    print("blender_exportar_etiquetas.py para sacar las etiquetas YOLO")
    print("verdaderas de cada frame.")
    print("=" * 70)


if __name__ == "__main__":
    main()
