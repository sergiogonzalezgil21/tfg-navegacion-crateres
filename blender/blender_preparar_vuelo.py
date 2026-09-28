"""
Prepara el vuelo de crucero: renderiza el mapa de referencia con la camara
ortografica (840 x 630 km exactos, 4:3) y anima CamaraNave a lo largo de una
trayectoria de 700 frames que baja de 170 a 120 km serpenteando en Y. Deja el
render configurado pero no lo lanza.

La camara del mapa es ortografica para que el encuadre sea el terreno exacto.
Con un render en 16:9 quedan franjas vacias a los lados y la escala en km que
deduce MATLAB del ancho de la imagen sale corta; con ortho_scale = 840,
km_por_pixel = 840 / mapWidth = 630 / mapHeight, correcto e isotropo.

Se ejecuta con terreno_lunar.blend abierto.
"""

import os
import math

import bpy
import numpy as np

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

ANCHO_KM = 840.0
ALTO_KM = 630.0
FOV_HORIZONTAL = 60.0

# ---- PASO 1: mapa de referencia ----
MAPA_W, MAPA_H = 4096, 3072          # 4:3, igual que el terreno (205 m/px)

# ---- PASO 2: trayectoria ----
# Estos valores NO son arbitrarios: salen de buscar en el espacio de
# parametros la ruta mas larga posible que deja 0 % de frames con menos de 3
# landmarks visibles y 0 frames saliendose del terreno.
FRAMES = 700
ALT_INICIAL_KM = 170.0               # arranca alto
ALT_FINAL_KM = 120.0                 # y desciende 50 km
X_INI_KM, X_FIN_KM = 100.0, 740.0    # cruza el mapa entero de oeste a este
Y_CENTRO_KM = 350.0
Y_AMPLITUD_KM = 130.0                # serpenteo amplio en Y
Y_CICLOS = 2.0                       # dos ondulaciones completas
# Resultado con los 180 landmarks del mapa de 840x630:
#   recorrido sobre el suelo   1263 km
#   landmarks por frame        media 6.3, minimo 3
#   frames con <3 landmarks    0 %
#   frames recortados          0 %
#   huella  196x110 km -> 139x78 km     escala  102 -> 72 m/pixel
# Con 700 frames la camara avanza 1.8 km entre frames, un 1.3 % del ancho del
# encuadre: solapamiento de sobra para la continuidad de la trayectoria, y
# suficiente variedad para que el dataset de entrenamiento no sea redundante.

# ---- PASO 3: render ----
MOTOR = 'CYCLES'                     # 'CYCLES' o 'EEVEE'
MUESTRAS = 64

# Gestion de color. Blender 4.x usa AgX por defecto: una curva de tono
# cinematografica que comprime mucho las luces y ensucia los grises. Para un
# render que va a alimentar a un detector no interesa: hace falta el mayor
# contraste util posible y una relacion directa entre lo que calcula el motor
# y el pixel que sale. Con 'Standard' esa relacion es directa.
VISTA = 'Standard'
EXPOSICION = 0.0

# Fuerza del sol, ajustada para que 'Standard' de un gris medio correcto.
# Con reflectancia lambertiana: L = S * cos(theta) * albedo / pi
# Con S=4, sol a 28 grados (cos 62 = 0.469) y albedo 0.41 de las tierras
# altas: L = 0.25, que en sRGB son unos 0.54. Deja sitio de sobra arriba para
# las paredes de crater iluminadas de frente, sin quemarlas.
SOL_FUERZA = 4.0
USAR_GPU = True
RENDERIZAR_MAPA_AHORA = True         # el mapa es 1 solo frame, es rapido
RENDERIZAR_VUELO_AHORA = False       # los 500 frames: se lanza aparte, bajo demanda

# =============================================================================


def ruta(nombre):
    return os.path.join(RUTA_PROYECTO, nombre)


def obj(nombre):
    o = bpy.data.objects.get(nombre)
    if o is None:
        raise RuntimeError(
            f"No encuentro '{nombre}'. Abre terreno_lunar.blend, el que creo "
            f"blender_construir_terreno.py.")
    return o


def configurar_motor():
    sc = bpy.context.scene

    if MOTOR == 'CYCLES':
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = MUESTRAS
        sc.cycles.use_denoising = True
        sc.cycles.max_bounces = 6
        sc.cycles.diffuse_bounces = 4

        if USAR_GPU:
            try:
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.get_devices()
                for tipo in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                    try:
                        prefs.compute_device_type = tipo
                        break
                    except TypeError:
                        continue
                n = 0
                for d in prefs.devices:
                    if d.type != 'CPU':
                        d.use = True
                        n += 1
                sc.cycles.device = 'GPU'
                print(f"  Cycles en GPU ({prefs.compute_device_type}, {n} dispositivos)")
            except Exception as e:
                print(f"  No he podido activar la GPU, sigo en CPU: {e}")
    else:
        # el nombre del motor cambio entre versiones de Blender
        for nombre in ('BLENDER_EEVEE_NEXT', 'BLENDER_EEVEE'):
            try:
                sc.render.engine = nombre
                break
            except TypeError:
                continue
        print(f"  Motor: {sc.render.engine}")

    sc.render.image_settings.file_format = 'PNG'
    sc.render.image_settings.color_mode = 'BW'
    sc.render.image_settings.color_depth = '8'

    configurar_color()


def configurar_color():
    """Gestion de color y exposicion.

    El primer mapa salio con un brillo maximo de 0.553 y una media de 0.231:
    subexpuesto y sin contraste, con los mares practicamente a negro. El
    culpable es AgX, el transformador de vista por defecto de Blender 4.x.
    """
    sc = bpy.context.scene
    try:
        sc.view_settings.view_transform = VISTA
        sc.view_settings.look = 'None'
        sc.view_settings.exposure = EXPOSICION
        sc.view_settings.gamma = 1.0
        print(f"  Vista: {VISTA}, exposicion {EXPOSICION}")
    except TypeError:
        print(f"  AVISO: esta version no acepta view_transform='{VISTA}'")

    # se relee lo que ha quedado puesto: asignar no garantiza que se aplique
    print(f"  view_transform efectivo: '{sc.view_settings.view_transform}'  "
          f"look '{sc.view_settings.look}'")

    sol = bpy.data.objects.get("Sol")
    if sol is not None and sol.type == 'LIGHT':
        sol.data.energy = SOL_FUERZA
        elev = 90.0 - math.degrees(sol.rotation_euler.x)
        print(f"  Sol: fuerza {SOL_FUERZA}, elevacion {elev:.1f} grados")

    corregir_espacio_color()


def corregir_espacio_color():
    """El mapa de albedo son reflectancias LINEALES, no una imagen sRGB.

    Este era el motivo real de que el render saliera oscuro, y no la fuerza
    del sol. albedo.png guarda 0.42 para las tierras altas; si Blender lo trata
    como sRGB lo lineariza y se queda en 0.147, casi tres veces menos. Con el
    sol a 4 W/m2 y 28 grados de elevacion:

        como dato   L = 0.245  ->  0.53 en pantalla   (correcto)
        como sRGB   L = 0.084  ->  0.32 en pantalla   (lo que salia)

    En los mares es aun peor: 0.33 frente a 0.09, practicamente negro.
    """
    for nombre in ("albedo.png", "detalle.png"):
        img = bpy.data.images.get(nombre)
        if img is None:
            continue
        previo = img.colorspace_settings.name
        if previo != 'Non-Color':
            img.colorspace_settings.name = 'Non-Color'
            print(f"  {nombre}: espacio de color {previo} -> Non-Color")
        else:
            print(f"  {nombre}: Non-Color (ya estaba bien)")


# =============================================================================
# PASO 1 - MAPA DE REFERENCIA
# =============================================================================

def renderizar_mapa():
    sc = bpy.context.scene
    cam_mapa = obj("CamaraMapa")

    # guardar el estado para dejarlo como estaba
    cam_previa = sc.camera
    res_previa = (sc.render.resolution_x, sc.render.resolution_y)
    ruta_previa = sc.render.filepath

    sc.camera = cam_mapa
    sc.render.resolution_x = MAPA_W
    sc.render.resolution_y = MAPA_H
    sc.render.filepath = ruta("mapa")          # Blender le pone la extension

    km_px_x = ANCHO_KM / MAPA_W
    km_px_y = ALTO_KM / MAPA_H
    print(f"  Encuadre: {ANCHO_KM} x {ALTO_KM} km en {MAPA_W}x{MAPA_H} px")
    print(f"  km por pixel: {km_px_x:.6f} en X, {km_px_y:.6f} en Y")
    if abs(km_px_x - km_px_y) > 1e-6:
        print("  AVISO: no es isotropo. Revisa MAPA_W/MAPA_H, deben ser 4:3.")

    if RENDERIZAR_MAPA_AHORA:
        print("  Renderizando mapa.png...")
        bpy.ops.render.render(write_still=True)
        print(f"  Guardado: {ruta('mapa.png')}")
        informar_exposicion("mapa.png")
    else:
        print("  (RENDERIZAR_MAPA_AHORA esta en False, no lo he renderizado)")

    sc.camera = cam_previa
    sc.render.resolution_x, sc.render.resolution_y = res_previa
    sc.render.filepath = ruta_previa


def informar_exposicion(nombre):
    """Lee el PNG recien escrito y dice si la exposicion es utilizable.

    Un render demasiado oscuro es el error silencioso mas facil de colar:
    parece que funciona, se ve algo en pantalla, y el detector luego no
    encuentra nada porque no hay contraste.

    Es SOLO un diagnostico, asi que va entero dentro de un try: que falle el
    informe no puede tumbar el script y dejar la camara sin animar.
    """
    path = ruta(nombre)
    if not os.path.isfile(path):
        return

    img = None
    try:
        img = bpy.data.images.load(path, check_existing=False)

        # Con foreach_get y numpy. Antes hacia list(img.pixels[0:n:4]) y eso
        # fallaba: bpy_prop_array no admite slice con paso. Y aunque lo
        # admitiera, convertir 50 millones de floats a una lista de Python
        # serian mas de 2 GB.
        n = len(img.pixels)
        buf = np.empty(n, dtype=np.float32)
        img.pixels.foreach_get(buf)
        px = buf[0::4]                      # solo el canal rojo

        mn = float(px.min())
        mx = float(px.max())
        media = float(px.mean())
        oscuros = float((px < 0.02).mean())

        print(f"  Exposicion: min {mn:.3f}  max {mx:.3f}  media {media:.3f}  "
              f"negros {100*oscuros:.1f} %")
        # Umbrales realistas para una vista nadir con sol rasante. El maximo
        # NO puede acercarse a 1: con el sol a 28 grados sobre el horizonte,
        # el suelo llano refleja cos(62) = 0.47 de la irradiancia, y solo las
        # paredes de crater inclinadas hacia el sol suben de ahi. Un maximo
        # de 0.70 y una mediana de 0.34 es exactamente lo que toca.
        if media < 0.20 or mx < 0.45:
            print("     AVISO: se queda corto de luz. Sube SOL_FUERZA.")
        elif media > 0.65 or (px > 0.99).mean() > 0.01:
            print("     AVISO: demasiado claro, se estan quemando luces.")
        else:
            print("     Rango correcto.")
    except Exception as e:
        print(f"  (no he podido medir la exposicion: {e})")
    finally:
        if img is not None:
            try:
                bpy.data.images.remove(img)
            except Exception:
                pass


# =============================================================================
# PASO 2 - TRAYECTORIA DE LA CAMARA
# =============================================================================

def _fcurves_de(accion):
    """Devuelve las F-curves de una accion, sea cual sea la version de Blender.

    Hasta 4.3 bastaba con accion.fcurves. En 4.4 llegaron las acciones por
    capas (slotted actions) y ese atributo desaparecio: ahora las curvas
    cuelgan de layers -> strips -> channelbags.
    """
    fc = getattr(accion, "fcurves", None)
    if fc is not None:
        return list(fc)

    salida = []
    for capa in getattr(accion, "layers", []):
        for tira in getattr(capa, "strips", []):
            bolsas = getattr(tira, "channelbags", None)
            if bolsas is None:
                continue
            for bolsa in bolsas:
                salida.extend(bolsa.fcurves)
    return salida


def suave(t):
    """smoothstep: arranca y acaba con velocidad nula, sin tirones."""
    return t * t * (3.0 - 2.0 * t)


def animar_camara():
    sc = bpy.context.scene
    cam = obj("CamaraNave")

    # limpiar cualquier animacion anterior
    cam.animation_data_clear()

    sc.frame_start = 1
    sc.frame_end = FRAMES

    media_fov = math.radians(FOV_HORIZONTAL / 2.0)
    aspecto = sc.render.resolution_y / sc.render.resolution_x

    recortados = 0
    huellas = []

    # Interpolacion lineal: con bezier la camara se pasa de largo en las
    # curvas. Se fija en las preferencias ANTES de insertar, que es la via que
    # funciona en todas las versiones.
    prefs_edit = bpy.context.preferences.edit
    interp_previa = prefs_edit.keyframe_new_interpolation_type
    prefs_edit.keyframe_new_interpolation_type = 'LINEAR'

    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        f = i + 1

        alt = ALT_INICIAL_KM + (ALT_FINAL_KM - ALT_INICIAL_KM) * suave(t)

        x = X_INI_KM + (X_FIN_KM - X_INI_KM) * t
        y = Y_CENTRO_KM + Y_AMPLITUD_KM * math.sin(2 * math.pi * Y_CICLOS * t)

        # La huella en el suelo no puede salirse del terreno: si se sale,
        # aparece fondo negro en el frame y ese frame no vale para entrenar.
        semi_x = alt * math.tan(media_fov)
        semi_y = semi_x * aspecto
        huellas.append((2 * semi_x, 2 * semi_y))

        x_min, x_max = semi_x, ANCHO_KM - semi_x
        y_min, y_max = semi_y, ALTO_KM - semi_y
        x_rec = min(max(x, x_min), x_max)
        y_rec = min(max(y, y_min), y_max)
        if abs(x_rec - x) > 1e-6 or abs(y_rec - y) > 1e-6:
            recortados += 1

        # a coordenadas de mundo: el terreno esta centrado en el origen
        cam.location = (x_rec - ANCHO_KM / 2.0, y_rec - ALTO_KM / 2.0, alt)

        # Nadir, sin guinada. El MATLAB actual NO estima orientacion: suma el
        # vector del frame al mapa sin rotarlo, asi que asume yaw = 0. Si algun
        # dia metes rotacion aqui, hay que arreglar antes el matching.
        cam.rotation_euler = (0.0, 0.0, 0.0)

        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)

    prefs_edit.keyframe_new_interpolation_type = interp_previa

    # por si acaso, se fuerza tambien sobre las curvas ya creadas
    if cam.animation_data and cam.animation_data.action:
        for fc in _fcurves_de(cam.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    print(f"  {FRAMES} keyframes, del 1 al {FRAMES}")
    print(f"  Altura: {ALT_INICIAL_KM} -> {ALT_FINAL_KM} km")
    print(f"  Huella en el suelo: {huellas[0][0]:.1f} x {huellas[0][1]:.1f} km "
          f"al principio, {huellas[-1][0]:.1f} x {huellas[-1][1]:.1f} km al final")
    if recortados:
        print(f"  {recortados} frames recortados para no salirse del terreno")

    # cuantos landmarks se ven de media, para saber si la ruta es aprovechable
    estimar_cobertura(huellas)


def estimar_cobertura(huellas):
    """Cuenta cuantos de los 18 landmark caen dentro de cada huella."""
    csv_path = ruta("mapa_landmarks.csv")
    if not os.path.isfile(csv_path):
        return

    import csv as _csv
    lm = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            lm.append((float(row["x_km"]), float(row["y_km"])))

    sc = bpy.context.scene
    cam = obj("CamaraNave")
    frame_previo = sc.frame_current

    cuenta = []
    for i in range(0, FRAMES, 5):
        sc.frame_set(i + 1)
        cx = cam.location.x + ANCHO_KM / 2.0
        cy = cam.location.y + ALTO_KM / 2.0
        w, h = huellas[i]
        n = sum(1 for (lx, ly) in lm
                if abs(lx - cx) < w / 2 and abs(ly - cy) < h / 2)
        cuenta.append(n)

    sc.frame_set(frame_previo)

    media = sum(cuenta) / len(cuenta)
    pobres = sum(1 for c in cuenta if c < 3)
    print(f"  Landmarks visibles: media {media:.1f} por frame")
    print(f"  Frames con menos de 3 landmarks: {100*pobres/len(cuenta):.0f} %")
    if pobres / len(cuenta) > 0.35:
        print("  AVISO: demasiados frames con pocos landmarks. Baja la altura")
        print("         final o reduce Y_AMPLITUD_KM para pasar mas por el centro.")


# =============================================================================
# PASO 3 - SALIDA
# =============================================================================

def configurar_salida():
    sc = bpy.context.scene
    salida = os.path.join(RUTA_PROYECTO, "Output", "")
    os.makedirs(salida, exist_ok=True)

    sc.camera = obj("CamaraNave")
    sc.render.resolution_x = 1920
    sc.render.resolution_y = 1080
    sc.render.resolution_percentage = 100
    sc.render.filepath = salida
    sc.render.use_file_extension = True
    sc.frame_step = 1

    print(f"  Salida: {salida}")
    print("  Los frames saldran como 0001.png ... 0500.png")


def main():
    print("=" * 70)
    print("PASO 0  Motor de render")
    configurar_motor()

    # La salida se configura ANTES de animar: animar_camara() calcula la huella
    # en el suelo a partir de la resolucion del render, y si todavia estuviera
    # puesta la del mapa (4:3) saldria una huella con el aspecto equivocado.
    print("PASO 1  Configuracion de salida")
    configurar_salida()

    print("PASO 2  Mapa de referencia (camara ortografica)")
    renderizar_mapa()

    print("PASO 3  Trayectoria de la camara")
    animar_camara()

    print("=" * 70)
    print("LISTO.")
    print()
    print("Para renderizar los 500 frames:  Render -> Render Animation (Ctrl+F12)")
    print("Cuando terminen, ejecuta blender_exportar_etiquetas.py")
    print("=" * 70)

    if RENDERIZAR_VUELO_AHORA:
        print("Renderizando los 500 frames. Esto tarda un buen rato...")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
