"""
Anade una segunda camara, "CamaraDescenso", con un descenso parabolico de 170
a 55 km: la guinada gira +-180 grados durante la bajada y se estabiliza en el
tramo final, como haria un aterrizador de verdad.

Al bajar, el area de la huella encoge con el cuadrado de la altura y el numero
de crateres de escala de mapa dentro del encuadre se desploma. En algun punto
baja de tres y el emparejamiento por tripletas se queda sin material. Este
vuelo esta hecho para cruzar ese limite y poder medirlo con mapas de distinta
densidad.

Para en 55 km por el terreno, no por la navegacion: mas abajo el albedo
(0.205 km/pixel) se estira mas de cuatro veces y la imagen se degrada, con lo
que no se sabria si el fallo viene del mapa o del render.

Se ejecuta con terreno_lunar.blend abierto. No rehace el mapa de referencia.
"""

import os
import math

import bpy

# =============================================================================

# --- Rutas relativas al propio fichero ------------------------------------
# El repositorio se puede clonar en cualquier carpeta, asi que ninguna ruta
# esta escrita a mano. __file__ no existe cuando el codigo se pega dentro de
# un editor (el de Blender, por ejemplo); en ese caso se usa el directorio de
# trabajo.
_AQUI = (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
         else os.path.abspath(os.getcwd()))

RUTA_PROYECTO = _AQUI          # carpeta blender/ del repositorio
SUBCARPETA = "Output_descenso"

ANCHO_KM, ALTO_KM = 840.0, 630.0
FOV_HORIZONTAL = 60.0
FRAMES = 700

ALT_INICIAL_KM = 170.0
ALT_FINAL_KM = 55.0          # ver la nota de arriba sobre por que no menos

X_INI_KM, X_FIN_KM = 130.0, 690.0
Y_CENTRO_KM = 350.0
Y_AMPLITUD_KM = 120.0
Y_CICLOS = 1.5

GIRO_AMPLITUD_DEG = 180.0
GIRO_CICLOS = 2.0
EST_INI, EST_FIN = 0.60, 0.82    # tramo en que la guinada se va a cero

RENDERIZAR_AHORA = False

# =============================================================================


def suave(t):
    """smootherstep: arranca y termina con derivada nula."""
    t = min(max(t, 0.0), 1.0)
    return t * t * t * (t * (t * 6 - 15) + 10)


def obj(nombre):
    o = bpy.data.objects.get(nombre)
    if o is None:
        raise RuntimeError(f"no existe el objeto '{nombre}' en la escena")
    return o


def _fcurves_de(accion):
    """Acceso a las F-curves valido en 4.4+ (acciones con slots) y anteriores."""
    if hasattr(accion, "fcurves") and len(accion.fcurves):
        return list(accion.fcurves)
    out = []
    for capa in getattr(accion, "layers", []):
        for tira in getattr(capa, "strips", []):
            for bolsa in getattr(tira, "channelbags", []):
                out.extend(bolsa.fcurves)
    return out


def altura(t):
    """Parabola: empieza plana y baja cada vez mas deprisa."""
    return ALT_INICIAL_KM + (ALT_FINAL_KM - ALT_INICIAL_KM) * (t * t)


def guinada(t):
    """Sinusoide con envolvente que se anula en la aproximacion final."""
    if t <= EST_INI:
        env = 1.0
    elif t >= EST_FIN:
        env = 0.0
    else:
        env = 1.0 - suave((t - EST_INI) / (EST_FIN - EST_INI))
    return math.radians(GIRO_AMPLITUD_DEG * env
                        * math.sin(2 * math.pi * GIRO_CICLOS * t))


def crear_camara():
    """Duplica los ajustes opticos de la camara del vuelo original."""
    base = bpy.data.objects.get("CamaraNave")
    if base is None:
        raise RuntimeError("falta 'CamaraNave': corre antes blender_preparar_vuelo")

    vieja = bpy.data.objects.get("CamaraDescenso")
    if vieja is not None:
        bpy.data.objects.remove(vieja, do_unlink=True)

    datos = base.data.copy()
    datos.name = "CamaraDescensoDatos"
    cam = bpy.data.objects.new("CamaraDescenso", datos)
    bpy.context.scene.collection.objects.link(cam)
    print(f"  Optica copiada de CamaraNave: f={datos.lens:.2f} mm, "
          f"FOV_h={math.degrees(datos.angle_x):.1f} grados")
    return cam


def animar():
    sc = bpy.context.scene
    cam = crear_camara()
    cam.animation_data_clear()

    sc.frame_start = 1
    sc.frame_end = FRAMES

    # La resolucion se fija AQUI, antes de calcular nada.
    #
    # El aspecto del render entra en el margen de seguridad (la huella en el
    # suelo es 16:9, no cuadrada). Si al ejecutar esto la escena estuviera
    # todavia con la resolucion del mapa, que es 4:3, saldria una trayectoria
    # DISTINTA de la que se renderizo, y las etiquetas no cuadrarian con las
    # imagenes. El script original ya avisaba de esta trampa y aun asi cai en
    # ella: dejaba la configuracion de salida para despues de animar.
    sc.render.resolution_x = 1920
    sc.render.resolution_y = 1080
    sc.render.resolution_percentage = 100

    media_fov = math.radians(FOV_HORIZONTAL / 2.0)
    aspecto = sc.render.resolution_y / sc.render.resolution_x
    print(f"  Render {sc.render.resolution_x}x{sc.render.resolution_y}, "
          f"aspecto {aspecto:.4f}")

    prefs = bpy.context.preferences.edit
    interp_previa = prefs.keyframe_new_interpolation_type
    prefs.keyframe_new_interpolation_type = 'LINEAR'

    recortados = 0
    for i in range(FRAMES):
        t = i / (FRAMES - 1)
        f = i + 1

        alt = altura(t)
        giro = guinada(t)

        # aproximacion decelerando: casi todo el alcance se cubre arriba
        x = X_INI_KM + (X_FIN_KM - X_INI_KM) * (1.0 - (1.0 - t) ** 2)
        # el serpenteo lateral se amortigua conforme se acerca al suelo
        y = Y_CENTRO_KM + Y_AMPLITUD_KM * ((1.0 - t) ** 1.5) \
            * math.sin(2 * math.pi * Y_CICLOS * t)

        # margen: con guinada la huella esta girada y su caja envolvente crece
        semi_x = alt * math.tan(media_fov)
        semi_y = semi_x * aspecto
        c, sn = abs(math.cos(giro)), abs(math.sin(giro))
        env_x = semi_x * c + semi_y * sn
        env_y = semi_x * sn + semi_y * c

        x_rec = min(max(x, env_x), ANCHO_KM - env_x)
        y_rec = min(max(y, env_y), ALTO_KM - env_y)
        if abs(x_rec - x) > 1e-6 or abs(y_rec - y) > 1e-6:
            recortados += 1

        cam.location = (x_rec - ANCHO_KM / 2.0, y_rec - ALTO_KM / 2.0, alt)
        cam.rotation_euler = (0.0, 0.0, giro)
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_euler", frame=f)

    prefs.keyframe_new_interpolation_type = interp_previa

    if cam.animation_data and cam.animation_data.action:
        for fc in _fcurves_de(cam.animation_data.action):
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    print(f"  {FRAMES} keyframes")
    print(f"  Altura: {ALT_INICIAL_KM:.0f} -> {ALT_FINAL_KM:.0f} km (parabolica)")
    print(f"  Guinada: +-{GIRO_AMPLITUD_DEG:.0f} grados hasta t={EST_INI:.2f}, "
          f"estabilizada desde t={EST_FIN:.2f} "
          f"(frames {int(EST_FIN*FRAMES)} al {FRAMES})")
    if recortados:
        print(f"  {recortados} frames recortados para no salirse del terreno")
    else:
        print("  0 frames recortados")

    # resumen para poder cotejar que es la MISMA trayectoria que se renderizo
    a0, a1 = altura(0.0), altura(1.0)
    print(f"  Comprobacion:  frame 1 -> altura {a0:.2f} km,"
          f"  frame {FRAMES} -> altura {a1:.2f} km")
    return cam


def prediccion_landmarks():
    """A que altura se queda sin landmarks cada mapa. Es el experimento."""
    import csv
    p = os.path.join(RUTA_PROYECTO, "crateres_catalogo.csv")
    if not os.path.isfile(p):
        print("  (no encuentro crateres_catalogo.csv, me salto la prediccion)")
        return

    D = []
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if float(r["frescura"]) >= 0.30:
                D.append(float(r["D_km"]))

    area = ANCHO_KM * ALTO_KM
    tan = math.tan(math.radians(FOV_HORIZONTAL / 2.0))
    # k sale de calibrar contra lo medido: 5.94 crateres/frame a D>=10 y 145 km
    n10 = sum(1 for d in D if d >= 10.0)
    huella145 = (2 * 145 * tan) ** 2 * (1080 / 1920)
    k = 5.94 / (n10 / area * huella145)

    print()
    print("  PREDICCION: crateres de mapa por frame durante el descenso")
    print(f"  {'altura':>8} | " + " ".join(f"{'D>='+str(d):>8}" for d in (10, 6, 4)))
    print("  " + "-" * 42)
    for alt in (170, 150, 130, 110, 100, 90, 80, 70, 60, 55):
        fila = []
        for dmin in (10, 6, 4):
            n = sum(1 for d in D if d >= dmin)
            huella = (2 * alt * tan) ** 2 * (1080 / 1920)
            fila.append(k * n / area * huella)
        marca = "  <-- por debajo de 3" if fila[0] < 3 and fila[1] >= 3 else ""
        print(f"  {alt:>6} km | " + " ".join(f"{v:>8.1f}" for v in fila) + marca)
    print()
    print("  Se cruza el umbral de 3 crateres a distinta altura con cada mapa.")
    print("  Esa cascada es el resultado que hay que medir y graficar.")


def configurar_salida(cam):
    sc = bpy.context.scene
    salida = os.path.join(RUTA_PROYECTO, SUBCARPETA, "")
    os.makedirs(salida, exist_ok=True)

    sc.camera = cam                 # <-- el exportador de etiquetas usa esta
    sc.render.resolution_x = 1920
    sc.render.resolution_y = 1080
    sc.render.resolution_percentage = 100
    sc.render.filepath = salida
    sc.render.use_file_extension = True
    sc.frame_step = 1
    print(f"  Salida: {salida}")


def main():
    print("=" * 70)
    print("CAMARA DE DESCENSO")
    print("=" * 70)
    print("PASO 1  Trayectoria")
    cam = animar()

    print("PASO 2  Salida")
    configurar_salida(cam)

    prediccion_landmarks()

    print("=" * 70)
    print("LISTO.")
    print()
    print("Render -> Render Animation (Ctrl+F12)")
    print()
    print("Cuando termine, en blender_exportar_etiquetas.py cambia:")
    print(f"    la carpeta de frames  ->  {SUBCARPETA}")
    print("    dir de labels         ->  labels_descenso")
    print("    trayectoria_real.csv  ->  trayectoria_descenso.csv")
    print("(la camara de la escena ya es CamaraDescenso, no hay que tocar eso)")
    print("=" * 70)

    if RENDERIZAR_AHORA:
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
