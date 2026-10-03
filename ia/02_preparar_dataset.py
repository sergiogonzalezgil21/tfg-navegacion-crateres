"""
Paso 2. Prepara el conjunto de Blender en formato YOLO.

El split no puede ser aleatorio. Los 700 frames son una sola trayectoria: la
camara avanza 1.8 km entre frames y ve una huella de 150 a 196 km, asi que dos
frames consecutivos comparten mas del 98 % del terreno. Repartirlos al azar
dejaria el frame 300 en train y el 301 en val, y el mAP mediria memorizacion.

Para que train y val no compartan nada, se separan por un hueco de huella/paso
frames, tomando la huella MAXIMA de la pasada para que la separacion valga
tambien en el tramo mas alto: con este vuelo salen 109 frames, que se descartan.
El hueco se calcula a partir de trayectoria_real.csv, no a ojo.

Produce dataset_blender/{train,val}/{images,labels} y data.yaml.
"""

import os
import csv
import math
import shutil

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

DIR_FRAMES = os.path.join(RUTA_BLENDER, "Output")
DIR_LABELS = os.path.join(RUTA_BLENDER, "labels")
CSV_TRAY = os.path.join(RUTA_BLENDER, "trayectoria_real.csv")
DIR_DATASET = os.path.join(RUTA_IA, "dataset_blender")

FRACCION_TRAIN = 0.62        # el resto va a val, menos el hueco
FOV_HORIZONTAL = 60.0
COPIAR = True                # False = enlaces simbolicos (necesita permisos)

# =============================================================================


def leer_trayectoria():
    filas = []
    with open(CSV_TRAY, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            filas.append((int(r["frame"]), float(r["x_km"]),
                          float(r["y_km"]), float(r["altura_km"])))
    return sorted(filas)


def calcular_hueco(filas):
    """Frames que hay que descartar entre train y val para que no compartan
    terreno. Sale de la geometria, no de una regla del pulgar."""
    pasos = [math.hypot(filas[i+1][1] - filas[i][1], filas[i+1][2] - filas[i][2])
             for i in range(len(filas) - 1)]
    paso = sum(pasos) / max(len(pasos), 1)
    huella = 2.0 * max(f[3] for f in filas) * math.tan(math.radians(FOV_HORIZONTAL / 2))
    hueco = int(math.ceil(huella / max(paso, 1e-6)))
    print(f"  paso medio entre frames : {paso*1000:.0f} m")
    print(f"  huella maxima           : {huella:.0f} km")
    print(f"  hueco necesario         : {hueco} frames  ({hueco*paso:.0f} km)")
    return hueco


def preparar_dir(base):
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = os.path.join(base, split, sub)
            os.makedirs(d, exist_ok=True)
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))


def colocar(frame, split):
    nombre = f"{frame:04d}"
    img_o = os.path.join(DIR_FRAMES, nombre + ".png")
    txt_o = os.path.join(DIR_LABELS, nombre + ".txt")
    if not (os.path.isfile(img_o) and os.path.isfile(txt_o)):
        return False
    img_d = os.path.join(DIR_DATASET, split, "images", nombre + ".png")
    txt_d = os.path.join(DIR_DATASET, split, "labels", nombre + ".txt")
    if COPIAR:
        shutil.copy2(img_o, img_d)
    else:
        os.symlink(img_o, img_d)
    shutil.copy2(txt_o, txt_d)
    return True


def main():
    print("=" * 70)
    filas = leer_trayectoria()
    n = len(filas)
    print(f"Trayectoria: {n} frames")

    hueco = calcular_hueco(filas)

    n_train = int(n * FRACCION_TRAIN)
    ini_val = n_train + hueco
    if ini_val >= n:
        raise RuntimeError(
            f"Con {n} frames y un hueco de {hueco} no queda sitio para val. "
            f"Baja FRACCION_TRAIN o alarga la trayectoria.")

    frames_train = [filas[i][0] for i in range(n_train)]
    frames_val = [filas[i][0] for i in range(ini_val, n)]
    descartados = n - len(frames_train) - len(frames_val)

    print(f"\n  train : frames {frames_train[0]}-{frames_train[-1]}  "
          f"({len(frames_train)})")
    print(f"  hueco : {descartados} frames descartados")
    print(f"  val   : frames {frames_val[0]}-{frames_val[-1]}  "
          f"({len(frames_val)})")

    # separacion real en el suelo entre el ultimo de train y el primero de val
    ft = filas[n_train - 1]; fv = filas[ini_val]
    d = math.hypot(fv[1] - ft[1], fv[2] - ft[2])
    print(f"  separacion en el suelo entre train y val: {d:.0f} km")

    preparar_dir(DIR_DATASET)
    ok_t = sum(colocar(f, "train") for f in frames_train)
    ok_v = sum(colocar(f, "val") for f in frames_val)
    print(f"\n  copiados: {ok_t} train, {ok_v} val")
    if ok_t < len(frames_train) or ok_v < len(frames_val):
        print("  AVISO: faltan frames. Puede que el render no haya terminado.")

    yaml = os.path.join(DIR_DATASET, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {DIR_DATASET}\n")
        f.write("train: train/images\n")
        f.write("val: val/images\n")
        f.write("nc: 1\n")
        f.write("names: ['crater']\n")
    print(f"  {yaml}")
    print("=" * 70)


if __name__ == "__main__":
    main()
