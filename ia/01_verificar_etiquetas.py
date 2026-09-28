"""
Paso 1. Comprueba las etiquetas antes de entrenar.

Las etiquetas salen de proyectar el catalogo por la camara de Blender. Si la
matriz, el FOV o el convenio de ejes tuvieran el minimo error, las cajas
saldrian desplazadas de forma sistematica, y un desplazamiento sistematico no
da ningun error: la red entrena igual, la perdida baja igual, y lo que aprende
es a detectar crateres corridos. Solo se ve mirando, asi que aqui se mira y se
mide.

Produce verificacion/frame_XXXX.png con las cajas dibujadas, y por consola una
comprobacion numerica de la alineacion.
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw

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
DIR_SALIDA = os.path.join(RUTA_IA, "verificacion")

N_MUESTRAS = 8          # cuantos frames dibujar
SEMILLA = 7

# =============================================================================


def leer_labels(path):
    cajas = []
    if not os.path.isfile(path):
        return cajas
    with open(path, encoding="utf-8") as f:
        for linea in f:
            p = linea.split()
            if len(p) >= 5:
                cajas.append([float(v) for v in p[1:5]])
    return cajas


def emparejar():
    """Empareja cada frame con su txt. El nombre debe coincidir."""
    if not os.path.isdir(DIR_FRAMES):
        raise RuntimeError(f"No existe {DIR_FRAMES}")
    pares = []
    for nombre in sorted(os.listdir(DIR_FRAMES)):
        if not nombre.lower().endswith(".png"):
            continue
        base = os.path.splitext(nombre)[0]
        txt = os.path.join(DIR_LABELS, base + ".txt")
        if os.path.isfile(txt):
            pares.append((os.path.join(DIR_FRAMES, nombre), txt))
    return pares


def contraste_en_cajas(img_gris, cajas, rng):
    """Compara la textura DENTRO de las cajas con la de cajas colocadas al azar.

    Un crater visto con sol rasante tiene media sombra y media pared
    iluminada, asi que su desviacion tipica interna es alta. El terreno
    llano de alrededor es mucho mas uniforme. Si las etiquetas estuvieran
    descolocadas, esta relacion se acercaria a 1.
    """
    H, W = img_gris.shape
    dentro, azar = [], []

    for (cx, cy, bw, bh) in cajas:
        x0 = int((cx - bw / 2) * W); x1 = int((cx + bw / 2) * W)
        y0 = int((cy - bh / 2) * H); y1 = int((cy + bh / 2) * H)
        x0, x1 = max(0, x0), min(W, x1)
        y0, y1 = max(0, y0), min(H, y1)
        if x1 - x0 < 6 or y1 - y0 < 6:
            continue
        dentro.append(img_gris[y0:y1, x0:x1].std())

        aw, ah = x1 - x0, y1 - y0
        ax = rng.randint(0, max(0, W - aw)); ay = rng.randint(0, max(0, H - ah))
        azar.append(img_gris[ay:ay + ah, ax:ax + aw].std())

    if not dentro:
        return None
    return float(np.mean(dentro)), float(np.mean(azar))


def main():
    os.makedirs(DIR_SALIDA, exist_ok=True)
    pares = emparejar()
    print(f"Frames con etiqueta: {len(pares)}")
    if not pares:
        raise RuntimeError(
            "No hay pares imagen/etiqueta. Comprueba que el render ha "
            "terminado y que blender_exportar_etiquetas.py se ha ejecutado.")

    n_cajas = [len(leer_labels(t)) for _, t in pares]
    print(f"Cajas por frame: media {np.mean(n_cajas):.1f}  "
          f"mediana {np.median(n_cajas):.0f}  min {min(n_cajas)}  max {max(n_cajas)}")
    pobres = sum(1 for n in n_cajas if n < 3)
    print(f"Frames con menos de 3 crateres: {pobres} "
          f"({100*pobres/len(n_cajas):.1f} %)")

    # distribucion de tamanos: es LA razon por la que el detector actual falla
    anchos = []
    for _, t in pares:
        anchos += [c[2] for c in leer_labels(t)]
    anchos = np.array(anchos)
    if len(anchos):
        print(f"\nAnchura normalizada de las cajas:")
        for p in (5, 25, 50, 75, 95):
            print(f"   p{p:<3d} {np.percentile(anchos, p):.4f}")
        print(f"   en la banda 0.10-0.25: {100*((anchos>0.10)&(anchos<0.25)).mean():.1f} %")
        print("   (el dataset de internet apenas tenia ejemplos en esa banda,")
        print("    y es justo donde caen los crateres de referencia)")

    rng = random.Random(SEMILLA)
    idx = sorted(rng.sample(range(len(pares)), min(N_MUESTRAS, len(pares))))

    ratios = []
    for k in idx:
        fimg, ftxt = pares[k]
        im = Image.open(fimg).convert("RGB")
        W, H = im.size
        cajas = leer_labels(ftxt)

        gris = np.asarray(im.convert("L")).astype(np.float32) / 255.0
        r = contraste_en_cajas(gris, cajas, rng)
        if r:
            ratios.append(r[0] / max(r[1], 1e-6))

        d = ImageDraw.Draw(im)
        for (cx, cy, bw, bh) in cajas:
            x0 = (cx - bw / 2) * W; x1 = (cx + bw / 2) * W
            y0 = (cy - bh / 2) * H; y1 = (cy + bh / 2) * H
            d.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
            d.line([(cx * W - 8, cy * H), (cx * W + 8, cy * H)], fill=(255, 0, 0), width=2)
            d.line([(cx * W, cy * H - 8), (cx * W, cy * H + 8)], fill=(255, 0, 0), width=2)

        base = os.path.splitext(os.path.basename(fimg))[0]
        salida = os.path.join(DIR_SALIDA, f"frame_{base}.png")
        im.save(salida)
        print(f"  {base}: {len(cajas)} cajas -> {os.path.basename(salida)}")

    if ratios:
        m = float(np.mean(ratios))
        print(f"\nTextura dentro de las cajas / textura en cajas al azar: {m:.2f}")
        if m > 1.5:
            print("   Las cajas caen sobre estructura real. Alineacion correcta.")
        elif m > 1.15:
            print("   Aceptable, pero conviene revisar las imagenes antes de entrenar.")
        else:
            print("   AVISO: las cajas no distinguen crater de terreno llano.")
            print("   Probablemente esten descolocadas. NO entrenes todavia.")

    print(f"\nImagenes anotadas en: {DIR_SALIDA}")
    print("Las cajas verdes deben rodear los crateres, con la cruz roja")
    print("en el centro. Si estan corridas de forma sistematica, avisa.")


if __name__ == "__main__":
    main()
