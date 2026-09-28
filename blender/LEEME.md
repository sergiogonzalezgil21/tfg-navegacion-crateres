# Fase 1 — Terreno y vuelos en Blender

Genera la superficie lunar sintética, las dos cámaras y la verdad de
referencia (etiquetas YOLO, trayectoria y mapa embarcado) con la que se validan
las fases siguientes.

## Archivos

| Archivo | Qué es |
|---|---|
| `terreno_lunar.py` | Generador del heightmap y del albedo, en numpy puro. No se ejecuta solo: lo importa el siguiente |
| `blender_construir_terreno.py` | Monta la escena entera. Es el que se ejecuta primero |
| `blender_preparar_vuelo.py` | Vuelo de crucero: renderiza el mapa de referencia y anima `CamaraNave` |
| `blender_exportar_etiquetas.py` | Etiquetas YOLO de cada frame, `trayectoria_real.csv` y `mapa_referencia.csv` |
| `blender_camara_descenso.py` | Añade `CamaraDescenso`: descenso de 170 a 55 km con giro en guiñada |
| `blender_preparar_vuelo_f4.py` | Variante del anterior para la fase 4 |
| `blender_exportar_descenso.py` | Lo mismo que `blender_exportar_etiquetas.py`, sobre el vuelo de descenso |
| `blender_fig_trayectoria3d.py` | Figura 3D de la trayectoria sobre el terreno, para la memoria |
| `preparar_impresion_3d.py` | Maqueta imprimible en 3D del terreno, 1:1 000 000, en 12 piezas |
| `comprobar_piezas.py` | Verifica que esas 12 piezas son estancas y encajan entre sí |

Los `.py` de Blender tienen que quedarse juntos en esta carpeta:
`blender_construir_terreno.py` importa `terreno_lunar.py` desde aquí, y todos
resuelven sus rutas a partir de la del propio fichero.

## Orden de ejecución

```
1. blender_construir_terreno.py     escena, texturas y catálogo de cráteres
2. blender_preparar_vuelo.py        mapa de referencia + CamaraNave animada
3. (render de CamaraNave)           700 PNG en Output/
4. blender_exportar_etiquetas.py    labels/, trayectoria_real.csv, mapa_referencia.csv

fase 4:
5. blender_camara_descenso.py       CamaraDescenso
6. blender_preparar_vuelo_f4.py     ajustes de render del descenso
7. (render de CamaraDescenso)       700 PNG en Output_descenso/
8. blender_exportar_descenso.py     labels_descenso/, trayectoria_descenso.csv
```

En Blender: pestaña **Scripting** → **Open** → **Run Script**. El progreso sale
por la consola del sistema (`Window → Toggle System Console`), y además queda
escrito en `registro.txt` junto al script.

Salidas de `blender_construir_terreno.py`:

```
altura.png              heightmap de 16 bits (solo documentación)
detalle.png             residuo de alta frecuencia, para el bump
albedo.png              mapa de albedo
crateres_catalogo.csv   41 702 cráteres: posición, diámetro, profundidad, frescura
mapa_landmarks.csv      los 180 cráteres de referencia, en km
```

## Las dos cámaras del mapa

- `CamaraNave` — perspectiva, FOV 60°, 1920×1080. La del satélite.
- `CamaraMapa` — **ortográfica**, `ortho_scale = 840`. Encuadra exactamente los
  840 × 630 km, que son 4:3.

El encuadre ortográfico y exacto importa: si el render del mapa saliera en 16:9
quedarían franjas vacías a los lados, y la escala en kilómetros que se deduce
del ancho de la imagen saldría corta. Con este encuadre
`km_por_pixel = 840/ancho = 630/alto` es correcto e isótropo.

## Qué modela el terreno

- Distribución de tamaños en ley de potencias, N(>D) ~ D^-2.1, 400 000 cráteres
  de fondo sobre 529 200 km²; entran al catálogo los de D ≥ 0.8 km
- Cráteres **simples** (cuenco parabólico) frente a **complejos** (fondo plano,
  terrazas, pico central), con la transición en 18 km como en la Luna
- Profundidades según **Pike (1977)**: `d = 0.196·D^1.010` (simples),
  `d = 1.044·D^0.301` (complejos)
- Borde elevado (~3.2 % del diámetro) y manto de eyecta que decae como r^-3.8
- Degradación por edad: los viejos son someros y romos, los jóvenes se
  superponen encima
- Mares basálticos con orilla definida y poco craterados, con crestas de arruga
- Rugosidad de regolito fBm multiescala
- Albedo: mares oscuros, halos y rayos de eyecta fresca

Los 180 cráteres landmark se reparten por todo el terreno con una separación
mínima de 34 km, y son los más frescos y nítidos del mapa para que destaquen
sobre el fondo.

## Parámetros

Están todos en la cabecera de `blender_construir_terreno.py`:

| Parámetro | Para qué |
|---|---|
| `SEED` | Otra semilla da un terreno distinto con los mismos landmarks |
| `SOL_ELEVACION` | 28° por defecto. Más bajo = sombras más largas y cráteres más legibles |
| `RES_HEIGHTMAP` | 8192 = 102 m/píxel. Con poca RAM se puede bajar a 4096 |
| `SUBDIV_X/Y` | 3000×2250 = 6.75 M vértices. Bajarlo si Blender va lento |
| `ALTURA_CAMARA_KM` | 120 km |

## Unidades

**1 unidad de Blender = 1 km.** A propósito: trabajar en metros con un terreno
de 840 km y la cámara a 120 000 m mete números de seis cifras en el z-buffer y
aparecen artefactos de precisión (rejilla negra, picos). En kilómetros todo
cabe entre 0 y 900 unidades y desaparecen.
