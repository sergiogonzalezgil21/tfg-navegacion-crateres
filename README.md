# Navegación óptica relativa al terreno mediante lectura de cráteres

Trabajo Fin de Grado — Grado en Ingeniería Aeroespacial en Aeronavegación
Escuela de Ingeniería de Fuenlabrada, Universidad Rey Juan Carlos
Curso 2025-2026

**Autor:** Sergio González Gil · **Tutor:** Hodei Urrutxua Cereijo

---

## Qué hace este sistema

Una nave que desciende sobre la Luna guiada solo por su unidad inercial acumula
deriva, y sin un sistema de posicionamiento global que la corrija acaba dentro
de una elipse de dispersión de varios kilómetros. Este trabajo implementa y
valida la alternativa que ha adoptado la industria en su versión más exigente:
**reconocer cráteres concretos en la imagen de una cámara y usarlos como puntos
de referencia contra un mapa embarcado**, igual que un sensor de estrellas
reconoce constelaciones.

El sistema recibe **únicamente píxeles**. No se le da ninguna estimación previa
de posición, de altitud ni de orientación, y de la imagen extrae las cuatro
componentes del vector de estados:

```
x = (x, y, h, ψ)     posición horizontal, altitud y guiñada
```

## Resultados

| Ensayo | Fotogramas resueltos | Error mediano |
|---|---|---|
| 1 · Vuelo de crucero, en diferido | 97,7 % | 0,150 km de posición |
| 2 · Vuelo de crucero, en tiempo real (4,4 Hz) | 97,3 % | 0,146 km |
| 3 · Descenso con la cámara girando ±180° en guiñada | 66,7 % | 0,112° de guiñada · 0,950 km de altitud |

El detector recupera el **99,4 %** de los cráteres visibles en condiciones
nominales. El ensayo 3 deja de funcionar por debajo de unos **118 km** de
altitud, punto en el que el campo de visión ya no abarca suficientes cráteres
del mapa; ese límite se caracteriza de forma cuantitativa en la memoria y se
demuestra que **no se resuelve añadiendo más cráteres al catálogo**.

## Cómo está organizado

El sistema son tres entornos encadenados que se comunican por ficheros.

| Carpeta | Entorno | Qué contiene |
|---|---|---|
| `blender/` | Python + Blender | Generación procedural del terreno lunar (840 × 630 km), montaje de la escena, definición de las trayectorias, renderizado de las secuencias y exportación de etiquetas y verdad de referencia |
| `etapa1/` | Python | Los cuatro scripts del primer entrenamiento del detector sobre imágenes lunares reales, de una fase temprana del trabajo |
| `ia/` | Python + Ultralytics | Preparación del conjunto de datos, entrenamiento en tres etapas, evaluación del detector y exportación de las detecciones |
| `matlab/` | MATLAB | Motor de emparejamiento geométrico, los tres ensayos de navegación y la generación de las figuras de la memoria |

Cada carpeta lleva su propio `LEEME.md` con el detalle. El **Anexo D** de la
memoria indica el orden exacto de ejecución para reconstruir todos los
resultados desde cero.

## Reproducir los resultados sin rehacer nada

El repositorio incluye los pocos ficheros de datos que hacen falta para que el
motor de navegación se pueda ejecutar tal cual: los mapas embarcados, las
trayectorias verdaderas y las detecciones ya exportadas. Suman 1,7 MB.

Con ellos, **en MATLAB y sin nada más**:

```matlab
fase3a_offline      % ensayo 1: 97,7 % de fotogramas, 0,150 km de error
fase4_cascada       % el muro de los 118 km, con los tres mapas de densidad
```

No hacen falta Blender, ni GPU, ni `ultralytics`: leen las detecciones del CSV
y resuelven la geometría. Los dos ensayos en tiempo real
(`fase3b_directo`, `fase4_directo`) sí necesitan el vídeo y el detector, y
`fase3a_trayectoria_3d` necesita el mapa de alturas; todo eso se reconstruye
ejecutando la cadena completa.

## La cadena completa

Todo el sistema es determinista: la semilla del generador de terreno es
`20260818` y reproducirla da el mismo mapa, los mismos 41 702 cráteres y las
mismas cifras que aparecen en la memoria.

```bash
# 1. Entorno
pip install numpy pillow ultralytics opencv-python

# 2. Terreno y renders (dentro de Blender)
#    blender/blender_construir_terreno.py
#    blender/blender_preparar_vuelo.py
#    blender/blender_exportar_etiquetas.py

# 3. Detector
python ia/02_preparar_dataset.py
python ia/03_entrenar.py
python ia/08_exportar_detecciones.py

# 4. Navegación (en MATLAB)
#    matlab/fase3a_offline.m
#    matlab/fase3b_directo.m
#    matlab/fase4_directo.m
```

## Rutas y portabilidad

No hay ninguna ruta absoluta en el código. Cada script resuelve dónde está a
partir de su propia ubicación, así que el repositorio funciona clonado en
cualquier carpeta. Solo hay dos cosas configurables:

| Variable | Dónde | Para qué |
|---|---|---|
| `PYTHON_EXE` | `matlab/fase3b_directo.m` y `fase4_directo.m` | Intérprete de Python con `ultralytics`. Vacío por defecto: se usa el que MATLAB ya tenga configurado (`pyenv`) |
| `TFG_PESOS_ETAPA1` | variable de entorno | Carpeta con los pesos del primer entrenamiento. Si no se define, los scripts caen a los pesos públicos de Ultralytics |

Los conjuntos de datos de la etapa 1 se descargan con
`etapa1/descargar_datasets.py`, que usa `kagglehub` y los deja en su caché
(`~/.cache/kagglehub`), que es donde los buscan los demás scripts de esa etapa.

## Lo que no está en el repositorio

Este repositorio es **el código**. Todo lo que sale de ejecutarlo se ha dejado
fuera, que son unos 1,6 GB:

- La escena y las mallas: `terreno_lunar.blend` (258 MB), `terreno_lunar.stl`
  (674 MB), la caché del relieve `H_8192.npy` (201 MB)
- Los mapas de altura, detalle y albedo a resolución completa (`detalle.png`,
  90 MB) y el catálogo de los 41 702 cráteres
- Las secuencias renderizadas (`Output/`, `Output_descenso/`), los vídeos y las
  etiquetas YOLO de los 1400 fotogramas
- El conjunto de entrenamiento y las salidas de Ultralytics
- Las tablas de tripletas precalculadas y las figuras de la memoria

Tampoco están los **pesos entrenados**. Regenerarlos exige los 700 renders, que
tampoco están, así que incluirlos no ahorraría ningún paso: quien quiera
reproducir el detector ejecuta la cadena desde Blender, y quien solo quiera ver
funcionar la navegación tiene las detecciones ya exportadas.

La memoria se entrega aparte, depositada en la biblioteca de la universidad.
