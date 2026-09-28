# MATLAB — navegación por cráteres

Todo vive en esta carpeta, sin subcarpetas por fase. Los ficheros que empiezan
por `fase3a_`, `fase3b_` o `fase4_` son los **experimentos** de cada bloque del
TFG; los que no llevan prefijo son el **motor**, compartido por todos.

## Motor (compartido, sin prefijo)

| Fichero | Qué hace |
|---|---|
| `cargar_mapa.m` | Lee el mapa embarcado y construye la tabla de tripletas con el descriptor 4-D invariante a escala y rotación. Poda por "cabe en un frame" (`lado_max_km`) y, opcionalmente, poda acoplada a la escala (`factor_lado`). |
| `buscar_tripletas.m` | Búsqueda binaria en la tabla ordenada. Sin toolboxes. |
| `leer_detecciones.m` | Lee `detecciones.csv` (píxeles, sin km ni altitud: la altitud es la incógnita). |
| `resolver_frame.m` | El navegador. De las detecciones de un frame saca posición, guiñada y altitud, con verificación por cercanía y por tamaño. **Es una función, no se llama por su nombre.** |
| `detector_yolo.py` | Módulo Python que MATLAB llama vía `pyenv` para correr YOLO en directo. |

## Fase 3A — navegación offline

| Fichero | Qué hace |
|---|---|
| `fase3a_offline.m` | Resuelve los 700 frames ya detectados, desde `detecciones.csv`. Resultado: 684/700, mediana 0.150 km. |
| `fase3a_trayectoria_3d.m` | Pinta la trayectoria real y la estimada en 3D sobre el terreno con su textura. Correr después de `fase3a_offline`. |
| `fase3a_resultados.csv` | Salida de `fase3a_offline`. |

## Fase 3B — navegación en directo

| Fichero | Qué hace |
|---|---|
| `fase3b_crear_video.m` | Monta los 700 PNG de `blender\Output` en `blender\vuelo.mp4`. Se corre una sola vez. |
| `fase3b_directo.m` | La demo: lee el vídeo, detecta con YOLO en vivo y va pintando las gráficas frame a frame. ~4.4 fps. |
| `fase3b_resultados.csv` | Salida de `fase3b_directo`. |

## Fase 4 — descenso con rotación en Z

Cámara nueva (`CamaraDescenso`): media parábola de 170 a 55 km que primero gira
dos ciclos de ±180° en guiñada y luego se estabiliza para aterrizar. Baja hasta
que el sistema falla, a propósito, para poder documentar dónde está el muro.

| Fichero | Qué hace |
|---|---|
| `fase4_crear_video.m` | Monta los 700 PNG de `blender\Output_descenso` en `blender\vuelo_descenso.mp4`. |
| `fase4_cascada.m` | Barrido con tres mapas de distinta densidad (D≥10, D≥6, D≥4 km) para ver cómo cae la cobertura con la altitud. Los tres colapsan hacia los 118 km. |
| `fase4_directo.m` | La demo en directo del descenso, al estilo de 3B: vídeo con las cajas dibujadas, mapa, altitud, guiñada real vs estimada y error. |
| `fase4_cascada.csv`, `fase4_directo_resultados.csv` | Salidas. |
| `trip_D10_f30.mat`, `trip_D6_f30.mat`, `trip_D4_f30.mat` | Tablas de tripletas precalculadas de cada mapa. |

Resultado de `fase4_directo`: 467/700 con solución (66.7 %), de las cuales 462
correctas (98.9 %, error < 5 km), mediana 0.313 km, primer fallo a 120.8 km de
altitud, a 6.8 fps.

## Orden en que se corre

```
Fase 3A:  fase3a_offline  ->  fase3a_trayectoria_3d
Fase 3B:  fase3b_crear_video  ->  fase3b_directo
Fase 4:   fase4_crear_video  ->  fase4_cascada  y  fase4_directo
```

`fase3b_directo` y `fase4_directo` usan el intérprete que tenga configurado
`pyenv`. Si el de por defecto no tiene `ultralytics` instalado, hay que fijar
`PYTHON_EXE` en la cabecera de configuración de esos dos scripts.
