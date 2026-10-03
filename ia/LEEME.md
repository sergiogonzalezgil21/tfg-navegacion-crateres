# Fase 2 — El detector

Doce pasos, en orden. Cada uno comprueba lo que hizo el anterior, y varios
existen porque el anterior dio un resultado que no podía ser cierto.

```
01_verificar_etiquetas.py   mide que las cajas caen sobre los cráteres
02_preparar_dataset.py      split train/val con hueco geométrico
03_entrenar.py              ajuste fino sobre los renders de Blender
04_evaluar.py               modelo de partida contra afinado, mismo val
05_metrica_navegacion.py    cráteres de escala de mapa por frame
06_diagnostico_deteccion.py por qué bajar conf reducía las detecciones
07_deteccion_limpia.py      la medida buena, sin el recorte del NMS
08_exportar_detecciones.py  detecciones en píxeles para MATLAB
09_barrido_conjunto.py      elección conjunta de D_mapa y K
10_robustez_rotacion.py     recall frente al giro de la cámara
11_detecciones_descenso.py  lo mismo que 08, sobre el vuelo de descenso
12_reentrenar_rotacion.py   reentrenamiento con aumentación por rotación
```

Requisito: el render de los frames tiene que haber terminado y
`blender_exportar_etiquetas.py` haber generado `labels/` y
`trayectoria_real.csv`.

---

## 01 — Verificar las etiquetas

Dibuja las cajas sobre una muestra de frames y **mide** la alineación:
compara la textura dentro de las cajas con la de cajas colocadas al azar. Un
cráter con sol rasante tiene media sombra y media pared iluminada, así que su
desviación típica interna es alta; el terreno llano es uniforme. Si la
relación baja de 1.15, las cajas están descolocadas.

Esto existe porque un desplazamiento sistemático de las etiquetas **no da
ningún error**: la red entrena igual, la pérdida baja igual, y lo que aprende
es a detectar cráteres corridos.

## 02 — El split, que no es obvio

Los 700 frames son una trayectoria continua: la cámara avanza 1.8 km entre
frames y ve una huella de 150–196 km, así que dos frames consecutivos comparten
más del 98 % del terreno. Un split aleatorio pondría el frame 300 en train y el
301 en val: son casi la misma imagen, y la validación daría un mAP altísimo y
falso, midiendo memorización en vez de generalización.

El script calcula el hueco necesario como huella/paso a partir de
`trayectoria_real.csv`, no a ojo, y descarta esos frames. Usa la huella máxima
de la pasada, 196 km, para que la separación valga también en el tramo más alto:
con este vuelo salen 109 frames, y quedan 434 de entrenamiento y 157 de
validación, separados por 200 km de terreno. Merece la pena: es la diferencia
entre una métrica real y una inventada.

## 03 — Entrenar

Ajuste fino, no entrenamiento desde cero: se parte de un modelo que ya
reconoce cráteres en imágenes reales de la Luna y de Marte, y lo que se le
enseña es el dominio nuevo. 60 épocas, `imgsz = 1024` — a 640 los cráteres
pequeños se quedan en 4 px y son indetectables.

**Volteos y rotaciones desactivados a propósito.** En un cráter con sol
rasante la sombra siempre cae al mismo lado, y esa asimetría es justo la señal
que usa el detector: voltear la imagen crea una iluminación que no existe en
ningún frame. Esa decisión se revisa en el paso 12, cuando la cámara empieza a
girar.

## 04 y 05 — Qué métrica decide

El mAP es lo estándar y hay que darlo, pero no es lo que decide aquí: lo que
importa es **cuántos cráteres de escala de mapa** hay en cada frame, porque el
emparejamiento por tripletas necesita tres y la verificación un cuarto.

El porcentaje de frames con ≥3 cráteres *cualesquiera* se satura al 100 %: se
etiquetan todos los visibles, unos 423 por frame y la mayoría de menos de 2 km,
y encontrar 3 de 423 lo consigue cualquiera. El paso 05 mide solo los que el
mapa embarcado lleva de verdad.

## 06 y 07 — Un resultado imposible

El modelo afinado detectaba **menos** cráteres de escala de mapa al **bajar**
el umbral de confianza. Bajar `conf` solo puede añadir cajas, así que ahí había
un artefacto y no un resultado.

Son dos efectos encadenados. El cupo de `max_det`: Ultralytics ordena por
confianza y se queda con las primeras cajas, de modo que al bajar `conf` entran
miles de cajas pequeñas y las pocas grandes se caen del corte. Y, sobre todo,
el presupuesto de tiempo del NMS, que es `2.0 + 0.05·lote` segundos: al
agotarse, las imágenes que quedaban del lote se devuelven **vacías, sin lanzar
ningún error**. Con lotes de 8 se perdía casi la mitad de los frames.

De ahí la regla que arrastra todo lo demás: **inferir de un frame en uno**.

## 08 y 09 — Punto de operación

`conf = 0.70`, lote de 1. Mismo recall sobre los cráteres de escala de mapa
(0.994) y la mitad de cajas basura, porque el detector ve los grandes con
confianza media 0.838 frente a 0.509 los pequeños.

La salida para MATLAB va **en píxeles**, sin kilómetros ni altitud: la altitud
es justo lo que el navegador tiene que estimar, y dársela hecha invalidaría la
validación. Los landmarks se eligen de forma invariante a escala, como las K
cajas más grandes del frame.

El umbral del mapa `D_mapa` y ese K se barren **a la vez**, porque están
acoplados: medir la precisión de «las K más grandes» contra un mapa de
D ≥ 16 km cuenta como fallo toda caja de 8 a 16 km, que con un mapa de
D ≥ 10 km serían aciertos.

## 10 y 12 — La cámara girada

En la fase 4 la cámara gira alrededor de su eje óptico. El navegador es
invariante a rotación; el detector, entrenado con `degrees = 0.0`, no: el sol
está fijo en el mundo, así que al girar la cámara las sombras cambian de lado
dentro de la imagen y un cráter con la sombra al revés parece un montículo.

No hace falta re-renderizar para medirlo. Para una cámara nadir sobre un plano,
girarla alrededor de su eje óptico da exactamente la imagen original rotada, así
que basta girar los frames que ya hay y medir dentro del círculo inscrito.

El paso 12 reentrena con `degrees = 180` sobre los **mismos datos** del vuelo
de crucero, de modo que el descenso sigue siendo prueba ajena y la mejora es
atribuible solo a la aumentación.
