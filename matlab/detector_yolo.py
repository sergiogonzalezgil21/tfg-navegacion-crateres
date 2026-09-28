"""Detector en vivo para las fases 3B y 4.

Modulo pensado para que MATLAB lo llame por la interfaz py.*, un frame cada
vez, mientras se reproduce el video.

El video lo lee Python y no MATLAB: pasar cada frame de 1920x1080 por la
frontera son 6.2 MB por llamada y, con el interprete fuera de proceso, eso se
serializa entero. Python abre el video una vez y lee secuencialmente, y cada
llamada devuelve el numero de frame procesado para que MATLAB compruebe que no
se han desincronizado.

Conviene ejecutarlo fuera de proceso: torch y numpy cargan librerias nativas
que chocan con las que MATLAB ya tiene cargadas, y en modo InProcess eso puede
tirar MATLAB entero sin aviso.

Punto de operacion validado en la fase 2: conf = 0.70 y lote de 1, con recall
0.994 sobre los crateres de escala de mapa. El lote de 1 no es negociable,
porque Ultralytics limita el tiempo de NMS a 2.0 + 0.05*lote segundos y, al
agotarse, descarta en silencio las imagenes que quedaban del lote; con lotes
de 8 se perdia el 40 % de los frames sin ningun error.

Salida: un bloque de bytes con un array float64 en orden C de forma (N, 5) con
cx_px, cy_px, w_px, h_px, conf. Se devuelve como bytes y no como lista porque
cruzar la frontera MATLAB-Python con listas de listas es lento y ambiguo de
tipos.
"""

import numpy as np

_modelo = None
_cap = None
_cfg = {}
_idx = 0


def cargar(pesos, imgsz=1024, conf=0.70, max_det=3000, bw_min=0.03):
    """Carga los pesos una sola vez. Devuelve el dispositivo que usara."""
    global _modelo, _cfg
    from ultralytics import YOLO
    import torch

    _modelo = YOLO(str(pesos))
    _cfg = dict(imgsz=int(imgsz), conf=float(conf),
                max_det=int(max_det), bw_min=float(bw_min))

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    nombre = torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU'
    return f"{dev} ({nombre})"


def calentar(alto=1080, ancho=1920, n=2):
    """Inferencias en vacio para que la primera del directo no salga lenta.

    La primera llamada a CUDA inicializa el contexto y compila kernels: puede
    tardar varios segundos. Si eso pasa con el tribunal delante, parece que
    el sistema se ha colgado.
    """
    if _modelo is None:
        raise RuntimeError("llama antes a cargar()")
    img = np.zeros((int(alto), int(ancho), 3), np.uint8)
    for _ in range(int(n)):
        _modelo.predict(img, imgsz=_cfg["imgsz"], conf=_cfg["conf"],
                        max_det=_cfg["max_det"], verbose=False)
    return "ok"


def abrir_video(ruta):
    """Abre el video y devuelve (n_frames, ancho, alto, fps)."""
    global _cap, _idx
    import cv2
    if _cap is not None:
        _cap.release()
    _cap = cv2.VideoCapture(str(ruta))
    if not _cap.isOpened():
        raise RuntimeError(f"no se pudo abrir el video: {ruta}")
    _idx = 0
    return (int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(_cap.get(cv2.CAP_PROP_FPS)))


def siguiente():
    """Lee el siguiente frame, lo pasa por YOLO y devuelve (n_frame, bytes).

    n_frame = 0 significa que el video se ha acabado.
    """
    global _idx
    if _modelo is None or _cap is None:
        raise RuntimeError("falta cargar() o abrir_video()")

    ok, img = _cap.read()          # OpenCV entrega BGR, que es lo que quiere
    if not ok:                     # Ultralytics: no hay que voltear nada
        return 0, b""
    _idx += 1

    r = _modelo.predict(img, imgsz=_cfg["imgsz"], conf=_cfg["conf"],
                        max_det=_cfg["max_det"], verbose=False)[0]

    if r.boxes is None or len(r.boxes) == 0:
        return _idx, b""

    xywh = r.boxes.xywh.cpu().numpy()
    cf = r.boxes.conf.cpu().numpy()

    # mismo filtro de tamano angular que uso 08_exportar_detecciones.py, para
    # que la demo trabaje con exactamente las mismas cajas que la validacion
    h_img, w_img = img.shape[:2]
    sel = (xywh[:, 2] / w_img) >= _cfg["bw_min"]
    xywh, cf = xywh[sel], cf[sel]

    salida = np.empty((len(cf), 5), np.float64)
    salida[:, :4] = xywh
    salida[:, 4] = cf
    return _idx, salida.tobytes()


def cerrar():
    global _cap
    if _cap is not None:
        _cap.release()
        _cap = None
    return "ok"
