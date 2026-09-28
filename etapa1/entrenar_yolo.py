from pathlib import Path
from ultralytics import YOLO
import multiprocessing

def main():
    BASE = Path(__file__).resolve().parent / "crater_detector"
    DATA_YAML = BASE / "data_combined.yaml"   # YAML del conjunto combinado

    print("Usando dataset:", DATA_YAML)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        workers=0,
        project=str(BASE / "runs_crater"),
        name="yolov8_crater",
        save=True
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()