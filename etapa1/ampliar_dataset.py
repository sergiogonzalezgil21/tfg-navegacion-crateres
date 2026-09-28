from pathlib import Path
import shutil

# Los conjuntos se descargan con descargar_datasets.py, que usa kagglehub y
# los deja en su cache. Si estan en otro sitio, basta con cambiar CACHE_KAGGLE.
CACHE_KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets"
SRC_BASE = (CACHE_KAGGLE / "riccardolagrassa" / "lu3m6tgt" /
            "versions" / "1" / "LU3M6TGT_yolo_format")
DST_BASE = Path(__file__).resolve().parent / "crater_detector" / "craters_combined"

SPLITS = ["train", "val"]

def copy_split(split):
    src_images = SRC_BASE / split / "images"
    src_labels = SRC_BASE / split / "labels"
    dst_images = DST_BASE / split / "images"
    dst_labels = DST_BASE / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img in src_images.rglob("*"):
        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            label = src_labels / f"{img.stem}.txt"
            if label.exists():
                new_img = f"kaggle_{img.name}"
                new_label = f"kaggle_{img.stem}.txt"
                shutil.copy2(img, dst_images / new_img)
                shutil.copy2(label, dst_labels / new_label)
                copied += 1
    print(f"{split}: copiados {copied} archivos")


def main():
    for split in SPLITS:
        copy_split(split)

    print("Listo.")
    print(f"Destino: {DST_BASE}")

if __name__ == "__main__":
    main()