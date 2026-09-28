from pathlib import Path
import shutil
import random

# Todo cuelga de la carpeta de este fichero: el repositorio es portable.
BASE = Path(__file__).resolve().parent / "crater_detector"
OUT = BASE / "dataset_crater"

# Los conjuntos se descargan con descargar_datasets.py, que usa kagglehub y
# los deja en su cache. Si estan en otro sitio, basta con cambiar CACHE_KAGGLE.
CACHE_KAGGLE = Path.home() / ".cache" / "kagglehub" / "datasets"
DATASET1 = (CACHE_KAGGLE / "lincolnzh" /
            "martianlunar-crater-detection-dataset" / "versions" / "2" / "craters")
DATASET2 = (CACHE_KAGGLE / "bhargavlc" / "lunar-craters-and-boulders" /
            "versions" / "1" / "lunardataset" / "dataset" / "crater")

IMG_EXTS = {".jpg", ".jpeg", ".png"}

def clear_output():
    if OUT.exists():
        shutil.rmtree(OUT)

def ensure_dirs():
    for split in ["train", "val", "test"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

def collect_from_split_dataset(root):
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = root / split / "images"
        lbl_dir = root / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img in img_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMG_EXTS:
                lbl = lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    pairs.append((img, lbl))
    return pairs

def collect_from_flat_dataset(root):
    pairs = []

    if not root.exists():
        print(f"[AVISO] No existe la ruta del dataset 2: {root}")
        return pairs

    for img in root.iterdir():
        if img.is_file() and img.suffix.lower() in IMG_EXTS:
            lbl = root / f"{img.stem}.txt"
            if lbl.exists():
                pairs.append((img, lbl))

    return pairs

def copy_split(items, split):
    for i, (img, lbl) in enumerate(items):
        img_name = f"{split}_{i:05d}{img.suffix.lower()}"
        lbl_name = f"{split}_{i:05d}.txt"

        shutil.copy2(img, OUT / "images" / split / img_name)
        shutil.copy2(lbl, OUT / "labels" / split / lbl_name)

def write_yaml():
    yaml_text = f"""path: {OUT.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: crater
"""
    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

def main():
    print(f"Carpeta base del proyecto: {BASE}")
    print(f"Salida del dataset final:   {OUT}")

    clear_output()
    ensure_dirs()

    print(f"\nLeyendo dataset 1: {DATASET1}")
    pairs1 = collect_from_split_dataset(DATASET1)
    print(f"Pares encontrados en dataset 1: {len(pairs1)}")

    print(f"\nLeyendo dataset 2: {DATASET2}")
    pairs2 = collect_from_flat_dataset(DATASET2)
    print(f"Pares encontrados en dataset 2: {len(pairs2)}")

    all_pairs = pairs1 + pairs2

    if not all_pairs:
        print("\nNo se encontraron pares imagen + txt.")
        print("El dataset 1 debería tener estructura train/valid/test con images y labels.")
        print("El dataset 2 solo se usará si existen archivos .txt junto a las imágenes.")
        return

    random.seed(42)
    random.shuffle(all_pairs)

    n = len(all_pairs)
    n_train = int(n * 0.7)
    n_val = int(n * 0.2)

    train = all_pairs[:n_train]
    val = all_pairs[n_train:n_train + n_val]
    test = all_pairs[n_train + n_val:]

    copy_split(train, "train")
    copy_split(val, "val")
    copy_split(test, "test")

    write_yaml()

    print("\nDataset preparado correctamente.")
    print(f"Train: {len(train)}")
    print(f"Val:   {len(val)}")
    print(f"Test:  {len(test)}")
    print(f"Total: {len(all_pairs)}")
    print(f"YAML:  {OUT / 'data.yaml'}")

if __name__ == "__main__":
    main()