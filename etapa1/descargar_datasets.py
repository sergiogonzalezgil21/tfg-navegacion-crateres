import kagglehub
from pathlib import Path

DATASETS = [
    "lincolnzh/martianlunar-crater-detection-dataset",
    "bhargavlc/lunar-craters-and-boulders",
]

download_paths = []

for ds in DATASETS:
    print(f"\nDescargando: {ds}")
    path = kagglehub.dataset_download(ds)
    print(f"Guardado en: {path}")
    download_paths.append(path)

print("\nResumen:")
for p in download_paths:
    print(p)

out_file = Path("download_paths.txt")
out_file.write_text("\n".join(download_paths), encoding="utf-8")
print(f"\nRutas guardadas en: {out_file.resolve()}")