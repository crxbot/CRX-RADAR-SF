import os
import sys
import json
from datetime import datetime

png_root = sys.argv[1]  # Ordner mit dem Radarbild
date = sys.argv[2] if len(sys.argv) > 2 else datetime.utcnow().strftime("%Y%m%d")

# Annahme: im Ordner liegt mindestens ein PNG
png_files = [f for f in os.listdir(png_root) if f.endswith(".png")]
if not png_files:
    raise FileNotFoundError(f"No PNG files found in {png_root}")

# Die Datei nehmen (falls mehrere, die neueste alphabetisch)
radar_file = sorted(png_files)[-1]

metadata = {
    "date": date,
    "file": radar_file,
    "generated_at": datetime.utcnow().isoformat() + "Z"  # UTC-Zeitpunkt der Generierung
}

# metadata.json liegt eine Ebene über dem Ordner
meta_path = os.path.join(os.path.dirname(png_root), "metadata.json")
with open(meta_path, "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Metadata written to {meta_path}")
