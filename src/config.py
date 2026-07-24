from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

for _p in (DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, MODELS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
