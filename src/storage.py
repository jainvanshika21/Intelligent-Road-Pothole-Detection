from __future__ import annotations

import time
from pathlib import Path

from .config import UPLOADS_DIR


def save_upload(file_name: str, data: bytes) -> Path:
    """Save an uploaded file to disk and return the saved path."""
    safe_name = Path(file_name).name
    ts = time.strftime("%Y%m%d_%H%M%S")
    target = UPLOADS_DIR / f"{ts}_{safe_name}"
    target.write_bytes(data)
    return target
