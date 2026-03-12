from __future__ import annotations

import sys
from pathlib import Path


def ensure_vnstock_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    local_repo = root / "vnstock"
    if local_repo.exists():
        repo_path = str(local_repo)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
