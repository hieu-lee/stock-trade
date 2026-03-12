#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().with_name("run_recommendations.py")
    os.execv(sys.executable, [sys.executable, str(script), "--profile", "vn30_core", *sys.argv[1:]])


if __name__ == "__main__":
    main()
