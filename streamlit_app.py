from __future__ import annotations

import runpy
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent / "geometric_math_reader_app"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

runpy.run_path(str(APP_DIR / "app.py"), run_name="__main__")
