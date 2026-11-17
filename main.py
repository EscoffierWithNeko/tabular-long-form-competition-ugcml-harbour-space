# -*- coding: utf-8 -*-
"""
Main entry script for [Kaggle]tabular-long-form-competition-ugcml-harbour-space
Redirects and runs baseline / latest enhanced pipeline.
"""
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    script = BASE_DIR / "baseline_lgb_regression.py"
    if script.exists():
        res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR))
        sys.exit(res.returncode)
    else:
        print(f"Target script not found: {script}")
        sys.exit(1)
