#!/usr/bin/env python3
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pipeline
import config


def main() -> None:
    run_pipeline.ensure_dirs()
    run_pipeline.run_assembly()
    print(f"[Assembly] complete. Output: {os.path.abspath(config.FINAL_OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
