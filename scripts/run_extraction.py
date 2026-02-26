#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pipeline
import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extraction only (no assembly)")
    parser.add_argument(
        "--build",
        nargs="+",
        default=["all"],
        help="Modules to build. Use 'all' or module names",
    )
    parser.add_argument(
        "--forcing-mode",
        choices=["legacy", "datm"],
        default=None,
        help="Override forcing extraction mode from config",
    )
    parser.add_argument(
        "--prepare-forcing",
        action="store_true",
        help="Preprocess DATM monthly forcing files into consolidated DS4~DS9 files.",
    )
    parser.add_argument(
        "--prepare-forcing-only",
        action="store_true",
        help="Only preprocess forcing files and exit.",
    )
    parser.add_argument(
        "--force-rebuild-forcing",
        action="store_true",
        help="Rebuild consolidated forcing files even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline.ensure_dirs()
    if args.forcing_mode:
        config.FORCING_MODE = args.forcing_mode
    if args.prepare_forcing or args.prepare_forcing_only:
        run_pipeline.prepare_forcing_inputs_from_datm(force_rebuild=args.force_rebuild_forcing)
    if args.prepare_forcing_only:
        print("[Extraction] forcing preprocessing completed.")
        return
    run_pipeline.run_extraction(args.build)
    print(f"[Extraction] complete. Artifacts: {os.path.abspath(config.ARTIFACT_ROOT)}")


if __name__ == "__main__":
    main()
