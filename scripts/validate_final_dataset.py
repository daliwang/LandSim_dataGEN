#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pipeline
import config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate assembled training dataset pickles.")
    parser.add_argument(
        "--config-input",
        type=str,
        default=None,
        help="Path to CNP_dataInput-style config (defaults to repo default in src/config.py).",
    )
    parser.add_argument(
        "--final-dir",
        type=str,
        default=None,
        help="Final dataset directory (default: config.FINAL_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=20,
        help="Number of rows to sample per forcing column for shape/type validation.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Validate only first N batches (useful for quick smoke tests).",
    )
    return parser.parse_args()


def _is_finite_array_like(x: Any) -> bool:
    try:
        arr = np.asarray(x, dtype=float)
    except Exception:
        return False
    return np.all(np.isfinite(arr))


def _check_forcing_column(df: pd.DataFrame, col: str, expected_len: int, sample_rows: int) -> List[str]:
    issues: List[str] = []
    if col not in df.columns:
        issues.append(f"Missing forcing column: {col}")
        return issues

    sample = df[col].iloc[:sample_rows].tolist()
    for idx, v in enumerate(sample):
        if isinstance(v, (float, int, np.floating, np.integer)):
            issues.append(f"{col}: row {idx} expected list/array, got scalar {type(v)}")
            continue
        if not isinstance(v, (list, tuple, np.ndarray)):
            issues.append(f"{col}: row {idx} expected list/array, got {type(v)}")
            continue
        if len(v) != expected_len:
            issues.append(f"{col}: row {idx} expected len={expected_len}, got len={len(v)}")
            continue
        if not _is_finite_array_like(v):
            issues.append(f"{col}: row {idx} contains non-finite values")
    return issues


def _check_numeric_column(df: pd.DataFrame, col: str, max_nan_ratio: float = 0.001) -> List[str]:
    issues: List[str] = []
    if col not in df.columns:
        return issues
    s = df[col]
    # Some columns may contain lists as strings due to pickling; coerce like assembler does.
    numeric = pd.to_numeric(s.astype(str).str.strip(), errors="coerce")
    nan_ratio = float(numeric.isna().mean()) if len(numeric) else 0.0
    if nan_ratio > max_nan_ratio:
        issues.append(f"{col}: NaN ratio too high ({nan_ratio:.4g}, max {max_nan_ratio:.4g})")
    if not np.all(np.isfinite(numeric.dropna().to_numpy(dtype=float))):
        issues.append(f"{col}: contains non-finite numeric values")
    return issues


def main() -> None:
    args = parse_args()
    if args.config_input:
        config.load_config(args.config_input)
    final_dir = args.final_dir or config.FINAL_OUTPUT_DIR
    final_dir = os.path.abspath(final_dir)

    if not os.path.isdir(final_dir):
        raise FileNotFoundError(f"Final dataset dir not found: {final_dir}")

    # Authoritative batch ids come from A_index_core.
    index_path = os.path.join(run_pipeline.module_dir("A_index_core"), "index_master.pkl")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Missing A_index_core index_master: {index_path}")
    index_df = pd.read_pickle(index_path)
    batch_ids = run_pipeline.get_batch_ids(index_df)
    if args.max_batches is not None:
        batch_ids = batch_ids[: int(args.max_batches)]

    forcing_cols = ["FLDS", "PSRF", "FSDS", "QBOT", "PRECTmms", "TBOT"]
    expected_months = int(config.YEARS_IN_DATA) * 12

    single_value_columns = [
        "landfrac",
        "LANDFRAC_PFT",
        "PCT_NATVEG",
        "AREA",
        "SOIL_COLOR",
        "SOIL_ORDER",
        "GPP",
        "Y_GPP",
        "HR",
        "AR",
        "NPP",
        "Y_HR",
        "Y_AR",
        "Y_NPP",
        "OCCLUDED_P",
        "SECONDARY_P",
        "LABILE_P",
        "APATITE_P",
    ]

    # Coordinate sanity
    lat_min, lat_max = float(config.LAT2), float(config.LAT1)
    lon_min, lon_max = float(config.LON1), float(config.LON2)

    ok = True
    total_issues = 0

    for batch_id in batch_ids:
        out_path = os.path.join(final_dir, f"training_data_batch_{batch_id:02d}.pkl")
        core_batch_path = os.path.join(run_pipeline.module_dir("A_index_core"), f"batch_{batch_id:02d}.pkl")

        expected_rows = None
        if os.path.exists(core_batch_path):
            core_df = pd.read_pickle(core_batch_path)
            expected_rows = len(core_df)
        else:
            expected_rows = int((index_df["batch_id"] == batch_id).sum())

        if not os.path.exists(out_path):
            ok = False
            total_issues += 1
            print(f"[batch {batch_id:02d}] MISSING: {out_path}")
            continue

        df = pd.read_pickle(out_path)
        if len(df) != expected_rows:
            ok = False
            total_issues += 1
            print(f"[batch {batch_id:02d}] row_count mismatch: got {len(df)}, expected {expected_rows}")

        # Latitude/Longitude basic checks.
        for coord_col, lo, hi in [("Latitude", lat_min, lat_max), ("Longitude", lon_min, lon_max)]:
            if coord_col in df.columns:
                vals = pd.to_numeric(df[coord_col].astype(str), errors="coerce").to_numpy(dtype=float)
                if np.isnan(vals).mean() > 0.0:
                    ok = False
                    total_issues += 1
                    print(f"[batch {batch_id:02d}] {coord_col}: contains NaNs")
                # Don’t enforce strict bounds for lon if your data uses wrapping; use a loose check.
                if np.nanmin(vals) < (lo - 1e-6) or np.nanmax(vals) > (hi + 1e-6):
                    print(f"[batch {batch_id:02d}] warning: {coord_col} out of expected range ({lo},{hi})")

        # Forcing columns: must be lists of floats with expected monthly length.
        for col in forcing_cols:
            issues = _check_forcing_column(df, col, expected_months, args.sample_rows)
            if issues:
                ok = False
                total_issues += len(issues)
                for msg in issues[:10]:
                    print(f"[batch {batch_id:02d}] {msg}")
                if len(issues) > 10:
                    print(f"[batch {batch_id:02d}] ... ({len(issues) - 10} more issues)")

        # Numeric scalar columns: basic NaN/non-finite checks.
        for col in single_value_columns:
            issues = _check_numeric_column(df, col, max_nan_ratio=0.01)
            if issues:
                ok = False
                total_issues += len(issues)
                for msg in issues:
                    print(f"[batch {batch_id:02d}] {msg}")

        # Expanded PCT columns sanity: after assembly, base columns are expanded and base columns dropped.
        for prefix in ["PCT_NAT_PFT_", "PCT_SAND_", "PCT_CLAY_"]:
            if any(c.startswith(prefix) for c in df.columns):
                continue
            # Only warn if the original base columns were likely requested.
            # (We can’t easily know from this script whether they were configured; so keep it a warning.)
            print(f"[batch {batch_id:02d}] warning: no columns found with prefix {prefix}")

        print(f"[batch {batch_id:02d}] OK (so far)")

    print("")
    if ok:
        print(f"Validation SUCCESS for {len(batch_ids)} batch(es).")
        sys.exit(0)
    else:
        print(f"Validation FAILED: total issues found: {total_issues}")
        sys.exit(1)


if __name__ == "__main__":
    main()

