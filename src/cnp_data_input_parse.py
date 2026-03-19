import os
import re
from typing import Dict, List


SECTION_MAP = {
    "TIME SERIES VARIABLES": "time_series_variables",
    "SURFACE PROPERTIES": "surface_properties",
    "PFT PARAMETERS": "pft_parameters",
    "SCALAR VARIABLES": "scalar_variables",
    "1D PFT VARIABLES": "pft_1d_variables",
    "2D VARIABLES": "variables_2d_soil",
    "RESTART_COL_1D_VARS": "restart_col_1d_vars",
}


def _parse_list_value(raw: str) -> List[str]:
    if not raw:
        return []
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def parse_cnp_data_input(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        return {"scalars": {}, "sections": {v: [] for v in SECTION_MAP.values()}}

    sections: Dict[str, List[str]] = {v: [] for v in SECTION_MAP.values()}
    scalars: Dict[str, str] = {}
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            matched_header = False
            for title, section_key in SECTION_MAP.items():
                if line.upper().startswith(title):
                    current_section = section_key
                    matched_header = True
                    break
            if matched_header:
                continue

            if line.startswith("•") and current_section:
                sections[current_section].extend(_parse_list_value(line.lstrip("•").strip()))
                continue

            scalar_match = re.match(r"^([A-Za-z0-9_]+)\s*[:=]\s*(.*)$", line)
            if scalar_match:
                key = scalar_match.group(1).strip()
                value = scalar_match.group(2).strip()
                scalars[key] = value
                continue

            if current_section and "," in line:
                sections[current_section].extend(_parse_list_value(line))
                continue

            if current_section and re.match(r"^[A-Za-z0-9_]+$", line):
                sections[current_section].append(line)

    for key, values in sections.items():
        sections[key] = list(dict.fromkeys(values))

    # -----------------------------
    # Flexibility upgrades
    # -----------------------------
    # 1) Alias keys for readability in config files.
    #    The downstream loader expects canonical keys like DS1_PATH, DS2_PATH, etc.
    alias_to_canonical = {
        "surfdata_path": "DS1_PATH",
        "init_h0_file": "DS2_PATH",
        "init_r_path": "DS10_PATH",
        "target_ho_path": "H0_LIST_PATHS",
        "target_r_path": "R_LIST_PATHS",
        "clm_params_path": "CLM_PARAMS_PATH",
        # Forcing readability (optional)
        "forcing_root": "DATM_ROOT",
    }
    for alias_key, canonical_key in alias_to_canonical.items():
        if alias_key in scalars and canonical_key not in scalars:
            scalars[canonical_key] = scalars[alias_key]

    # 2) Expand variables embedded in scalar values.
    #    Supports: $KEY, ${KEY}, {KEY}
    #    This is needed for configs that reference other scalars,
    #    e.g. `${EXPID}` inside DS2_PATH templates.
    def _expand(value: str, var_map: Dict[str, str]) -> str:
        new_v = value
        for k, v in var_map.items():
            if not k or not isinstance(v, str):
                continue
            new_v = new_v.replace(f"${k}", v)
            new_v = new_v.replace(f"${{{k}}}", v)
            new_v = new_v.replace(f"{{{k}}}", v)
        return new_v

    # Iteratively expand a few times to handle cases where one scalar
    # template expands into another template.
    var_map: Dict[str, str] = {k: v for k, v in scalars.items() if isinstance(v, str)}
    for _ in range(5):
        before = dict(scalars)
        for k, v in list(scalars.items()):
            if not isinstance(v, str):
                continue
            scalars[k] = _expand(v, var_map)
        if before == scalars:
            break

    return {"scalars": scalars, "sections": sections}
