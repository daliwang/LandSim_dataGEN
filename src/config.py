import os
from pathlib import Path

from cnp_data_input_parse import parse_cnp_data_input


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_INPUT_FILE = PROJECT_ROOT / "config" / "CNP_dataInput.txt"
ACTIVE_CONFIG_INPUT_FILE = str(CONFIG_INPUT_FILE)

# =========================
# Base Paths
# =========================
# Note: BASE_OUTPUT_ROOT will be overridden by CNP_dataInput.txt
BASE_OUTPUT_ROOT = str(PROJECT_ROOT / "output")
PIPELINE_ROOT = os.path.join(BASE_OUTPUT_ROOT, "modular_by_input_v1")
ARTIFACT_ROOT = os.path.join(PIPELINE_ROOT, "artifacts")
FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_ROOT, "final_dataset")
MANIFEST_ROOT = os.path.join(PIPELINE_ROOT, "manifests")

# =========================
# Input Files
# =========================
# Note: All file paths are defined in CNP_dataInput.txt and will override these defaults
FILE_PATHS = {}

# =========================
# Variable Definitions
# =========================
# Note: These will be overridden by CNP_dataInput.txt if defined there
RESTART_PFT_VARS = []
RESTART_COL_1D_VARS = []
RESTART_COL_2D_VARS = []
STATIC_SURFACE_VARS_2D = []
STATIC_SURFACE_VARS_3D = []
HISTORY_GRID_VARS_2D = []

FORCING_MODULE_MAP = {
    "A_forcing_ds4_flds": ("ds4", "FLDS"),
    "A_forcing_ds5_psrf": ("ds5", "PSRF"),
    "A_forcing_ds6_fsds": ("ds6", "FSDS"),
    "A_forcing_ds7_qbot": ("ds7", "QBOT"),
    "A_forcing_ds8_prectmms": ("ds8", "PRECTmms"),
    "A_forcing_ds9_tbot": ("ds9", "TBOT"),
}

# Note: These will be overridden by CNP_dataInput.txt if defined there
FORCING_MODE = "legacy"
DATM_ROOT = ""
DATM_START_YEAR = 1901
DATM_END_YEAR = 2023
# Initialize with empty values for all forcing variables
DATM_TOKEN_MAP = {
    "FLDS": "",
    "PSRF": "",
    "FSDS": "",
    "QBOT": "",
    "PRECTmms": "",
    "TBOT": "",
}
DATM_FORCING_PATHS = {
    "FLDS": "",
    "PSRF": "",
    "FSDS": "",
    "QBOT": "",
    "PRECTmms": "",
    "TBOT": "",
}
PFT_TARGET_VARS = []

LAT1, LON1 = 90, 0
LAT2, LON2 = -90, 360
BATCH_SIZE = 1000
TIME_SERIES_LENGTH = 179580
STEPS_PER_DAY = 4
YEARS_IN_DATA = 123
DAYS_PER_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

ALL_MODULES = [
    "A_index_core",
    "A_ds1_surface",
    "A_ds2_history_x",
    "A_ds10_restart_x",
    "A_h0_list_y",
    "A_r_list_y",
    "A_forcing_ds4_flds",
    "A_forcing_ds5_psrf",
    "A_forcing_ds6_fsds",
    "A_forcing_ds7_qbot",
    "A_forcing_ds8_prectmms",
    "A_forcing_ds9_tbot",
    "A_clm_params_pft",
]


def _to_int(raw: str, fallback: int) -> int:
    try:
        return int(raw)
    except Exception:
        return fallback


def _parse_list(raw: str):
    values = [part.strip() for part in raw.split(",")]
    return [value for value in values if value]


def _apply_cnp_data_input_overrides(config_path: str) -> None:
    global BASE_OUTPUT_ROOT
    global PIPELINE_ROOT
    global ARTIFACT_ROOT
    global FINAL_OUTPUT_DIR
    global MANIFEST_ROOT
    global LAT1
    global LAT2
    global LON1
    global LON2
    global BATCH_SIZE
    global TIME_SERIES_LENGTH
    global STEPS_PER_DAY
    global YEARS_IN_DATA
    global FORCING_MODE
    global DATM_ROOT
    global DATM_START_YEAR
    global DATM_END_YEAR
    global RESTART_PFT_VARS
    global RESTART_COL_2D_VARS
    global RESTART_COL_1D_VARS
    global HISTORY_GRID_VARS_2D
    global STATIC_SURFACE_VARS_2D
    global STATIC_SURFACE_VARS_3D
    global PFT_TARGET_VARS
    global ACTIVE_CONFIG_INPUT_FILE

    ACTIVE_CONFIG_INPUT_FILE = config_path
    parsed = parse_cnp_data_input(config_path)
    scalars = parsed.get("scalars", {})
    sections = parsed.get("sections", {})

    if scalars.get("BASE_OUTPUT_ROOT"):
        BASE_OUTPUT_ROOT = scalars["BASE_OUTPUT_ROOT"]
        PIPELINE_ROOT = os.path.join(BASE_OUTPUT_ROOT, "modular_by_input_v1")
        ARTIFACT_ROOT = os.path.join(PIPELINE_ROOT, "artifacts")
        FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_ROOT, "final_dataset")
        MANIFEST_ROOT = os.path.join(PIPELINE_ROOT, "manifests")

    path_key_map = {
        "DS1_PATH": "ds1",
        "DS2_PATH": "ds2",
        "DS4_PATH": "ds4",
        "DS5_PATH": "ds5",
        "DS6_PATH": "ds6",
        "DS7_PATH": "ds7",
        "DS8_PATH": "ds8",
        "DS9_PATH": "ds9",
        "DS10_PATH": "ds10",
        "CLM_PARAMS_PATH": "clm_params",
    }
    for scalar_key, file_key in path_key_map.items():
        if scalars.get(scalar_key):
            FILE_PATHS[file_key] = scalars[scalar_key]

    if scalars.get("H0_LIST_PATHS"):
        FILE_PATHS["h0_list"] = _parse_list(scalars["H0_LIST_PATHS"])
    if scalars.get("R_LIST_PATHS"):
        FILE_PATHS["r_list"] = _parse_list(scalars["R_LIST_PATHS"])

    LAT1 = _to_int(scalars.get("LAT1", str(LAT1)), LAT1)
    LAT2 = _to_int(scalars.get("LAT2", str(LAT2)), LAT2)
    LON1 = _to_int(scalars.get("LON1", str(LON1)), LON1)
    LON2 = _to_int(scalars.get("LON2", str(LON2)), LON2)
    BATCH_SIZE = _to_int(scalars.get("BATCH_SIZE", str(BATCH_SIZE)), BATCH_SIZE)
    TIME_SERIES_LENGTH = _to_int(scalars.get("TIME_SERIES_LENGTH", str(TIME_SERIES_LENGTH)), TIME_SERIES_LENGTH)
    STEPS_PER_DAY = _to_int(scalars.get("STEPS_PER_DAY", str(STEPS_PER_DAY)), STEPS_PER_DAY)
    YEARS_IN_DATA = _to_int(scalars.get("YEARS_IN_DATA", str(YEARS_IN_DATA)), YEARS_IN_DATA)

    FORCING_MODE = (scalars.get("FORCING_MODE", FORCING_MODE) or FORCING_MODE).strip().lower()
    DATM_ROOT = scalars.get("DATM_ROOT", DATM_ROOT)
    DATM_START_YEAR = _to_int(scalars.get("DATM_START_YEAR", str(DATM_START_YEAR)), DATM_START_YEAR)
    DATM_END_YEAR = _to_int(scalars.get("DATM_END_YEAR", str(DATM_END_YEAR)), DATM_END_YEAR)

    token_overrides = {
        "DATM_FLDS_TOKEN": "FLDS",
        "DATM_PSRF_TOKEN": "PSRF",
        "DATM_FSDS_TOKEN": "FSDS",
        "DATM_QBOT_TOKEN": "QBOT",
        "DATM_PRECTMMS_TOKEN": "PRECTmms",
        "DATM_TBOT_TOKEN": "TBOT",
    }
    for scalar_key, var_name in token_overrides.items():
        if scalars.get(scalar_key):
            DATM_TOKEN_MAP[var_name] = scalars[scalar_key]

    path_overrides = {
        "DATM_FLDS_PATH": "FLDS",
        "DATM_PSRF_PATH": "PSRF",
        "DATM_FSDS_PATH": "FSDS",
        "DATM_QBOT_PATH": "QBOT",
        "DATM_PRECTMMS_PATH": "PRECTmms",
        "DATM_TBOT_PATH": "TBOT",
    }
    for scalar_key, var_name in path_overrides.items():
        if scalars.get(scalar_key):
            DATM_FORCING_PATHS[var_name] = scalars[scalar_key]

    if sections.get("time_series_variables"):
        time_vars = sections["time_series_variables"]
        module_keys = list(FORCING_MODULE_MAP.keys())
        if len(time_vars) == len(module_keys):
            FORCING_MODULE_MAP[module_keys[0]] = ("ds4", time_vars[0])
            FORCING_MODULE_MAP[module_keys[1]] = ("ds5", time_vars[1])
            FORCING_MODULE_MAP[module_keys[2]] = ("ds6", time_vars[2])
            FORCING_MODULE_MAP[module_keys[3]] = ("ds7", time_vars[3])
            FORCING_MODULE_MAP[module_keys[4]] = ("ds8", time_vars[4])
            FORCING_MODULE_MAP[module_keys[5]] = ("ds9", time_vars[5])

    if sections.get("surface_properties"):
        static_surface = sections["surface_properties"]
        STATIC_SURFACE_VARS_2D = [name for name in static_surface if name not in {"PCT_SAND", "PCT_CLAY", "PCT_NAT_PFT"}]
        STATIC_SURFACE_VARS_3D = [name for name in static_surface if name in {"PCT_SAND", "PCT_CLAY", "PCT_NAT_PFT"}]

    if sections.get("scalar_variables"):
        HISTORY_GRID_VARS_2D = sections["scalar_variables"]

    if sections.get("pft_1d_variables"):
        RESTART_PFT_VARS = sections["pft_1d_variables"]

    if sections.get("variables_2d_soil"):
        RESTART_COL_2D_VARS = sections["variables_2d_soil"]

    if sections.get("restart_col_1d_vars"):
        RESTART_COL_1D_VARS = sections["restart_col_1d_vars"]

    if sections.get("pft_parameters"):
        normalized = []
        for name in sections["pft_parameters"]:
            normalized.append(name[4:] if name.startswith("pft_") else name)
        PFT_TARGET_VARS = normalized


def load_config(config_input_file: str | None = None) -> None:
    """
    Load or reload configuration from a CNP_dataInput-style text file.

    If config_input_file is None, use the default CONFIG_INPUT_FILE path.
    """
    config_path = config_input_file if config_input_file is not None else str(CONFIG_INPUT_FILE)
    _apply_cnp_data_input_overrides(config_path)


# Load configuration once at import time using the default config file.
load_config()

