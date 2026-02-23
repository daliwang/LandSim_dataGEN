import os

# =========================
# Base Paths
# =========================
BASE_OUTPUT_ROOT = "/mnt/DATA/0_oak_data/3_dataset_based_construction"
PIPELINE_ROOT = os.path.join(BASE_OUTPUT_ROOT, "modular_by_input_v1")
ARTIFACT_ROOT = os.path.join(PIPELINE_ROOT, "artifacts")
FINAL_OUTPUT_DIR = os.path.join(BASE_OUTPUT_ROOT, "final_dataset")
MANIFEST_ROOT = os.path.join(PIPELINE_ROOT, "manifests")

# =========================
# Input Files (aligned with 93_based_73_refactor.py)
# =========================
FILE_PATHS = {
    "ds1": "/mnt/DATA/0_oak_data/0_dataset_construction/surfdata_0.9x1.25_hist_1700_17pfts_c240731.nc",
    "ds2": "/mnt/DATA/0_oak_data/0_dataset_construction/0_new_correct_20_restart/20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.h0.0021-01-01-00000.nc",
    "ds4": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_FLDS_1901-2023_z01.nc",
    "ds5": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_PSRF_1901-2023_z01.nc",
    "ds6": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_FSDS_1901-2023_z01.nc",
    "ds7": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_QBOT_1901-2023_z01.nc",
    "ds8": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_PRECTmms_1901-2023_z01.nc",
    "ds9": "/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_TBOT_1901-2023_z01.nc",
    "ds10": "/mnt/DATA/0_oak_data/0_dataset_construction/0_new_correct_20_restart/20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc",
    "h0_list": ["/mnt/DATA/0_oak_data/0_dataset_construction/20251201_TRENDY2024_default_ICB1850CNPRDCTCBC.elm.h0.0801-01-01-00000.nc"],
    "r_list": ["/mnt/DATA/0_oak_data/0_dataset_construction/20251201_TRENDY2024_default_ICB1850CNPRDCTCBC.elm.r.0801-01-01-00000.nc"],
    "clm_params": "/home/UNT/dg0997/all_gdw/0_oak_weather/16_add_4_surf_input_output/clm_params.nc",
}

# =========================
# Variable Definitions
# =========================
RESTART_PFT_VARS = [
    "totvegc", "deadstemn", "deadcrootn", "deadstemp", "deadcrootp",
    "leafc", "leafc_storage", "frootc", "frootc_storage",
    "deadcrootc", "deadstemc", "tlai", "leafn", "leafn_storage", "frootn",
    "frootn_storage", "leafp", "leafp_storage", "frootp", "frootp_storage",
    "livestemc", "livestemc_storage", "livestemn", "livestemn_storage",
    "livestemp", "livestemp_storage", "deadcrootc_storage", "deadstemc_storage",
    "livecrootc", "livecrootc_storage", "deadcrootn_storage", "deadstemn_storage",
    "livecrootn", "livecrootn_storage", "deadcrootp_storage", "deadstemp_storage",
    "livecrootp", "livecrootp_storage",
    "leafc_xfer", "frootc_xfer", "livestemc_xfer", "deadstemc_xfer",
    "livecrootc_xfer", "deadcrootc_xfer", "gresp_xfer",
    "leafn_xfer", "frootn_xfer", "livestemn_xfer", "deadstemn_xfer",
    "livecrootn_xfer", "deadcrootn_xfer",
    "leafp_xfer", "frootp_xfer", "livestemp_xfer", "deadstemp_xfer",
    "livecrootp_xfer", "deadcrootp_xfer",
    "cpool", "npool", "ppool", "xsmrpool",
]
RESTART_COL_1D_VARS = []
RESTART_COL_2D_VARS = [
    "cwdn_vr", "secondp_vr", "cwdp_vr", "soil3c_vr", "soil4c_vr", "cwdc_vr",
    "soil1c_vr", "soil1n_vr", "soil1p_vr", "soil2c_vr", "soil2n_vr", "soil2p_vr",
    "soil3n_vr", "soil3p_vr", "soil4n_vr", "soil4p_vr", "litr1c_vr", "litr2c_vr",
    "litr3c_vr", "litr1n_vr", "litr2n_vr", "litr3n_vr", "litr1p_vr", "litr2p_vr",
    "litr3p_vr", "sminn_vr", "smin_no3_vr", "smin_nh4_vr", "labilep_vr",
    "occlp_vr", "primp_vr", "solutionp_vr",
]

STATIC_SURFACE_VARS_2D = [
    "LANDFRAC_PFT", "PCT_NATVEG", "AREA", "SOIL_COLOR", "SOIL_ORDER",
    "OCCLUDED_P", "SECONDARY_P", "LABILE_P", "APATITE_P",
]
STATIC_SURFACE_VARS_3D = ["PCT_SAND", "PCT_CLAY", "PCT_NAT_PFT"]
HISTORY_GRID_VARS_2D = ["GPP", "HR", "AR", "NPP"]

FORCING_MODULE_MAP = {
    "A_forcing_ds4_flds": ("ds4", "FLDS"),
    "A_forcing_ds5_psrf": ("ds5", "PSRF"),
    "A_forcing_ds6_fsds": ("ds6", "FSDS"),
    "A_forcing_ds7_qbot": ("ds7", "QBOT"),
    "A_forcing_ds8_prectmms": ("ds8", "PRECTmms"),
    "A_forcing_ds9_tbot": ("ds9", "TBOT"),
}

PFT_TARGET_VARS = [
    "deadwdcn", "frootcn", "leafcn", "lflitcn", "livewdcn", "c3psn", "croot_stem", "crop", "dleaf",
    "dsladlai", "evergreen", "fcur", "flivewd", "flnr", "fr_fcel", "fr_flab", "fr_flig", "froot_leaf",
    "grperc", "grpnow", "leaf_long", "lf_fcel", "lf_flab", "lf_flig", "rholnir", "rholvis", "rhosnir", "rhosvis",
    "roota_par", "rootb_par", "rootprof_beta", "season_decid", "slatop", "smpsc", "smpso", "stem_leaf", "stress_decid",
    "taulnir", "taulvis", "tausnir", "tausvis", "woody", "xl", "z0mr",
]

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

