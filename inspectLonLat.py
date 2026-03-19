import netCDF4 as nc

ds = nc.Dataset("/gpfs/wolf2/cades/cli185/proj-shared/wangd/AI_data/TES_NORTH_dataset/ERA5_10PCTSITES/history_restart_files/uELM_TESNorthERA510PCT_I1850CNPRDCTCBC.elm.h0.0021-01-01-00000.nc")
print(ds.variables["lat"][:].min(), ds.variables["lat"][:].max())
print(ds.variables["lon"][:].min(), ds.variables["lon"][:].max())
ds.close()