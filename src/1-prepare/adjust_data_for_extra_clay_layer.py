"""
# script to adjust data for extra clay layer

made by: Romee van Dam
date: 22-01-26

"""

#%% imports
import imod
import numpy as np
import xarray as xr

#%% paths and parameters
path_base_dir = r"p:\afbouw-nazorg-bodemsanering\Weert\gwq_model_weert\DBASE"
path_vcw_org = fr"{path_base_dir}\BCF\VCW_L3.idf"
path_vcw_3a = fr"{path_base_dir}\BCF\VCW_part1"
path_vcw_3b = fr"{path_base_dir}\BCF\VCW_part2"

perc = 0.10 # bovenste 10% van laag


path_bron_l3_start = f"{path_base_dir}/SSM/CTVC_voor_1998_L3.idf"
path_bron_l5_start = f"{path_base_dir}/SSM/CTVC_voor_1998_L5.idf"
path_bron_l7_start = f"{path_base_dir}/SSM/CTVC_voor_1998_L7.idf"
path_bron_l3_eind = f"{path_base_dir}/SSM/CTVC_vanaf_1998_L3.idf"
path_bron_l5_eind = f"{path_base_dir}/SSM/CTVC_vanaf_1998_L5.idf"
path_bron_l7_eind = f"{path_base_dir}/SSM/CTVC_vanaf_1998_L7.idf"

path_top = fr"{path_base_dir}\DIS\TOP.idf"
path_dz = fr"{path_base_dir}\DIS\transport\DZ_L*.idf"
path_dz_output = fr"{path_base_dir}\DIS\transport\DZ_extra_layer.idf"


#%% adjust vcw

vcw_org = imod.idf.open(path_vcw_org)

# split in two layers
vcw_3a = perc*vcw_org
vcw_3b = (1-perc)*vcw_org


imod.idf.save(path_vcw_3a, vcw_3a,pattern="{name}_l{layer}{extension}")
imod.idf.save(path_vcw_3b, vcw_3b,pattern="{name}_l{layer}{extension}")




#%% adjust bronzones

bron_l3_start = imod.idf.open(path_bron_l3_start)
bron_l5_start = imod.idf.open(path_bron_l5_start)
bron_l7_start = imod.idf.open(path_bron_l7_start)
bron_l3_eind = imod.idf.open(path_bron_l3_eind )
bron_l5_eind = imod.idf.open(path_bron_l5_eind)
bron_l7_eind = imod.idf.open(path_bron_l7_eind)


bron_l6_eind =bron_l5_eind.where(bron_l5_eind == 9400, other=np.nan)
bron_l6_eind["layer"] = [6]
bron_l8_start = bron_l7_start.copy()
bron_l8_start["layer"] = [8]
bron_l8_eind = bron_l7_eind.copy()
bron_l8_eind["layer"] = [8]

imod.idf.save(f"{path_base_dir}/SSM/CTVC_zonder_spoortraject.idf", bron_l3_start, pattern="{name}_l{layer}{extension}")
imod.idf.save(f"{path_base_dir}/SSM/CTVC_zonder_spoortraject.idf", bron_l5_start, pattern="{name}_l{layer}{extension}")
imod.idf.save(f"{path_base_dir}/SSM/CTVC_zonder_spoortraject_nieuwe_lagenindeling.idf", bron_l8_start, pattern="{name}_l{layer}{extension}")
imod.idf.save(f"{path_base_dir}/SSM/CTVC_met_spoortraject_na_1998_nieuwe_lagenindeling.idf", bron_l8_eind, pattern="{name}_l{layer}{extension}") #toch wel bronzone in 2e zandlaag
imod.idf.save(f"{path_base_dir}/SSM/CTVC_spoortraject_na_1998_nieuwe_lagenindeling.idf", bron_l6_eind, pattern="{name}_l{layer}{extension}")

#%% adjust dz

top = imod.idf.open(path_top)
dz = imod.idf.open(path_dz)


# --- Split layer 6 ---
l = 6
dz6_top = dz.sel(layer=l) * 0.10      # 10%
dz6_bot = dz.sel(layer=l) * 0.90      # 90%
dz6_bot = dz6_bot.assign_coords(layer = dz6_bot["layer"].values+1 )


# --- Build the new 14-layer thickness array ---
# Keep layers 1..5 as-is
dz_1_5 = dz.sel(layer=slice(1, l-1))  # 1..5

# Old layers 7..13 will become 8..14
dz_7_13_shift = dz.sel(layer=slice(l+1, dz.sizes["layer"]))  # 7..13
dz_7_13_shift =dz_7_13_shift.assign_coords(layer = dz_7_13_shift["layer"].values+1)

# add together
dz_new = xr.concat(
    [dz_1_5, dz6_top, dz6_bot, dz_7_13_shift],
    dim="layer"
)

imod.idf.save(path_dz_output, dz_new)

# %%
