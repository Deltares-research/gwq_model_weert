"""
# script to extend model domain

project: weert
made by: Romee van Dam
date: 30-03-26

"""
#%% imports
import imod
import numpy as np
import xarray as xr

#%% paths and parameters

path_base = r"p:\afbouw-nazorg-bodemsanering\Weert\gwq_model_weert\DBASE"
path_cbnd = fr"{path_base}/bas/cbnd.IDF"
out_path  = fr"{path_base}\bas\cbnd_groter.IDF"



xmax_model = 172000.0
ymax_model = 361995.0


NEW_BOUNDARY = -1.0   # nieuwe boven + rechterrand
OLD_BOUNDARY_TO = 1.0 # oude rand (was -1) moet nu 1 worden


#%%
cbnd = imod.idf.open(path_cbnd)


dx = float(cbnd["dx"])
dy = float(cbnd["dy"]) 


# Oude randen (celcentra)
old_xmax = float(cbnd.x.max())
old_ymax = float(cbnd.y.max())
xmin = float(cbnd.x.min())
ymin = float(cbnd.y.min())


# --- nieuw grid-coördinaten maken (celcentra) ---
# x oplopend
x_new = np.arange(xmin, xmax_model + 0.5 * dx, dx)

# y aflopend (dy negatief): start bij ymax_model en stap naar beneden
y_new = np.arange(ymax_model, ymin + 0.5 * dy, dy)

# --- reindex naar groter raster ---
cbnd_big = cbnd.reindex(x=x_new, y=y_new, method=None)

# Vul nieuwe gebieden eerst met 1 (interieur) als “default”
# (want alles binnen het domein is in principe geen boundary)
cbnd_big = cbnd_big.fillna(OLD_BOUNDARY_TO)

# ------------------------------------------------------------
# 1) Oude rand (originele bovenrij en rechterkolom) naar 1 zetten
#    Dit is precies de rand die voorheen -1 was.
# ------------------------------------------------------------
old_x_edge = xr.apply_ufunc(np.isclose, cbnd_big["x"], old_xmax).broadcast_like(cbnd_big)
old_y_edge = xr.apply_ufunc(np.isclose, cbnd_big["y"], old_ymax).broadcast_like(cbnd_big)

cbnd_big = cbnd_big.where(~(old_x_edge | old_y_edge), other=OLD_BOUNDARY_TO)

# ------------------------------------------------------------
# 2) Nieuwe buitenrand (bovenste rij + rechterkolom van het nieuwe grid) naar -1
# ------------------------------------------------------------
new_xmax = float(cbnd_big.x.max())
new_ymax = float(cbnd_big.y.max())

new_x_edge = xr.apply_ufunc(np.isclose, cbnd_big["x"], new_xmax).broadcast_like(cbnd_big)
new_y_edge = xr.apply_ufunc(np.isclose, cbnd_big["y"], new_ymax).broadcast_like(cbnd_big)

cbnd_big = cbnd_big.where(~(new_x_edge | new_y_edge), other=NEW_BOUNDARY)

# --- schrijven ---
imod.idf.write(out_path, cbnd_big)

print("Klaar:", out_path)
print("Nieuw extent x:", float(cbnd_big.x.min()), float(cbnd_big.x.max()))
print("Nieuw extent y:", float(cbnd_big.y.min()), float(cbnd_big.y.max()))
print("Shape:", cbnd_big.shape)
