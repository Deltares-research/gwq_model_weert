""" 
script to make coss-sections and topview of contamination plume and kh in Griftpark
Project: 11209163-015- Kempenweg Weert
author: Romee van Dam
date: 04-2026


"""
#%%
# check if correct python /pixi environment is used

import sys
import site
print("python:", sys.executable)
print("site-packages:", site.getsitepackages())
#%% imports

import imod
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import Point, LineString
import xugrid as xu
from matplotlib.colors import LogNorm, ListedColormap, BoundaryNorm
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import geopandas as gpd
import contextily as ctx
import pandas as pd
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import re
import xugrid as xu
import geopandas as gpd
from shapely.ops import unary_union, polygonize
import datetime as dtmod


#%% paths

piezo = False
scenario = "basis_v4.0_extra_laag_toekomst_pomp_uit_tot2326_fd" #"scenario_gat_in_kleilaagv2_verplaatstv2"
p_dir = r"p:\afbouw-nazorg-bodemsanering\Weert"
path_dbase = fr"{p_dir}\gwq_model_weert\DBASE"
path_results_folder = fr"{path_dbase}\runfiles\{scenario}\conc"
path_conc = fr"{path_results_folder}/conc_*_l*.idf"
path_top = fr"{path_dbase}/DIS/TOP.idf"
path_dz= fr"{path_dbase}/DIS/transport/DZ_extra_layer_l*.idf"
# path_bot = fr"{path_dbase}/DIS/BOT.idf"
# path_kh = fr"{path_dbase}/LPF/v2/kh_l*.idf"
# path_kh_hetero = fr"{path_dbase}/LPF/v2/kh_heterogeen_l*.idf"
# path_kh_gat = fr"{path_dbase}/LPF/v2/kh_heterogeen_gat_in_kleilaagv2_l*.idf"
# path_kva = fr"{path_dbase}/LPF/v2/KVA_heterogeen_l*.idf"
# path_damwand = rf"{p_dir}\model_suzanne\DBASE\HFB\damwand_cleaned_version.gen"
# path_transect1= fr"{p_dir}\TRANSECT_diagonal3.GEN"
# path_transect2 = fr"{p_dir}\TRANSECT_diagonal2.GEN"
# path_peilbuizen = fr"{p_dir}\Peilfilters_WVP2.ipf"
# path_prov = fr"{p_dir}\legenda\provincies\2021_provincies_zonder_water.shp"
path_legends = fr"{path_dbase}\legend"
if piezo == True:
    path_output_folder = rf"{p_dir}\gwq_model_weert\figures\{scenario}_met_peilbuis"
else:
    path_output_folder  = rf"{p_dir}\gwq_model_weert\figures\{scenario}"

species_names = [
    "chloroform"]

layer_1wvp = [1,2,3,4,5]
layer_2wvp = [8,9,10,11,12,13,14]

#aqtd_kh_criterion = 2.5
#aqtd_kh_criterion_hetero = 5


os.makedirs(path_output_folder, exist_ok = True)

#%% # definitions

def refine_grid(source, target):
    da_refined = xu.CentroidLocatorRegridder(source=source, target=target).regrid(source)
    #da_refined["dx"] = target["dx"]
    #da_refined["dy"] = target["dy"]
    return da_refined


def damwand_chainages_along_transect(transect: LineString, damwand_gdf, default_halfwidth=0.25):
    """
    Return list of intervals [(s0, s1), ...] in meters along transect where damwand crosses.
    Point intersections -> small interval around s; line intersections -> full overlap interval.
    """
    s_intervals = []
    for geom in damwand_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        inter = transect.intersection(geom)
        if inter.is_empty:
            continue
        if isinstance(inter, Point):
            s = transect.project(inter)
            s_intervals.append((s - default_halfwidth, s + default_halfwidth))
        elif isinstance(inter, LineString):
            s0 = transect.project(Point(inter.coords[0]))
            s1 = transect.project(Point(inter.coords[-1]))
            s_intervals.append((min(s0, s1), max(s0, s1)))
        else:
            # MultiPoint / MultiLineString
            try:
                for part in inter.geoms:
                    if isinstance(part, Point):
                        s = transect.project(part)
                        s_intervals.append((s - default_halfwidth, s + default_halfwidth))
                    elif isinstance(part, LineString):
                        s0 = transect.project(Point(part.coords[0]))
                        s1 = transect.project(Point(part.coords[-1]))
                        s_intervals.append((min(s0, s1), max(s0, s1)))
            except Exception:
                pass
    # Merge small overlaps
    s_intervals.sort()
    merged = []
    for s0, s1 in s_intervals:
        if not merged or s0 > merged[-1][1]:
            merged.append([s0, s1])
        else:
            merged[-1][1] = max(merged[-1][1], s1)
    return [(a, b) for a, b in merged]


def interp_profile_at_s(profile_da, s_values, coord_name='s', method='linear'):
    """
    Interpolate a 1D profile DataArray (values vs chainage coord) at s_values.
    - profile_da: xr.DataArray with a single spatial coord (e.g., 's')
    - s_values: scalar or array of chainages (meters)
    """
    s_axis = np.asarray(profile_da.coords[coord_name].values)
    z_vals = np.asarray(profile_da.values)
    s_values = np.atleast_1d(s_values)

    if method == 'nearest':
        idx = np.abs(s_axis[:, None] - s_values[None, :]).argmin(axis=0)
        return z_vals[idx]
    else:
        return np.interp(s_values, s_axis, z_vals)



def create_colorbar(species, transparent = False):
    "definition to create settings for colorbar per species"

    # Read legend file
    legend_file = fr"{path_legends}/legend_{species}.csv" 
    df = pd.read_csv(legend_file, skiprows=1)  # skip first line with metadata

    # Prepare cmap/norm from your leg files

    lower_bounds = df["LOWERBND"].tolist()
    upper_bounds = df["UPPERBND"].tolist()
    original_colors = [(r/255, g/255, b/255) for r, g, b in zip(df["IRED"], df["IGREEN"], df["IBLUE"])]
    labels = df["DOMAIN"].tolist()

    if transparent == True:
        # --- MODIFY WHITE TO TRANSPARENT (RGBA) ---
        colors = []
        for r, g, b in original_colors:
            if (r, g, b) == (1.0, 1.0, 1.0):   # detect white
                colors.append((r, g, b, 0.0)) # transparent
            else:
                colors.append((r, g, b, 0.8)) # fully opaque
    else:
        colors = original_colors


    # Boundaries: first upper bound (top class), then each lower bound (remaining class edges)
    bounds = [upper_bounds[0]] + lower_bounds   # length must be len(colors)+1

    # IMPORTANT: keep original order from the file. Do NOT sort.

    if not all(np.diff(bounds) > 0):
        bounds = bounds[::-1]
        colors = colors[::-1]
        labels = labels[::-1]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, ncolors=cmap.N)

    # test colorbar
    # fig, ax = plt.subplots(figsize=(6, 1), layout='constrained')

    # fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
    #              cax=ax, orientation='horizontal',
    #              label="Discrete intervals with extend='both' keyword")
    return labels, bounds, colors, cmap, norm, species

#%% # definitions piezometer plot


def project_points_onto_transect(transect: LineString, df_points):
    """
    Compute orthogonal projection of piezometers onto the transect.
    Returns a DataFrame with extra columns: 's' (chainage along transect, m),
    'foot_x', 'foot_y' (coordinates of projection), and 'dist' (perpendicular distance, m).
    
    df_points must have columns 'x', 'y', and the elevation fields you want to plot.
    """
    s_vals = []
    foot_x = []
    foot_y = []
    dists = []
    for x, y in zip(df_points["x"].values, df_points["y"].values):
        pt = Point(float(x), float(y))
        s = transect.project(pt)                # chainage along line (meters)
        foot = transect.interpolate(s)          # projected point on line
        s_vals.append(s)
        foot_x.append(foot.x)
        foot_y.append(foot.y)
        dists.append(pt.distance(transect))     # shortest distance (meters)
    out = df_points.copy()
    out["s"] = s_vals
    out["foot_x"] = foot_x
    out["foot_y"] = foot_y
    out["dist"] = dists
    return out



def plot_piezometers_on_xsec(
    ax,
    piezos_projected,
    x_offset=0.0,
    show_labels=True,
    label_kwargs=None,
    lw_filter=2.0,
    lw_pb=2.0,
    color_filter="tab:blue",      # blauwe dashed lijn
    color_pb="tab:blue",              # effen grijze lijn
    zorder=60,
):
    """
    Plot:
      - Filter als blauwe dashed verticale lijn (bovenkant filter -> onderkant filter).
      - Verbinding van filtertop naar bovenkant peilbuis als effen grijze verticale lijn.
    Geen maaiveld-markering.
    """
    if label_kwargs is None:
        label_kwargs = dict(fontsize=8, rotation=0, va="bottom", ha="left")

    for _, row in piezos_projected.iterrows():
        s = float(row["s"]) + x_offset

        z_filter_top = float(row["bovenkant filter (m NAP)"])
        z_filter_bot = float(row["onderkantfilter (m NAP)"])
        z_pb_top     = float(row["bovenkant pb (m NAP)"])
        name         = str(row["naam"])

        # 1) Filter: blauwe dashed verticale lijn + kapjes
        ax.plot([s, s], [z_filter_bot, z_filter_top],
                color=color_filter, lw=lw_filter, linestyle=(0, (1, 1)), zorder=zorder)
        ax.plot([s-0.5, s+0.5], [z_filter_top, z_filter_top],
                color=color_filter, lw=lw_filter, zorder=zorder)
        ax.plot([s-0.5, s+0.5], [z_filter_bot, z_filter_bot],
                color=color_filter, lw=lw_filter, zorder=zorder)

        # 2) Verbinding filtertop -> bovenkant peilbuis: effen grijs
        ax.plot([s, s], [z_filter_top, z_pb_top],
                color=color_pb, lw=lw_pb, linestyle="solid", zorder=zorder+1)

        # 3) Optioneel: label naast filtertop
        if show_labels:
            ax.text(s + 1.0, z_filter_top, name, **label_kwargs)

    # Legend entries (geen maaiveld-item)
    filt_patch = mpatches.Patch(facecolor="none", edgecolor=color_filter, label="Peilfilter (:)")
    pb_patch   = mpatches.Patch(facecolor=color_pb,  edgecolor=color_pb,  label="Peilbuis")

    return [filt_patch, pb_patch]


#%% prepare data

# open files
conc = imod.idf.open(path_conc)
dx, xmin, xmax, dy, ymin, ymax = imod.util.spatial_reference(conc)
conc = imod.util.where(conc<0, 0, conc)

top_l1 = imod.idf.open(path_top).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
dz = imod.idf.open(path_dz).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
bot = top_l1 -dz
top = bot + dz


# bot = imod.idf.open(path_bot).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
# kh = imod.idf.open(path_kh).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
# kh_hetero = imod.idf.open(path_kh_hetero).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
# kh_gat = imod.idf.open(path_kh_gat).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))
# #kva = imod.idf.open(path_kva).sel(x=slice(xmin, xmax), y = slice(ymax,ymin))

# create like
like = conc.isel(time= 0).copy(data=np.zeros_like(conc.isel(time= 0).data))
like = like.drop_vars([ "time"]) 
like["dx"] = dx 
like["dy"] = dy


# regrid data
top_10m = refine_grid(top, like).compute()
bot_10m = refine_grid(bot, like).compute()
# kh_10m = refine_grid(kh, like)
# kh_10m =kh_10m.assign_coords(top=top_10m,bottom=bot_10m)
# kh_10m_org = kh_10m.copy()
# kh_10m.loc[dict(layer=kh_hetero.layer)] = kh_hetero

# add top and bottom as coordinates for cross_section
conc["top"] = top_10m
conc["bottom"] = bot_10m

# transects
# transect1 =  imod.gen.read(path_transect1)
# transect2 =  imod.gen.read(path_transect2)

# shapefile provinces
# shp_prov = gpd.read_file(path_prov)

# peilbuizen
# peilbuizen = imod.ipf.read(path_peilbuizen)
# peilbuizen.loc[peilbuizen["naam"] == "BPT B2 WVP2", "naam"] = "BPT B2"
# peilbuizen.loc[peilbuizen["naam"] == "BPT B (64) WVP2","naam"] = "BPT B (64)"
# peilbuizen_trans1 = peilbuizen.loc[peilbuizen["naam"].isin(["DV1", "56", "51", "BPT B (64)", "BPT B2" ])]
# peilbuizen_trans2 = peilbuizen.loc[peilbuizen["naam"].isin(["35"])]


#%% refined template for correct plotting of cross-section

# IMPORTANT!!! the grids need to be refined to avoid to large jumps (ds) in transect

# create template
if scenario in ["scenario_puur_product_in2ewvpv3","scenario_lage_afbraakv3", "scenario_lage_afbraakv3v2"]:
    dx_new = 2.0 #1.0
    dy_new = -2.0 # -1.0
else: 
    dx_new = 1.0 #1.0
    dy_new = -1.0 # -1.0


y_coords = np.arange(ymax, ymin, dy_new) + 0.5*dy_new # center of cell
x_coords = np.arange(xmin, xmax, dx_new) + 0.5*dx_new # center of cell
layers = conc["layer"].values

nrow = y_coords.size
ncol = x_coords.size
nlay = layers.size

# define template
dims = ("layer", "y", "x")
coords = {
    "layer": layers,
    "y": y_coords, # center of cell
    "x": x_coords, # center of cell
    "dx": dx_new,
    "dy":dy_new,
}

like_refined = xr.DataArray(np.nan, coords, dims)

# refine datasets for correct plotting
top_refined = refine_grid(top_10m, like_refined)
bottom_refined  = refine_grid(bot_10m, like_refined)

conc_refined = refine_grid(conc, like_refined)
# kh_refined = refine_grid(kh_10m, like_refined)
# kh_org_refined = refine_grid(kh_10m_org, like_refined)

conc_refined["top"] = top_refined
conc_refined["bottom"] = bottom_refined

# kh_org_refined["top"] = top_refined
# kh_org_refined["bottom"]= bottom_refined

# kh_refined["top"] = top_refined
# kh_refined["bottom"]= bottom_refined

# create aquitards
#aquitards_refined = kh_org_refined < aqtd_kh_criterion
#aquitards_zonder_hetero_refined = aquitards_refined.copy()
#aquitards_zonder_hetero_refined.loc[dict(layer=kh_hetero.layer)] = 0
# aquitards_hetero_refined = kh_refined < aqtd_kh_criterion_hetero

# create mask damwand
# xx, yy = np.meshgrid(kh_refined.x.values,kh_refined.y.values )    # shape (Ny, Nx) #Maak 2D grid met cell-centers
# inside_mask = np.zeros(xx.shape, dtype=bool)
# # check if cell is within damwand or not
# for iy in range(yy.shape[0]):
#     for ix in range(xx.shape[1]):
#         p = Point(xx[iy, ix], yy[iy, ix])
#         inside_mask[iy, ix] = damwand_polygon.contains(p)
# # create dataarray
# mask_damwand = xr.DataArray(
#     inside_mask,
#     dims=("y", "x"),
#     coords={"y": kh_refined.y.values, "x": kh_refined.x.values}
# )
# # kh within damwand
# kh_within_damwand= kh_refined.where(mask_damwand)


#%% create cross-sections

# # plot cross_section of concentration per species, time and transect
# print("plot cross_sections")

# #for species in conc.sel(species=slice(0,13))["species"].values:
# for species in conc.sel(species=slice(1))["species"].values:
#     for time in conc["time"].values:
           

#         # different time format
#         date_ddmmyyyy = time.astype('datetime64[D]').astype(object).strftime("%d-%m-%Y")
#         time_short = time.astype(str)[:10]

#         print(species_names[species-1], time_short) 
#         # select time
#         conc_slice = conc_refined.sel(species=species, time=time)
#         #conc_slice = damwand_da.sel(species=species, time=time)

#         for transect_reversed, name in zip([transect1, transect2],["profiel 1", "profiel 2"]):

#             # Define a transect
#             coords = list(transect_reversed["geometry"][0].coords)[::-1]
#             transect = LineString(coords)

#             # Extract cross-section along transect
#             xsect = imod.select.cross_section_linestring(conc_slice, transect)
#             # create aquitards for plotting
#             #xaqtd = imod.select.cross_section_linestring(aquitards_zonder_hetero_refined, transect)
#             # create hetero aquitards
#             xaqtd_hetero = imod.select.cross_section_linestring(aquitards_hetero_refined, transect)

#             # settings for plot
#             # colors = ["#ffffff", "#ffd700", "#ffa500", "#ff4500", "#ff0000"]  # yellow → red gradient
#             labels, bounds, colors, cmap, norm, species_name = create_colorbar(species)
#             levels = [1e-14, 1e-11, 1e-8, 1e-5] # not used / overwritten but needed as parameter in imod.visualize.cross_section()
#             aqtd_patch = mpatches.Patch(
#             color="grey", alpha=0.5, hatch="////", ec="k", label=f"Scheidende laag (kh<{aqtd_kh_criterion_hetero} m/d)"
#             )


#             # Plot cross-section of conc data
#             fig, ax = plt.subplots(figsize=(10, 6))
#             imod.visualize.cross_section(
#                 xsect,
#                 aquitards= xaqtd_hetero, #xaqtd,
#                 colors = colors,
#                 levels = levels,
#                 layers=False,
#                 fig=fig,
#                 ax=ax,
#                 kwargs_pcolormesh={
#                     "norm": norm, #LogNorm(vmin=levels[0], vmax=levels[-1]),
#                     "cmap": cmap, #"YlOrRd"  # use colors/levels instead of a continuous cmap
#                 },
#                 kwargs_colorbar={
#                     "plot_colorbar" : False # we create the colorbar manually
#                 },
#                 kwargs_aquitards={
#                     "alpha": 0.5,
#                     "facecolor": "grey",
#                     "hatch": "////",
#                     "edgecolor": "k",
#                 },            
#             )

#             # create heterogeneous aquitards

#             # layers_second = [21, 22, 23, 24, 25, 26]

#             # xs_hetero = xsect.s.values
#             # top_hetero = xsect["top"].values        # (layer, s)
#             # bottom_hetero = xsect["bottom"].values  # (layer, s)

#             # for k in layers_second:
#             #     k_idx = k - 1   # convert layer number → zero-based index
                
#             #     # Build polygon coordinates for this layer
#             #     xcoords = np.concatenate([xs_hetero, xs_hetero[::-1]])  # forward + backward
#             #     zcoords = np.concatenate([top_hetero[k_idx], bottom_hetero[k_idx][::-1]])
                
#             #     poly = mpatches.Polygon(
#             #         np.column_stack([xcoords, zcoords]),
#             #         facecolor="white",
#             #         edgecolor="k",
#             #         alpha=0.5,
#             #         hatch="\\\\",
#             #         linewidth=0.5,
#             #         zorder=6,
#             #     )
#             #     ax.add_patch(poly)


#             # legend
#             # aqtd2_patch = mpatches.Patch(
#             #     facecolor="white",
#             #     alpha=0.5,
#             #     hatch="\\\\",
#             #     edgecolor="k",
#             #     label="Diepte heterogene kleilagen",
#             # )

   
#             ax.set_title(f"Dwarsdoorsnede concentraties {species_names[species-1]}, {name}, op {date_ddmmyyyy} ")
#             ax.set_xlabel("Afstand over profiel [m]")
#             ax.set_ylabel("Hoogte [m NAP]")


            
#             # Draw the damwand from TOP down to bottom of layer 20 at location of transect

#             # Sample TOP and bottom of layer 26 along the transect (1D profiles vs distance)
#             top_cs      = imod.select.cross_section_linestring(top.sel(layer=1), transect)
#             bottom_cs = imod.select.cross_section_linestring(bot.sel(layer=20), transect)          
#             # Find damwand crossings as chainages along the transect
#             s_intervals = damwand_chainages_along_transect(transect, default_halfwidth=0.25)  # ~0.5 m visible width


#             # Create ScalarMappable with the same norm and cmap
#             sm = ScalarMappable(norm=norm, cmap=cmap)
#             # Ensure the mappable has a valid range; either set clim or attach a dummy array
#             sm.set_clim(bounds[0], bounds[-1])
#             # Create a discrete colorbar using the boundaries
#             cbar = fig.colorbar(
#                 sm,
#                 ax=ax,
#                 boundaries=bounds,          # ensure discrete bins
#                 spacing='uniform',     # bins sized by interval widths
#             )
#             # Compute bin centers
#             bin_centers = 0.5 * (np.array(bounds[:-1]) + np.array(bounds[1:]))
#             # Apply to colorbar
#             cbar.set_ticks(bin_centers)
#             #cbar.set_ticks(bounds[:-1]) # lable at tick 
#             cbar.set_ticklabels(labels)
#             cbar.set_label("Concentratie")


#             # insert location of crossection

#             axins = inset_axes(ax, width=1.6, height=1.8, loc=4,borderpad=1)
#             axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#             #shp_prov.plot(ax=axins, fc="None", ec="k", lw=0.2)
#             axins.plot(*transect.xy, "r--", lw=1)
#             # add basemap
            
#             xmin_t, ymin_t, xmax_t, ymax_t = transect.bounds
#             # Add a small buffer to avoid zero width/height
#             buffer = 50  # meters
#             axins.set_xlim(xmin_t - buffer, xmax_t + buffer)
#             axins.set_ylim(ymin_t - buffer, ymax_t + buffer)
#             ctx.add_basemap(axins, crs="EPSG:28992", source=ctx.providers.OpenStreetMap.Mapnik)



#             # --- Build base legend entries you already have
#             # aqtd_patch = mpatches.Patch(
#             #     color="grey", alpha=0.5, hatch="////", ec="k",
#             #     label=f"Scheidende laag (kh<{aqtd_kh_criterion_hetero} m/d)"
#             # )

#             # Start legend lists (so we can safely extend later)
#             # legend_handles = [aqtd_patch] #[aqtd_patch, aqtd2_patch, dw_patch]
#             # legend_labels = [h.get_label() for h in legend_handles]
#             # ax.legend(legend_handles, legend_labels, loc="lower left", frameon=True)

#             # add piezometers to graph
#             if piezo == True:
#                 # --- Project piezometers onto transect chainage (loodrecht)
#                 # Pick the correct set for this transect name; else -> None (explicit)
#                 if name == "profiel 1":
#                     pb_sel = peilbuizen_trans1
#                 elif name == "profiel 2":
#                     pb_sel = peilbuizen_trans2
#                 else:
#                     pb_sel = None  # Unknown transect name: nothing to plot

#                 if pb_sel is not None and len(pb_sel) > 0:
#                     # Ensure required columns exist exactly as named
#                     required_cols = [
#                         "x", "y",
#                         "Maaiveldhoogten (m NAP)",
#                         "bovenkant pb (m NAP)",
#                         "bovenkant filter (m NAP)",
#                         "onderkantfilter (m NAP)",
#                         "naam",
#                     ]
#                     missing = [c for c in required_cols if c not in pb_sel.columns]

#                     if len(missing) == 0:
#                         # Compute orthogonal projection to the current transect
#                         pb_proj = project_points_onto_transect(transect, pb_sel)

#                         # Optional: only show piezometers within X meters from the transect
#                         #max_dist = 50.0  # adjust as needed, or comment out to show all
#                         #pb_proj = pb_proj.loc[pb_proj["dist"] <= max_dist].copy()

#                         if len(pb_proj) > 0:
#                             # Plot vertical filter segments + markers + labels

#                             piezo_handles = plot_piezometers_on_xsec(
#                                 ax, pb_proj,
#                                 x_offset=0.0,
#                                 show_labels=True,
#                                 label_kwargs=dict(fontsize=8, rotation=0, va="bottom", ha="left"),
#                                 lw_filter=1.0,
#                                 lw_pb=1.0,
#                                 color_filter="blue",
#                                 color_pb="blue",
#                                 zorder=60,
#                             )

#                             # Merge legend items (zonder duplicaten)
#                             for h in piezo_handles:
#                                 label = h.get_label()
#                                 if label and label not in legend_labels:
#                                     legend_handles.append(h)
#                                     legend_labels.append(label)


#                             ax.legend(legend_handles, legend_labels, loc="lower left", frameon=True)
#                         else:
#                             print(f"[INFO] No piezometers within {max_dist} m of {name}; nothing plotted.")
#                     else:
#                         # ELSE (missing required columns)
#                         print(f"[WARN] Missing columns in peilbuizen for {name}: {missing}. Skipping plotting piezometers.")


#             #plt.show()
#             plt.savefig(rf"{path_output_folder}/cross_section_conc_{species_name}_{name}_{time_short}.jpeg", dpi = 500)
#             plt.close()





#%% plot concentration 2e wvp

print("plot concentration 2e wvp")
# plot concentration per species, time
for layer in [8,9]:
    species = species_names[0]
    for time in conc["time"].values:
        
        # different time format
        date_ddmmyyyy = time.strftime("%d-%m-%Y")
        time_short = date_ddmmyyyy
        print(species, time_short)
        # select time
        conc_slice = conc.sel(time=time)

        # colorbar settings
        labels, bounds, colors, cmap, norm, species_name = create_colorbar(species, transparent= True)
        colors_t= colors
        colors_t[0] = (1.0, 1.0, 1.0, 0) 
        cmap_t = ListedColormap(colors_t)
        norm_t = BoundaryNorm(bounds, ncolors=cmap_t.N)

        # change values <S to np.nan for visibility of the background map
        conc_slice = imod.util.where(conc_slice<bounds[1], np.nan, conc_slice )
    
        # plot maximum concentration
        fig, ax = plt.subplots(figsize=(8, 6))
        da_max = conc_slice.sel(layer=layer)#.max("layer")
        im = da_max.plot(ax=ax, cmap=cmap_t, norm=norm_t, add_colorbar=False )  # we will add a manual colorbar
        ax.set_title(f"Concentratie in 2e wvp (laag {layer}) voor {species}, {date_ddmmyyyy} ")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        
        # Add basemap:
        basemap_img = ctx.add_basemap(
            ax,
            crs="EPSG:28992",
            source=ctx.providers.OpenStreetMap.Mapnik,
            alpha =1.0
        )
        # Keep your data visible
        im.set_zorder(10)
        
        # create colorbar manually
        # Create ScalarMappable with the same norm and cmap
        sm = ScalarMappable(norm=norm_t, cmap=cmap_t)
        # Ensure the mappable has a valid range; either set clim or attach a dummy array
        sm.set_clim(bounds[0], bounds[-1])
        # Create a discrete colorbar using the boundaries
        cbar = fig.colorbar(
            sm,
            ax=ax,
            boundaries=bounds,          # ensure discrete bins
            spacing='uniform',     # bins sized by interval widths
        )
        # Place one tick per bin (left edges); you can also use bin centers if you prefer
        
        # Compute bin centers
        bin_centers = 0.5 * (np.array(bounds[:-1]) + np.array(bounds[1:]))
        # Apply to colorbar
        cbar.set_ticks(bin_centers)
        #cbar.set_ticks(bounds[:-1]) # lable at tick 
        cbar.set_ticklabels(labels)
        cbar.set_label("Concentratie")
        plt.savefig(rf"{path_output_folder}/conc_2ewvp_l{layer}_{species}_{time_short}.jpeg", dpi=500)
        plt.close()




#%% plot concentration 1e wvp

print("plot concentration 1e wvp")
# plot concentration per species, time

species = species_names[0]
for time in conc["time"].values:
      
    # different time format
    date_ddmmyyyy = time.strftime("%d-%m-%Y") #time.astype('datetime64[D]').astype(dtmod.datetime).strftime("%d-%m-%Y")
    time_short = date_ddmmyyyy
    print(species_names[0], time_short)
    # select time
    conc_slice = conc.sel(time=time)

    # colorbar settings
    labels, bounds, colors, cmap, norm, species_name = create_colorbar(species, transparent= True)
    colors_t= colors
    colors_t[0] = (1.0, 1.0, 1.0, 0) 
    cmap_t = ListedColormap(colors_t)
    norm_t = BoundaryNorm(bounds, ncolors=cmap_t.N)

    # change values <S to np.nan for visibility of the background map
    conc_slice = imod.util.where(conc_slice<bounds[1], np.nan, conc_slice )
  
    # plot maximum concentration
    fig, ax = plt.subplots(figsize=(8, 6))
    da_max = conc_slice.sel(layer=5)#.max("layer")
    im = da_max.plot(ax=ax, cmap=cmap_t, norm=norm_t, add_colorbar=False, )  # we will add a manual colorbar
    ax.set_title(f"Concentratie in 1e wvp (laag 5) voor {species_names[0]}, {date_ddmmyyyy} ")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    
    # Add basemap:
    basemap_img = ctx.add_basemap(
        ax,
        crs="EPSG:28992",
        source=ctx.providers.OpenStreetMap.Mapnik,
        alpha =1.0
    )
    # Keep your data visible
    im.set_zorder(10)
    
    # create colorbar manually
    # Create ScalarMappable with the same norm and cmap
    sm = ScalarMappable(norm=norm_t, cmap=cmap_t)
    # Ensure the mappable has a valid range; either set clim or attach a dummy array
    sm.set_clim(bounds[0], bounds[-1])
    # Alternatively: sm.set_array(np.array([bounds[0], bounds[-1]]))
    # Create a discrete colorbar using the boundaries
    cbar = fig.colorbar(
        sm,
        ax=ax,
        boundaries=bounds,          # ensure discrete bins
        spacing='uniform',     # bins sized by interval widths
    )
    # Place one tick per bin (left edges); you can also use bin centers if you prefer
    
    # Compute bin centers
    bin_centers = 0.5 * (np.array(bounds[:-1]) + np.array(bounds[1:]))
    # Apply to colorbar
    cbar.set_ticks(bin_centers)
    #cbar.set_ticks(bounds[:-1]) # lable at tick 
    cbar.set_ticklabels(labels)
    cbar.set_label("Concentratie")
    plt.savefig(rf"{path_output_folder}/conc_1ewvp_l5_{species_names[0]}_{time_short}.jpeg", dpi=500)
    plt.close()

#%% plot kh cross-section

# print("plot kh")

# # plot cross_section of kh per  transect


# for transect_reversed, name in zip([transect1, transect2],["profiel 1", "profiel 2"]):
#     # Define a transect
#     coords = list(transect_reversed["geometry"][0].coords)[::-1]
#     transect = LineString(coords)
#     # Extract cross-section along transect
#     xsect = imod.select.cross_section_linestring(kh_refined, transect)
#     # create aquitards for plotting
#     # xaqtd = imod.select.cross_section_linestring(aquitards_zonder_hetero_refined, transect)
#     # create hetero aquitards
#     # xaqtd_hetero = imod.select.cross_section_linestring(aquitards_hetero_refined, transect)

#     # create colorbar for kh for aquifer and aquitards
#     levels = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5,
#           10, 20, 30, 40, 50, 60]
    
    
#     colors_aquitards = [

#         "#ff1500", # 0.5
#         "#ff3300",
#         "#ff6600",
#         "#ff8000",
#         "#ff9900",
#         "#ffb300",
#         "#ffcc00",
#         "#ffe600",
#         "#ffff00",
#         "#ffff66"   # 5.0 (yellow)
#     ]
#     colors_aquifer = [

#     "#e6ffe6",  # 5 - 10  (very light green)
#     "#ccffcc",  # 10
#     "#b3ffb3",  # 20
#     "#99ff99",  # 30
#     "#66ff66",  # 40
#     "#33cc33",  # 50 (strong green)
#     "#1f7a1f" # 60 (strong green)

#     ]
#     colors = colors_aquitards + colors_aquifer
  
#     # aqtd_patch = mpatches.Patch(
#     # color="grey", alpha=0.5, hatch="////", ec="k", label=f"Scheidende laag (kh<{aqtd_kh_criterion} m/d)"
#     # )


#     # Plot cross-section of conc data
#     fig, ax = plt.subplots(figsize=(10, 6))
#     imod.visualize.cross_section(
#         xsect,
#         #aquitards=xaqtd,
#         colors = colors,
#         levels = levels,
#         layers=False,
#         fig=fig,
#         ax=ax,
#         kwargs_colorbar={
#             "extend": "min",
#             "label": "kh (m/d)"
#             },

#         # kwargs_aquitards={
#         #     "alpha": 0.5,
#         #     "facecolor": "grey",
#         #     "hatch": "////",
#         #     "edgecolor": "k",
#         # },            
#     )

#     # # create heterogeneous aquitards

#     # layers_second = [21, 22, 23, 24, 25, 26]

#     # xs_hetero = xsect.s.values
#     # top_hetero = xsect["top"].values        # (layer, s)
#     # bottom_hetero = xsect["bottom"].values  # (layer, s)

#     # for k in layers_second:
#     #     k_idx = k - 1   # convert layer number → zero-based index
        
#     #     # Build polygon coordinates for this layer
#     #     xcoords = np.concatenate([xs_hetero, xs_hetero[::-1]])  # forward + backward
#     #     zcoords = np.concatenate([top_hetero[k_idx], bottom_hetero[k_idx][::-1]])
        
#     #     poly = mpatches.Polygon(
#     #         np.column_stack([xcoords, zcoords]),
#     #         facecolor="white",
#     #         edgecolor="k",
#     #         alpha=0.5,
#     #         hatch="\\\\",
#     #         linewidth=0.5,
#     #         zorder=6,
#     #     )
#     #     ax.add_patch(poly)


#     # # legend
#     # aqtd2_patch = mpatches.Patch(
#     #     facecolor="white",
#     #     alpha=0.5,
#     #     hatch="\\\\",
#     #     edgecolor="k",
#     #     label="diepte heterogene kleilagen",
#     # )



   

    
#     # Draw the damwand from TOP down to bottom of layer 20 at location of transect
#     # Sample TOP and bottom of layer 26 along the transect (1D profiles vs distance)
#     top_cs      = imod.select.cross_section_linestring(top.sel(layer=1), transect)
#     bottom_cs = imod.select.cross_section_linestring(bot.sel(layer=20), transect)          
#     # Find damwand crossings as chainages along the transect
#     s_intervals = damwand_chainages_along_transect(transect, damwand, default_halfwidth=0.25)  # ~0.5 m visible width
#     damwand_patches = []
#     for s0, s1 in s_intervals:
#         s_mid = 0.5 * (s0 + s1)
#         z_top_mid = float(interp_profile_at_s(top_cs, [s_mid])[0])
#         z_bot_mid = float(interp_profile_at_s(bottom_cs, [s_mid])[0])
#         width  = max(0.2, (s1 - s0))      # ensure visible width
#         height = z_top_mid - z_bot_mid
#         rect = mpatches.Rectangle(
#             (s_mid - width / 2.0, z_bot_mid),
#             width,
#             height,
#             facecolor="black",
#             edgecolor="black",
#             linewidth=3,
#             alpha=1.0,
#             zorder=50,
#         )
#         ax.add_patch(rect)
#         damwand_patches.append(rect)
#     # Legend entry for the wall and aquitards
#     dw_patch = mpatches.Patch(facecolor="black", edgecolor="black", label="Schermwand")


#     # insert location of crossection
#     axins = inset_axes(ax, width=1.6, height=1.8, loc=4,borderpad=1)
#     axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
#     #shp_prov.plot(ax=axins, fc="None", ec="k", lw=0.2)
#     axins.plot(*transect.xy, "r--", lw=1)
#     # add basemap
#     damwand.plot(ax=axins, color="black", linewidth=1.0, zorder=30)
    
#     xmin_t, ymin_t, xmax_t, ymax_t = transect.bounds
#     # Add a small buffer to avoid zero width/height
#     buffer = 50  # meters
#     axins.set_xlim(xmin_t - buffer, xmax_t + buffer)
#     axins.set_ylim(ymin_t - buffer, ymax_t + buffer)
#     ctx.add_basemap(axins, crs="EPSG:28992", source=ctx.providers.OpenStreetMap.Mapnik)
#     # --- Build base legend entries you already have
#     # aqtd_patch = mpatches.Patch(
#     #     color="grey", alpha=0.5, hatch="////", ec="k",
#     #     label=f"Scheidende laag (kh<{aqtd_kh_criterion_hetero} m/d)"
#     # )
#     dw_patch = mpatches.Patch(facecolor="black", edgecolor="black", label="Schermwand")
#     # Start legend lists (so we can safely extend later)
#     #legend_handles = [aqtd_patch, aqtd2_patch, dw_patch]
#     legend_handles = [dw_patch]
#     legend_labels = [h.get_label() for h in legend_handles]
#     ax.legend(legend_handles, legend_labels, loc="lower left", frameon=True)
    
#     ax.set_title(f"Dwarsdoorsnede horizontale doorlatendheid (kh) in m/d voor {name} ")
#     ax.set_xlabel("Afstand over profiel [m]")
#     ax.set_ylabel("Hoogte [m NAP]")
    
#     # draw piezometers into graph
#     if piezo == True:
#         # --- Project piezometers onto transect chainage (loodrecht)
#         # Pick the correct set for this transect name; else -> None (explicit)
#         if name == "profiel 1":
#             pb_sel = peilbuizen_trans1
#         elif name == "profiel 2":
#             pb_sel = peilbuizen_trans2
#         else:
#             pb_sel = None  # Unknown transect name: nothing to plot
#         if pb_sel is not None and len(pb_sel) > 0:
#             # Ensure required columns exist exactly as named
#             required_cols = [
#                 "x", "y",
#                 "Maaiveldhoogten (m NAP)",
#                 "bovenkant pb (m NAP)",
#                 "bovenkant filter (m NAP)",
#                 "onderkantfilter (m NAP)",
#                 "naam",
#             ]
#             missing = [c for c in required_cols if c not in pb_sel.columns]
#             if len(missing) == 0:
#                 # Compute orthogonal projection to the current transect
#                 pb_proj = project_points_onto_transect(transect, pb_sel)
#                 # Optional: only show piezometers within X meters from the transect
#                 #max_dist = 50.0  # adjust as needed, or comment out to show all
#                 #pb_proj = pb_proj.loc[pb_proj["dist"] <= max_dist].copy()
#                 if len(pb_proj) > 0:
#                     # Plot vertical filter segments + markers + labels
#                     piezo_handles = plot_piezometers_on_xsec(
#                         ax, pb_proj,
#                         x_offset=0.0,
#                         show_labels=True,
#                         label_kwargs=dict(fontsize=8, rotation=0, va="bottom", ha="left"),
#                         lw_filter=1.0,
#                         lw_pb=1.0,
#                         color_filter="blue",
#                         color_pb="blue",
#                         zorder=60,
#                     )
#                     # Merge legend items (zonder duplicaten)
#                     for h in piezo_handles:
#                         label = h.get_label()
#                         if label and label not in legend_labels:
#                             legend_handles.append(h)
#                             legend_labels.append(label)
#                     ax.legend(legend_handles, legend_labels, loc="lower left", frameon=True)
#                 else:
#                     print(f"[INFO] No piezometers within {max_dist} m of {name}; nothing plotted.")
#             else:
#                 # ELSE (missing required columns)
#                 print(f"[WARN] Missing columns in peilbuizen for {name}: {missing}. Skipping plotting piezometers.")
#     #plt.show()
#     plt.savefig(rf"{path_output_folder}/cross_section_kh_{name}.jpeg", dpi = 500)
#     plt.close()
