import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patches as mpatches
from fueltype_analysis_attributes import \
    Wet_Shrubland, Wet_Shrubland_labels, CM_Wet_Shrubland, \
    Wet_Forest, Wet_Forest_labels, CM_Wet_Forest, \
    Grassland, Grassland_labels, CM_Grassland, \
    Dry_forest, Dry_forest_labels, CM_Dry_forest, \
    Shrubland, Shrubland_labels, CM_Shrubland, \
    High_elevation, High_elevation_labels, CM_High_elevation, \
    Mallee, Mallee_labels, CM_Mallee 

import argparse

'''
Initialise argument parsing: 
GCM for the different ensemble members: ACCESS1-0 BNU-ESM 
CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M INM-CM4 
IPSL-CM5A-LR MRI-CGCM3
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
GCM = args.GCM
scen = args.scen
timespan = args.timespan

### Get predictor dataset for lat and lon coordinates
if GCM == 'mode':
  df = pd.read_csv('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
                   'dynamics_simulations/CFA/ML/input/cache/'
                   'ft.ACCESS1-0.'+scen+'_'+timespan+'.csv')
else:
  df = pd.read_csv('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
                   'dynamics_simulations/CFA/ML/input/cache/'
                   'ft.'+GCM+'.'+scen+'_'+timespan+'.csv')  

### Drop Temperate Grassland / Sedgeland (3020) and
### Eaten Out Grass when it's NOT on public land  
df = df.loc[~((df['FT'] == 3020) & (df['tenure'] == 0)),:]
df = df.loc[~((df['FT'] == 3046) & (df['tenure'] == 0)),:]

### Set inf to Nan
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_dropna = df.dropna()

### Read in projected fuel type distribution
df_fut = pd.read_csv('../output/csv/csv_FT/'+scen+'_'+timespan+
                     '/'+GCM+'_'+scen+'_'+timespan+'.csv')

### Combine dataframes
df_dropna['mode']= df_fut[GCM].values.flatten()

### Select relevant columns
df_sel = df_dropna[['lat','lon','mode']]
df_final = df_sel.set_index(['lat','lon'])

### Convert to xarray dataset
ds = df_final.to_xarray()

### This is a bit clunky but I set up a colormap for each fuel group to manage
### to assign each fuel type a specific color consistent across the report
def colormap_fueltype(ft_list, ft_colors):
  cmap = mpl.colors.ListedColormap(ft_colors)
  ft_list.append(4000)
  bounds = ft_list
  norm = BoundaryNorm(bounds, cmap.N)
  return(cmap,bounds,norm)

### Make mask
def mask_vic(dataset,ft_list):
  da_copy = dataset
  mask = np.isin(da_copy, ft_list)
  mx = np.ma.masked_array(da_copy, ~mask)
  return(mx)

lats = ds.lat.values.tolist()
lons = ds.lon.values.tolist()

### Set up plot
fig, ax = plt.subplots(figsize=(11, 6),
                       subplot_kw={'projection': ccrs.PlateCarree()})

def plot(dataset, ft_list, ft_colors):
  global lats
  global lons

  ### Get attributes for colormap
  cmap,bounds,norm = colormap_fueltype(ft_list, ft_colors)

  ### Get data
  data = mask_vic(dataset,ft_list)

  ### Make map
  im = ax.pcolormesh(lons,
                     lats,
                     data,
                     cmap=cmap,
                     norm=norm,
                     rasterized=True,
                     transform=ccrs.PlateCarree())

### Call function for each fuel group: Pretty clunky but couldn't figure out 
### another way to link fuel type labels and specific colors
plot(ds['mode'].values,Wet_Shrubland,CM_Wet_Shrubland)
plot(ds['mode'].values,Wet_Forest,CM_Wet_Forest)
plot(ds['mode'].values,Grassland,CM_Grassland)
plot(ds['mode'].values,Dry_forest,CM_Dry_forest)
plot(ds['mode'].values,Shrubland,CM_Shrubland)
plot(ds['mode'].values,High_elevation,CM_High_elevation)
plot(ds['mode'].values,Mallee,CM_Mallee)

'''
Set the extent: 
subtract 0.05 from the first latitude
add 0.05 to the last longitude
'''

ax.set_extent([lons[0] - 0.05, lons[-1] + 0.05,
               lats[0] - 0.05, lats[-1] + 0.05],
               crs=ccrs.PlateCarree())

### Add state borders
ax.add_feature(cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces_lines', '10m',
    edgecolor='k', facecolor='none',
    linewidth=1.0, linestyle='solid'))

ax.coastlines()

### Adding state borders looks a bit ugly in NW where vertical state borders 
### of NSW and Vic are offset, also in SE unnecessary border. Then ACT is this 
### random blob - so added patches to cover those areas (probably easier getting
### Victoria shape from a shape file but I didn't have one at the time)

### NSW border SE
ax.add_patch(mpatches.Rectangle(xy=[149.5, -37.313], width=0.6, height=0.85,
                                facecolor='w',
                                zorder=12,
                                angle=338,
                                transform=ccrs.PlateCarree())
                        )

### NSW border NW
ax.add_patch(mpatches.Rectangle(xy=[140.992, -34.005], width=0.02, height=0.08,
                                facecolor='w',
                                zorder=12,
                                transform=ccrs.PlateCarree())
                        )

### ACT
ax.add_patch(mpatches.Rectangle(xy=[148.7, -35.95], width=0.8, height=0.85,
                                facecolor='w',
                                zorder=12,
                                transform=ccrs.PlateCarree())
                        )

### Drop spines
ax.spines['geo'].set_visible(False)

### Reintroduce left and bottom spine
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

### Show ticklabels left and bottom
ax.xaxis.set_visible(True)
ax.yaxis.set_visible(True)

### Save figure
plt.tight_layout()
plt.savefig('figures/mode_'+scen+'_'+timespan+'.jpg',dpi=1000)
