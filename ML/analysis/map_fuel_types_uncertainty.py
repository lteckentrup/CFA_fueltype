import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

import argparse

'''
Initialise argument parsing: 
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
scen = args.scen
timespan = args.timespan

### Set pathway
pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/')

### Get lat and lon coordinates from predictor file
df = pd.read_csv(pathway+'input/cache/pred.ACCESS1-0.'+scen+'_'+timespan+'.csv')

### Drop Temperate Grassland / Sedgeland (3020) and
### Eaten Out Grass when it's NOT on public land  
df = df.loc[~((df['FT'] == 3020) & (df['tenure'] == 0)),:]
df = df.loc[~((df['FT'] == 3046) & (df['tenure'] == 0)),:]

### Drop Water, sand, no vegetation (3000)
df.replace(3000, np.nan, inplace=True)

### Drop Non-Combustible (3047)
df.replace(3047, np.nan, inplace=True)

### Drop Orchard / Vineyard (3097),
### Softwood Plantation (3098),
### Hardwood Plantation (3099)
df.replace(3097, np.nan, inplace=True)
df.replace(3098, np.nan, inplace=True)
df.replace(3099, np.nan, inplace=True)

### Set inf to Nan
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df_dropna = df.dropna()

### Read in future projection
df_fut = pd.read_csv(pathway+'output/csv/csv_FT/'+scen+'_'+timespan+'/mode_'+
                     scen+'_'+timespan+'.csv')

### Combine dataframes 
df['mode']= df_fut['mode_counts'].values.flatten()

### Select relevant columns and prepare to convert to xarray dataset
df_final = df[['lat','lon','mode']].set_index(['lat','lon'])

### Convert to xarray dataset
ds = df_final.to_xarray()

# Extract latitudes and longitudes
lats = ds.lat.values
lons = ds.lon.values

### Set up plot
fig, ax = plt.subplots(figsize=(11, 6), 
                       subplot_kw={'projection': ccrs.PlateCarree()})

### Potential count of GCMs to indicate uncertainty
GCM_count = [1,2,3,4,5,6,7,8,9]

### Set colorbar
cmap_cbar = 'inferno_r'
cmap_colors = plt.cm.get_cmap(cmap_cbar,len(GCM_count))
colors = list(cmap_colors(np.arange(len(GCM_count))))
cmap = mpl.colors.ListedColormap(colors[:], '')

### Plot map
im = ax.pcolormesh(lons,
                   lats,
                   ds['mode'].values,
                   cmap=cmap,
                   vmin=1,vmax=9,
                   rasterized=True,
                   transform=ccrs.PlateCarree())

'''
Set the extent/ padding: 
subtract 0.05 from the first latitude and longitude
add 0.05 to the last latitude and longitude 
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

plt.tight_layout()
plt.savefig('figures/map_fuel_types_mode_'+scen+'_'+timespan+'_uncertainty.jpg',
            dpi=1000)
