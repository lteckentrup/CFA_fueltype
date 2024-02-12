import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import argparse

'''
Initialise argument parsing: 
var for variable (in report I only used FPC for this figure)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)

args = parser.parse_args()

### Assign variables
var = args.var

### Make map of vegetation distribution based on FPC
def make_map(var,PFT,position):
    ds = xr.open_dataset('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
                         'dynamics_simulations/CFA/LPJ-GUESS/output/netCDF/'
                         'eval_ERA5-Land/'+var+'_1950-2022_mask.nc')
    
    ### Get average over last 20 years of reanalysis simulation
    ds_mean = ds.sel(time=slice('2003','2021')).mean(dim='time')

    ### Get percentage of FPC per PFT relative to total FPC
    da = (ds_mean[PFT]/ds_mean['Total'])*100

    ### Get latitude and longitude info
    lat, lon = da.lat, da.lon

    ### Set projection
    projection = ccrs.PlateCarree()

    ### Set levels for colorbar
    levels = [0,1,2,3,4,5,10,20,30,40,50,100]

    ### Set colors for colorbar
    cols = ['#d3d3d3', '#FDE725FF', '#BBDF27FF', '#7AD151FF', 
            '#43BF71FF', '#22A884FF', '#21908CFF', '#2A788EFF', 
            '#35608DFF', '#414487FF', '#482576FF', '#440154FF']
    cmap = ListedColormap(cols)

    ### Set boundary norm
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=len(cols))

    ### Plot the bad boi
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, norm=norm)

    ### Add state borders
    states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

    axs[position].add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '10m',
        edgecolor='k', facecolor='none',
        linewidth=1.0, linestyle='solid'))

    axs[position].coastlines()

    ### Mask some shizzle
    ### NSW border SE
    axs[position].add_patch(mpatches.Rectangle(xy=[149.5,-37.313],
                                               width=0.6,height=0.85,
                                               facecolor='w',
                                               zorder=12,
                                               angle=338,
                                               transform=ccrs.PlateCarree())
                                               )

    ### NSW border NW
    axs[position].add_patch(mpatches.Rectangle(xy=[140.992,-34.005],
                                               width=0.02,height=0.08,
                                               facecolor='w',
                                               zorder=12,
                                               transform=ccrs.PlateCarree())
                                               )

    ### ACT
    axs[position].add_patch(mpatches.Rectangle(xy=[148.7,-35.95],
                                        width=0.8,height=0.85,
                                        facecolor='w',
                                        zorder=12,
                                        transform=ccrs.PlateCarree())
                                        )
    
    ### Padding
    axs[position].set_extent([140.9,150.0,-39.2,-33.9], 
                      crs=ccrs.PlateCarree())

    ### Drop spines
    axs[position].spines['geo'].set_visible(False)

    ### Reintroduce left and bottom spine
    axs[position].spines['left'].set_visible(True)
    axs[position].spines['bottom'].set_visible(True)

    ### Set up colorbar
    cax = plt.axes([0.02, 0.08, 0.96, 0.035])
    cbar = fig.colorbar(p, cax=cax, ticks=levels, orientation='horizontal',
                        extend='neither', label='Percentage of total FPC [%]')

### Set up figure
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(10,7))
axs=axs.flatten()

### Make lists for magic for loop
PFT_short_names=['CRT','ST','SM','SS','SuS','MS','XS','C3','C4']
PFT_long_names=['Cool rainforest','Tall sclerophyll','Medium sclerophyll',
                'Short sclerophyll','Subalpine sclerophyll','Mesic shrubs',
                'Xeric shrubs','C$_3$ grasses','C$_4$ grasses']
positions=[0,1,2,3,4,5,6,7,8]
title_index=['a)','b)','c)','d)','e)','f)','g)','h)','i)']

### Loop through plot command, and adjust subplot titles
for PFTs, PFTl,p,ti in zip(PFT_short_names, PFT_long_names,positions,title_index):
    make_map(var,PFTs,p)
    axs[p].set_title(PFTl)
    axs[p].set_title(ti, loc='left')

### Adjust ticklabels on axes
for p in (0,3,6):
    ### Show ticklabels left
    axs[p].yaxis.set_visible(True)

for p in (6,7,8):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)

### Adjust plot layour
plt.subplots_adjust(top=0.95, left=0.05, right=0.975, bottom=0.15,
                    wspace=0.2, hspace=0.1)

plt.savefig('figures/map_'+var+'_PFT_dist.png',dpi=400)
