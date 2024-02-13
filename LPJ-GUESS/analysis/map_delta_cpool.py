import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

import argparse

'''
Initialise argument parsing: 
scen for RCP scenario (rcp45 or rcp85)
first_year and last_year for fut. projection timeslice (2045-2059 or 2085-2099)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--first_year', type=str, required=True)
parser.add_argument('--last_year', type=str, required=True)

args = parser.parse_args()

### Assign variables
scen=args.scen
first_year=args.first_year
last_year=args.last_year

### Set pathway where input files are located
pathwayIN = ('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
             'dynamics_simulations/CFA/LPJ-GUESS/')

def get_data(var,first_year,last_year,GCM,scen):
    global pathwayIN

    ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+
                         GCM+'_'+scen+'/'+var+'_1960-2099.nc')

    ### Get future projection
    ds_fut = ds.sel(time=slice(first_year,last_year)).mean(dim='time')

    ### Get historical
    ds_hist = ds.sel(time=slice('2000','2014')).mean(dim='time')

    ### Show cpools as simple difference
    diff = ds_fut['Total'] - ds_hist['Total']

    return(diff) 

### Set up plot for maps
def make_map(var,first_year,last_year,scen,position):

    ### Get ensemble average of change in different cpools
    diff = (get_data(var,first_year,last_year,'CNRM-CERFACS-CNRM-CM5',scen)+ \
            get_data(var,first_year,last_year,'CSIRO-BOM-ACCESS1-0',scen)+ \
            get_data(var,first_year,last_year,'MIROC-MIROC5',scen)+ \
            get_data(var,first_year,last_year,'NOAA-GFDL-GFDL-ESM2M',scen))/4
    
    ### Set levels for colorbar
    if var in ('cmass', 'cmass_wood'):
        levels = [-5,-2,-1,-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,
                  0.01,0.02,0.05,0.1,0.2,0.5,1,2,5]

    elif var == 'clitter':
        levels = [-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,-0.005,-0.002,-0.001,
                  0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]
    else:
        levels = [-0.5,-0.2,-0.1,-0.05,-0.02,-0.01,-0.005,-0.002,-0.001,
                  0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5]

    ### Set colorbar
    pal = sns.color_palette('BrBG', 18)
    cols = pal.as_hex()
    cols[8] = '#d3d3d3'
    cmap = ListedColormap(cols)

    ### Get latitude and longitude coordinates
    lat, lon = diff.lat, diff.lon

    ### Set projection
    projection = ccrs.PlateCarree()   

    ### Set boundary norm
    bounds = levels
    norm = BoundaryNorm(bounds, ncolors=len(cols))

    ### Plot map
    p = axs[position].pcolormesh(lon, lat, diff, cmap=cmap, norm=norm)

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
    if var == 'clitter':
        cax = plt.axes([0.1, 0.08, 0.8, 0.035])
        cbar = fig.colorbar(p, 
                            cax=cax, 
                            ticks=levels, 
                            orientation='horizontal',
                            extend='neither', 
                            label='$\Delta$ Carbon pool [kgC m$^{-2}$]')
    if var == 'cmass':
        cax = plt.axes([0.1, 0.58, 0.8, 0.035])
        cbar = fig.colorbar(p, 
                            cax=cax, 
                            ticks=levels, 
                            orientation='horizontal',
                            extend='neither', 
                            label='$\Delta$ Carbon pool [kgC m$^{-2}$]')
            
fig, axs = plt.subplots(nrows=2,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(10,7))
axs=axs.flatten()

var_short_names = ['cmass','cmass_wood','cmass_leaf','clitter']
var_titles = ['Carbon stored in vegetation','Carbon stored in wood',
              'Carbon stored in leaves','Carbon stored in litter']
positions=[0,1,2,3]
title_index=['a)','b)','c)','d)']

### Loop through plot command, and adjust subplot titles
for vars, vart,p,ti in zip(var_short_names, var_titles,positions,title_index):
    make_map(vars,first_year,last_year,scen,p)
    axs[p].set_title(vart)
    axs[p].set_title(ti, loc='left')

### Adjust ticklabels on axes
for p in (0,2):
    ### Show ticklabels left
    axs[p].yaxis.set_visible(True)

for p in (2,3):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)

plt.subplots_adjust(top=0.95, left=0.025, 
                    right=0.975, bottom=0.15,
                    wspace=0.08, hspace=0.5)

plt.savefig('figures/map_delta_cpool_'+scen+'_'+first_year+'-'+
            last_year+'.png',dpi=400)
