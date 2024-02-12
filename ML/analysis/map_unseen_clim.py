import numpy as np
import pandas as pd
import xarray as xr
import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

### For state masking
import geopandas as gpd
import odc.geo.xr
import rasterio
from rasterio import features

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

### Open shape file with state boundaries: 1 is Victoria
gdf = gpd.read_file('shape_files/STE_2021_AUST_GDA2020.shp')
state = gdf.iloc[[1]]

### Create state mask for Victoria
def create_mask(index,GCM,scen,timespan):
    ### Open target netCDF for grid info
    ds = xr.open_dataset('netCDF/'+GCM+'_'+scen+'_'+timespan+'.nc')
    state = gdf.iloc[[index]].to_crs(crs='EPSG:4326')
    state_mask = rasterio.features.geometry_mask(state.geometry,
                                                 out_shape=ds.odc.geobox.shape,
                                                 transform=ds.odc.geobox.affine,
                                                 all_touched=False,
                                                 invert=False)

    state_mask = xr.DataArray(state_mask, dims=('latitude', 'longitude'))    
    return(state_mask)    

### Plot 'unseen climate' variables for individual GCMs
def make_map_GCM(GCM,scen,timespan,position):
    ### Make Victoria mask
    mask = create_mask(1,GCM,scen,timespan)

    ### Open dataset
    ds = xr.open_dataset('netCDF/'+GCM+'_'+scen+'_'+timespan+'.nc')

    ### Mask variable
    ds_masked = ds.where(~mask)

    ### Get DataArray
    da = ds_masked['out_of_range']
    print(np.unique(da.values.flatten()))

    ### Get latitude and longitude info
    lat, lon = ds_masked.latitude, ds_masked.longitude

    ### Set projection
    projection = ccrs.PlateCarree()

    '''
    Which combinations do not occur?
    RCP4.5 mid: 7,8,9,10,12,13,14,15 do not occur
    RCP4.5 long 7,12,13,15 do not occur

    RCP8.5 mid: 7,8,9,10,12,13,14,15 do not occur
    RCP8.5 long: 3,8,9,10,12,13,14,15 do not occur
    
    Across both RCP scenarios and timespans, the following combinations can
    occur
    levels = [0,1,2,3,4,5,6,7,8,9,10,11,14]

    and I chose the following colormap for each level specifically

    cols = ['#d3d3d3','#ebac23','#b80058','#008cf9','#006e00','#00bbad',
            '#d163e6','#b24502','#ff9287','#5954d6','#00c6f8','#878500',
            '#796880','#00a76c','#c0affb']
    '''
    
    ### RCP4.5 and RCP8.5 2045 - 2060 have the same variable combinations
    ### Select indices + respective colors; reset tick locs
    if ((scen == 'rcp45' or scen == 'rcp85') and timespan == 'mid'):
        levels = [0,1,2,3,4,5,6,11]
        cols = ['#d3d3d3', '#ebac23', '#b80058', '#008cf9', 
                '#006e00', '#00bbad', '#d163e6', '#878500']
        tick_locs = [0.5,1.5,2.5,3.5,4.5,5.5,8.5,11.5]

    ### Select indices + respective colors; reset tick locs for RCP4.5 2085-2100
    elif (scen == 'rcp45' and timespan == 'long'):
        levels = [0,1,2,3,4,5,6,8,9,10,11,14]
        cols = ['#d3d3d3', '#ebac23', '#b80058', '#008cf9', '#006e00', 
                '#00bbad', '#d163e6', '#ff9287', '#5954d6', '#00c6f8', 
                '#878500', '#c0affb']
        tick_locs = [0.5,1.5,2.5,3.5,4.5,5.5,7,8.5,9.5,10.5,12,14.5]

    ### Select indices + respective colors; reset tick locs for RCP8.5 2085-2100
    elif (scen == 'rcp85' and timespan == 'long'):
        levels = [0,1,2,4,5,6,7,11]
        cols = ['#d3d3d3', '#ebac23', '#b80058', '#006e00', 
                '#00bbad', '#d163e6', '#b24502', '#878500']
        tick_locs = [0.5,1.5,3,4.5,5.5,6.5,9,11.5]

    ### Set boundary norm
    bounds = np.append(levels, levels[-1] + 1)
    norm = BoundaryNorm(bounds, ncolors=len(cols))

    ### Generate colormap
    cmap = ListedColormap(cols)
    
    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, norm=norm)

    ### Set subplot title
    axs[position].set_title(GCM,fontsize=10)

    ### Plot colorbar
    cax = plt.axes([0.15, 0.075, 0.02, 0.85])
    cbar = fig.colorbar(p, cax=cax, ticks=levels, orientation='vertical',
                        extend='neither', ticklocation='left') 
    cbar.set_ticks(tick_locs)
    cbar.ax.invert_yaxis()

    ### Update ticklabels for colorbar
    if ((scen == 'rcp45' or scen == 'rcp85') and timespan == 'mid'):
        cbar.ax.set_yticklabels(['', 'T', 'Pr', 'RH', 'Pr$_{seas}$', 'T+Pr', 
                                 'T+RH', 'T+PR+RH'],fontsize=10)

    elif (scen == 'rcp45' and timespan == 'long'):
        levels = [0,1,2,3,4,5,6,8,9,10,11,14]
        cbar.ax.set_yticklabels(['', 'T', 'Pr', 'RH', 'Pr$_{seas}$', 'T+Pr', 
                                 'T+RH', 'Pr+RH', 'Pr+Pr$_{seas}$', 
                                 'RH+Pr$_{seas}$', 'T+PR+RH', 
                                 'T+RH+Pr$_{seas}$'],fontsize=10)
    elif (scen == 'rcp85' and timespan == 'long'):
        levels = [0,1,2,4,5,6,7,11]
        cbar.ax.set_yticklabels(['', 'T', 'Pr', 'Pr$_{seas}$', 'T+Pr', 'T+RH', 
                                 'T+Pr$_{seas}$','T+PR+RH'],fontsize=10)

### Read in 'unseen climate' data and apply Vic mask
def get_unseen_clim(GCM,scen,timespan):
    mask = create_mask(1,GCM,scen,timespan)
    ds = xr.open_dataset('netCDF/'+GCM+'_'+scen+'_'+timespan+'.nc')
    ds_masked = ds.where(~mask)
    da = ds_masked['out_of_range']

    ### Binary map: set to one when 'unseen climate', 0 when not
    da = xr.where(da > 0, 1, da)
    return(da)

### Plot GCM ensemble: How many GCMs project unseen climate in a pixel
def make_map_ens(scen,timespan,position):    
    da = get_unseen_clim('ACCESS1-0',scen,timespan) + \
         get_unseen_clim('BNU-ESM',scen,timespan) + \
         get_unseen_clim('CSIRO-Mk3-6-0',scen,timespan) + \
         get_unseen_clim('GFDL-CM3',scen,timespan) + \
         get_unseen_clim('GFDL-ESM2G',scen,timespan) + \
         get_unseen_clim('GFDL-ESM2M',scen,timespan) + \
         get_unseen_clim('INM-CM4',scen,timespan) + \
         get_unseen_clim('IPSL-CM5A-LR',scen,timespan) + \
         get_unseen_clim('MRI-CGCM3',scen,timespan)
    
    ### Get latitude and longitude info
    lat, lon = da.latitude, da.longitude

    ### Set projection
    projection = ccrs.PlateCarree()

    ### Set levels: 0 - 9 (Ensemble has 9 GCMs)
    levels = np.arange(0,10,1)

    ### Set boundary norm
    bounds = np.append(levels, levels[-1] + 1)
    norm = BoundaryNorm(bounds, ncolors=len(levels))

    ### Set colorcolormap
    pal = sns.color_palette('magma_r', len(levels[1:]))
    cols = pal.as_hex()

    ### Set 0 to lightgrey
    cols.insert(0,'#d3d3d3')
 
    ### Generate colormap
    cmap = ListedColormap(cols)
    
    ### Plot map
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, norm=norm)

    ### Set subplot title
    axs[position].set_title('Ensemble',fontsize=10)

    ### Plot colorbar
    cax = plt.axes([0.93, 0.05, 0.02, 0.15])
    cbar = fig.colorbar(p, cax=cax, ticks=levels, orientation='vertical',
                        extend='neither')
    cbar.ax.set_title('# GCMs',fontsize=10)

    ### Update ticklabels for colorbar
    tick_locs = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
    cbar.set_ticks(tick_locs)
    cbar.ax.yaxis.set_minor_locator(plt.NullLocator())
    cbar.ax.set_yticklabels(levels)
    cbar.ax.invert_yaxis()

### Set up figure
fig, axs = plt.subplots(nrows=5,ncols=2,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(7,8))
axs=axs.flatten()

### Get ready for some fancey for loops
GCMs=['ACCESS1-0','BNU-ESM','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G',
      'GFDL-ESM2M','INM-CM4','IPSL-CM5A-LR','MRI-CGCM3']
positions=[0,1,2,3,4,5,6,7,8,9]
labels=['a)','b)','c)',
        'd)','e)','f)',
        'g)','h)','i)',
        'k)']

### Plot subplots
for GCM, p in zip(GCMs, positions):
    make_map_GCM(GCM,scen,timespan,p)

make_map_ens(scen,timespan,9)

### Add state borders
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

for p,l in zip(positions,labels):
    ### Add state borders
    axs[p].add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '10m',
        edgecolor='k', facecolor='none',
        linewidth=1.0, linestyle='solid'))

    axs[p].coastlines()

    ### Mask some shizzle
    ### NSW border SE
    axs[p].add_patch(mpatches.Rectangle(xy=[149.5,-37.313],width=0.6,height=0.85,
                                    facecolor='w',
                                    zorder=12,
                                    angle=338,
                                    transform=ccrs.PlateCarree())
                            )

    ### NSW border NW
    axs[p].add_patch(mpatches.Rectangle(xy=[140.992,-34.005],width=0.02,height=0.08,
                                    facecolor='w',
                                    zorder=12,
                                    transform=ccrs.PlateCarree())
                            )

    ### ACT
    axs[p].add_patch(mpatches.Rectangle(xy=[148.7,-35.95],width=0.8,height=0.85,
                                    facecolor='w',
                                    zorder=12,
                                    transform=ccrs.PlateCarree())
                            )
    
    ### Padding
    axs[p].set_extent([140.9,150.0,-39.2,-33.9], 
                      crs=ccrs.PlateCarree())

    ### Drop spines
    axs[p].spines['geo'].set_visible(False)

    ### Reintroduce left and bottom spine
    axs[p].spines['left'].set_visible(True)
    axs[p].spines['bottom'].set_visible(True)

    axs[p].set_title(l,fontsize=10,loc='left')

for p in (0,2,4,6,8):
    ### Show ticklabels left
    axs[p].yaxis.set_visible(True)

for p in (8,9):
    ### Show ticklabels bottom
    axs[p].xaxis.set_visible(True)

plt.subplots_adjust(top=0.95, left=0.25, 
                    right=0.9, bottom=0.05,
                    wspace=0.08, hspace=0.2)

# plt.show()
plt.savefig('figures/map_unseen_clim_'+scen+'_'+timespan+'.png',dpi=400)
