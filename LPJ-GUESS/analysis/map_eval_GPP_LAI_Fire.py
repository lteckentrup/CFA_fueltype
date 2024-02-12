import numpy as np
import seaborn as sns
import xarray as xr

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

### Set up function to plot map
def make_map(var,config,position,source):

    ### Read in data: If data are based on LPJ-GUESS then
    if source == 'LPJ':
        if var == 'agpp':
            ds = xr.open_dataset('../'+config+'/agpp_timmean.nc')
            da = ds.Total[0]
            levels = np.arange(0,2.2,0.2)
        elif var == 'mlai':
            ds = xr.open_dataset('../'+config+'/mlai_timmean.nc')
            da = ds.mlai[0]
            levels = np.arange(0,5.5,0.5) 
        elif var == 'Fire':
            ds = xr.open_dataset('../'+config+'/Fire_timmean.nc')
            da = ds['Fire'][0]
            levels = [0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
            ncols=11
        elif var == 'BurntFr':
            ds = xr.open_dataset('../'+config+'/annual_burned_area_timmean.nc')
            da = ds['BurntFr'][0]
            levels = [0,0.01,0.02,0.05,0.1,0.2,0.5]
            ncols=9

    ### Read in data: If data are based on remote sensing or similar, then:
    else:
        if var == 'agpp':
            ds = xr.open_dataset('AusEFlux_GPP_timmean.nc')
            levels = np.arange(0,2.2,0.2)
            da = ds[var][0]
        elif var == 'mlai':
            ds = xr.open_dataset('MODIS_LAI_timmean.nc')
            levels = np.arange(0,5.5,0.5)
            da = ds[var][0]
        elif var == 'Fire':
            ds = xr.open_dataset('CAMS-GFAS_fire_CO2_timmean.nc')
            da = ds['co2fire'][0]/12
            levels = [0,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2]
            ncols=11
        elif var == 'BurntFr':
            ds = xr.open_dataset('Fire_CCI_BF_timmean.nc')
            da = ds['burned_area'][0]/100
            levels = [0,0.01,0.02,0.05,0.1,0.2,0.5]
            ncols=9

    ### Get latitude and longitude info
    lat, lon = da.lat, da.lon

    ### Set projection
    projection = ccrs.PlateCarree()

    ### Set colors for colorbar
    if var in ('agpp','mlai'):
        cols = ['#d3d3d3', '#FDE725FF', '#BBDF27FF', '#7AD151FF', 
                '#43BF71FF', '#22A884FF', '#21908CFF', '#2A788EFF', 
                '#35608DFF', '#414487FF', '#482576FF', '#440154FF']
    else:
        pal = sns.color_palette('inferno_r', ncols)
        cols = pal.as_hex()
        cols.insert(0,'#d3d3d3')
    cmap = ListedColormap(cols)

    ### Set boundary norm
    bounds = np.append(levels, levels[-1] + 1)
    norm = BoundaryNorm(bounds, ncolors=len(cols))  

    ### Plot the bad boi
    p = axs[position].pcolormesh(lon, lat, da, cmap=cmap, norm=norm)

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

    ### Plot individual colorbars
    if position == 1:
        cbar = fig.colorbar(p, ticks=levels, orientation='horizontal',
                            extend='max', fraction=0.1, pad=0.15,shrink=2.5,
                            label='GPP [kgC m$^{-2}$ yr$^{-1}$]')
    if position == 4:
        cbar = fig.colorbar(p, ticks=levels, orientation='horizontal',
                            extend='max', fraction=0.1, pad=0.15,shrink=2.5,
                            label='LAI [m$^{2}$ m$^{-2}$]')

    if position == 7:
        cbar = fig.colorbar(p, ticks=levels, orientation='horizontal',
                            extend='neither', fraction=0.1, pad=0.15,shrink=2.5,
                            label='Burned area fraction [-]')
    if position == 10:
        cbar = fig.colorbar(p, ticks=levels, orientation='horizontal',
                            extend='max', fraction=0.1, pad=0.15,shrink=2.5, 
                            aspect=25,
                            label='Fire CO$_2$ [kgC m$^{-2}$ yr$^{-1}$]')
        
        cbar.set_ticks(levels)
        cbar.ax.set_xticklabels(['0','0.001','0.002','0.005',
                                 '0.01','0.02','0.05','0.1','0.2'])

### Set up figure           
fig, axs = plt.subplots(nrows=4,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(8,12))

axs=axs.flatten()

### Plot GPP comparison
var='agpp'
make_map(var,'',0,'True','') ### AusElux
make_map(var,'Vic_default',1,'True','LPJ') ### LPJ global config
make_map(var,'Vic_final',2,'True','LPJ') ### LPJ CFA config

### Plot LAI comparison: Take mlai (annual LAI is maximum LAI)
var='mlai'
make_map(var,'',3,'True','') ### MODIS
make_map(var,'Vic_default',4,'True','LPJ') ### LPJ global config
make_map(var,'Vic_final',5,'True','LPJ') ### LPJ CFA config

### Plot burned fraction comparison
var='BurntFr'
make_map(var,'',6,'True','') ### FireCCI
make_map(var,'Vic_default',7,'True','LPJ') ### LPJ global config
make_map(var,'Vic_final',8,'True','LPJ') ### LPJ CFA config

### Plot fire CO2 emission comparison
var='Fire'
make_map(var,'',9,'True','') ### CAMS-GFAS
make_map(var,'Vic_default',10,'True','LPJ') ### LPJ global config
make_map(var,'Vic_final',11,'True','LPJ') ### LPJ CFA config

### Set subplot titles
fontsize=11
axs[0].set_title('GPP AusEFlux',fontsize=fontsize)
axs[1].set_title('GPP global config.',fontsize=fontsize)
axs[2].set_title('GPP VFA config.',fontsize=fontsize)

axs[3].set_title('LAI MODIS',fontsize=fontsize)
axs[4].set_title('LAI global config.',fontsize=fontsize)
axs[5].set_title('LAI VFA config.',fontsize=fontsize)

axs[6].set_title('BAF FireCCI',fontsize=fontsize)
axs[7].set_title('BAF global config.',fontsize=fontsize)
axs[8].set_title('BAF VFA config.',fontsize=fontsize)

axs[9].set_title('Fire CO$_2$ CAMS-GFAS',fontsize=fontsize)
axs[10].set_title('Fire CO$_2$ global config.',fontsize=fontsize)
axs[11].set_title('Fire CO$_2$ VFA config.',fontsize=fontsize)

axs[0].set_title('a)', loc='left')
axs[1].set_title('b)', loc='left')
axs[2].set_title('c)', loc='left')
axs[3].set_title('d)', loc='left')
axs[4].set_title('e)', loc='left')
axs[5].set_title('f)', loc='left')
axs[6].set_title('g)', loc='left')
axs[7].set_title('h)', loc='left')
axs[8].set_title('i)', loc='left')
axs[9].set_title('j)', loc='left')
axs[10].set_title('k)', loc='left')
axs[11].set_title('l)', loc='left')

### Adjust plot layout
plt.subplots_adjust(top=0.99, left=0.025, right=0.975, bottom=0.07,
                    wspace=0.08, hspace=0.1)

#plt.show()
plt.savefig('figures/map_eval_GPP_LAI_Fire.png',dpi=400)

