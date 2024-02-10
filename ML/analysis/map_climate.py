import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import CenteredNorm
import argparse

### For state masking
import geopandas as gpd
import odc.geo.xr
import rasterio

'''
Initialise argument parsing: 
ensstat for ens statistic ('' for mean, 'CV' for coefficient of variation)
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--ensstat', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
ensstat = args.ensstat
scen = args.scen
timespan = args.timespan

### Set pathway
pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/input/clim/')

### Read in variables
def read_in_var(var,scen,timespan,ensstat):
    global pathway

    ### Adjust for file names
    if timespan == 'mid':
        timespan='20452060'
    elif timespan == 'long':
        timespan='20852100'

    ### Ensemble averages
    if ensstat != 'CV':
        ds_fut = xr.open_dataset(pathway+scen+'_'+timespan+'_'+var+'.nc')
        ds_hist = xr.open_dataset(pathway+'history_20002015_'+var+'.nc')
        diff = ds_fut.layer - ds_hist.layer
        return(diff)

    ### Ensemble coefficient of variation
    else:
        ds_fut = xr.open_dataset(pathway+scen+'_'+timespan+'_'+var+'_CV.nc')
        return(ds_fut.layer)        

### Open shape file with state boundaries: 1 is Victoria
gdf = gpd.read_file('shape_files/STE_2021_AUST_GDA2020.shp')
state = gdf.iloc[[1]]

### Create state mask for Victoria
def create_mask(index):
    state = gdf.iloc[[index]].to_crs(crs='EPSG:4326')
    state_mask = rasterio.features.geometry_mask(state.geometry,
                                            out_shape=da.odc.geobox.shape,
                                            transform=da.odc.geobox.affine,
                                            all_touched=False,
                                            invert=False)

    state_mask = xr.DataArray(state_mask, dims=('latitude', 'longitude'))    
    return(state_mask)    

### Make mask
da = read_in_var('annual_tmax',scen,timespan,'')
mask = create_mask(1)
da_mask = da.where(~mask)

### Individual subplot in panel plot
def make_plot(var,scen,timespan,ensstat,label,ax,title,cmap):
    global mask

    ### Get data
    da = read_in_var(var,scen,timespan,ensstat)

    ### Apply Vic mask
    da_mask = da.where(~mask)

    if ensstat != 'CV':
        norm = None if var == 'annual_tmax' else CenteredNorm()
    else:
        norm = CenteredNorm()

    c = ax.contourf(da_mask.longitude,
                    da_mask.latitude,
                    da_mask.values,
                    levels=10,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    norm=norm
                    )
         
    cbar = plt.colorbar(c, 
                        ax=ax, 
                        orientation='horizontal', 
                        label=label,
                        fraction=0.1, 
                        pad=0.05)
    
    ax.set_title(title,loc='left')

    ### Add state borders
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '10m',
        edgecolor='k', facecolor='none',
        linewidth=1.0, linestyle='solid'))

    ax.coastlines()

    ### Adding state borders looks a bit ugly in NW where vertical state borders 
    ### of NSW and Vic are offset, also in SE unnecessary border. Then ACT is this 
    ### random blob - so added patches to cover those areas (probably easier getting
    ### Victoria shape from the shape file)

    ### NSW border SE
    ax.add_patch(mpatches.Rectangle(xy=[149.5,-37.313],width=0.6,height=0.85,
                                    facecolor='w',
                                    zorder=12,
                                    angle=338,
                                    transform=ccrs.PlateCarree())
                            )

    ### NSW border NW
    ax.add_patch(mpatches.Rectangle(xy=[140.992,-34.005],width=0.02,height=0.08,
                                    facecolor='w',
                                    zorder=12,
                                    transform=ccrs.PlateCarree())
                            )

    ### ACT
    ax.add_patch(mpatches.Rectangle(xy=[148.7,-35.95],width=0.8,height=0.85,
                                    facecolor='w',
                                    zorder=12,
                                    transform=ccrs.PlateCarree())
                            )
    
    ax.set_extent([140.9,150.0,-39.2,-33.9], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.axis('off')

### Create the figure and grid specification
def set_up_figure(scen,timespan,ensstat,cmap_list):
    fig = plt.figure(figsize=(8,8),layout='constrained')
    spec = fig.add_gridspec(3, 2)

    make_plot('annual_tmax',
              scen,
              timespan,
              ensstat,
              '$\Delta$ T$_{max}$ [K]',
              fig.add_subplot(spec[0, 0], 
                              projection=ccrs.PlateCarree()), 
              'a) Maximum temperature', 
              cmap_list[0])

    make_plot('annual_pr',
              scen,
              timespan,
              ensstat,
              '$\Delta$ MAP [mm]',
              fig.add_subplot(spec[0, 1], 
                              projection=ccrs.PlateCarree()), 
              'b) Mean annual precipitation', 
              cmap_list[1])

    make_plot('pr_seasonality',
              scen,
              timespan,
              ensstat,
              '$\Delta$ Precipitation seasonality [-]',
              fig.add_subplot(spec[1, 0], 
                              projection=ccrs.PlateCarree()), 
              'c) Precipitation seasonality', 
              cmap_list[2])

    make_plot('annual_rh',
              scen,
              timespan,
              ensstat,
              '$\Delta$ RH$_{min}$ [%]',
              fig.add_subplot(spec[1, 1], 
                              projection=ccrs.PlateCarree()), 
              'd) Minimum relative humidity', 
              cmap_list[3])

    make_plot('lai_jan_5km',
              scen,
              timespan,
              ensstat,
              '$\Delta$ LAI$_{opt}$ [m$^2$m$^{-2}$]',
              fig.add_subplot(spec[2, :], 
                              projection=ccrs.PlateCarree()), 
              'e) LAI', 
              cmap_list[4])

    if ensstat == 'CV':
        plt.savefig('figures/map_climate_'+scen+'_'+timespan+'_CV.png',dpi=500)
    else:
        plt.savefig('figures/map_climate_'+scen+'_'+timespan+'.png',dpi=500)

cmap_list_mean=['YlOrRd','BrBG','BrBG','BrBG','BrBG']
cmap_list_CV=['magma_r','magma_r','magma_r','magma_r','magma_r']

# Create the figure and grid specification
if ensstat == 'CV':
    cmap_list = cmap_list_CV
else:
    cmap_list = cmap_list_mean

set_up_figure(scen,timespan,ensstat,cmap_list)
