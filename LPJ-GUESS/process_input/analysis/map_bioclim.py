import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from  matplotlib.colors import ListedColormap, BoundaryNorm

### For state masking
import geopandas as gpd
import odc.geo.xr
import rasterio
from rasterio import features

import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches

### Open shape file with state boundaries: 1 is Victoria
gdf = gpd.read_file('/g/data/w97/lt0205/HIE/FIRMS/FIRMS/STE_2021_AUST_GDA2020.shp')
state = gdf.iloc[[1]]

### Open file with species occurrence
df_occ = pd.read_csv('aus_flora_occurrence_corrected_euc_vic.csv')

### Create geopandas dataframe from occurence df
gdf_occ = gpd.GeoDataFrame(df_occ, geometry=gpd.points_from_xy(df_occ['longitude'], 
                                                               df_occ['latitude']))

### Mask values outside of Victoria in occurrence dataframe
gdf_vic = gpd.sjoin(gdf_occ, state, op='within')

### Select columns of interest from occurrence dataframe
df_vic = gdf_vic[['taxon','longitude','latitude']]

df_PFT = pd.read_csv('../CFA_PFT/PFT_classes.csv')

### Get list of species per PFT
def grab_trait(PFT_index):
    ### Get a list of all CFA PFTs
    PFT_list = df_PFT.PFT.drop_duplicates().to_list()

    ### Select PFT based on index
    PFT = PFT_list[PFT_index]
    PFT_taxa = df_PFT[df_PFT['PFT'] == PFT].taxon.to_list()

    ### Print which PFT
    print(PFT)

    ### Select PFT species in occurrence list (which has the latitude and longitude information)
    PFT_occ = df_occ[df_occ['taxon'].isin(PFT_taxa)]

    return(PFT,PFT_occ) 

### Create state mask for Victoria
def create_mask(index):
    ds = xr.open_dataset('../../ERA-Land/2t/tcmax_est.nc')
    state = gdf.iloc[[index]].to_crs(crs='EPSG:4326')
    state_mask = rasterio.features.geometry_mask(state.geometry,
                                                 out_shape=ds.odc.geobox.shape,
                                                 transform=ds.odc.geobox.affine,
                                                 all_touched=False,
                                                 invert=False)

    state_mask = xr.DataArray(state_mask, dims=('latitude', 'longitude'))    
    return(state_mask)    

def make_map(index,position):
    mask = create_mask(1)
    ds = xr.open_dataset('../../ERA-Land/2t/tcmin_est.nc')
    ds_masked = ds.where(~mask)
    lat, lon = ds_masked.latitude, ds_masked.longitude

    projection = ccrs.PlateCarree()

    p = axs[position].contourf(lon, 
                               lat, 
                               ds_masked['temp'][0,:,:]-273.15, 
                               levels=10,
                               add_labels=True,
                               cmap='coolwarm', 
                               transform=projection)
    axs[position].axis('off')
    axs[position].coastlines()
    axs[position].set_title('bla')

    cax = plt.axes([0.02, 0.06, 0.96, 0.035])
    cbar = fig.colorbar(p, cax=cax, orientation='horizontal',
                        extend='neither', 
                        label='Minimum 20-year coldest month mean temperature [$^{\circ}$C]')

    p = axs[position].contour(lon, 
                              lat, 
                              ds_masked['temp'][0,:,:],
                              levels=10,
                              colors='black', 
                              linewidths=0.5,
                              transform=projection)
    
    PFT, PFT_coords = grab_trait(index)
    
    ### Add coastline
    axs[position].add_feature(cfeature.COASTLINE)

    ### Add scatter plot of species occurrence
    axs[position].scatter(PFT_coords['longitude'], 
                          PFT_coords['latitude'], 
                          color='k',
                          s=1,
                          transform=projection)

    axs[position].axis('off')
    
fig, axs = plt.subplots(nrows=3,ncols=3,
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        figsize=(10,7))
axs=axs.flatten()

states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')

positions = [0,1,2,3,4,5,6,7,8]
labels = ['a)','b)','c)',
          'd)','e)','f)',
          'g)','h)','i)']

make_map(8,0)
make_map(13,1)
make_map(12,2)
make_map(9,3)
make_map(10,4)
make_map(1,5)
make_map(3,6)
make_map(2,7)
make_map(7,8)

for p,l in zip(positions,labels):
    axs[p].set_title(l,loc='left')
    axs[p].add_patch(mpatches.Rectangle(xy=[149.5, -37.3], width=25, height=1,
                                        facecolor='w',
                                        zorder=12,
                                        angle=340,
                                        transform=ccrs.PlateCarree())
                            )
    axs[p].add_feature(states_provinces,edgecolor='black',linewidth=0.5)
    axs[p].set_extent([140.9, 150.0, -39.2, -33.9], crs=ccrs.PlateCarree())
    axs[p].add_feature(cfeature.COASTLINE,linewidth=0.5)
    axs[p].axis('off')

axs[0].set_title('Cool rainforest')
axs[1].set_title('Subalpine')
axs[2].set_title('Tall sclerophyll')
axs[3].set_title('Medium sclerophyll')
axs[4].set_title('Short sclerophyll')
axs[5].set_title('Mesic shrub')
axs[6].set_title('Xeric shrub')
axs[7].set_title('C$_3$ grass')
axs[8].set_title('C$_4$ grass')

plt.suptitle('PFT occurrence')
plt.subplots_adjust(top=0.9, left=0.025, right=0.975, bottom=0.1,
                    wspace=0.08, hspace=0.2)

# plt.show()
plt.savefig('PFT_occ_tcmin_est.png',dpi=400)
