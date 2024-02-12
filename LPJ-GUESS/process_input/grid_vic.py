import numpy as np
import pandas as pd
from decimal import Decimal

### For state masking
import geopandas as gpd
import odc.geo.xr
import rasterio
from rasterio import features

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--res', type=str, required=True)
args = parser.parse_args()

### Choose decimal package to avoid drama with precision
### Get longitudes for Vic
start = Decimal('140')
stop = Decimal('150.1')
step = Decimal(args.res)

lon_array = np.arange(start, stop + step, step, dtype=object)
lon = np.array(lon_array, dtype=float)

### Get latitudes for Vic
start = Decimal('33')
stop = Decimal('40')
step = Decimal(args.res)
lat_array = np.arange(start, stop + step, step, dtype=object) 
lat = np.array(lat_array, dtype=float)* (-1)

# Create grid based on lat and lon
lon_grid, lat_grid = np.meshgrid(lon, lat)

# Flatten arrays
latitude_values = lat_grid.ravel()
longitude_values = lon_grid.ravel()

# Create a DataFrame from the flattened arrays: lon, lat order
data = {
    'longitude': longitude_values,
    'latitude': latitude_values
}

# Make dataframe
df = pd.DataFrame(data)

### Select points that are in Vic: Read in shape file with state boundaries; index 1 = Vic
gdf = gpd.read_file('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/dynamics_simulations/'
                    'CFA/ML/analysis/shape_files/STE_2021_AUST_GDA2020.shp')
state = gdf.iloc[[1]]

### Create geopandas df and join df and state boundaries
gdf_vic = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], 
                                                           df['latitude']))

### Drop points outside Vic
gdf_vic = gpd.sjoin(gdf_vic, state, op='within')

### Select grid points
df_vic = gdf_vic[['longitude', 'latitude']]

### Write file: use resolution as identifier
res_fname = args.res
df_vic.to_csv(res.replace('.','')+'_grid.txt', 
              sep=' ', 
              index=False)
