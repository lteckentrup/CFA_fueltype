import xarray as xr
import rasterio
import numpy as np
import rioxarray as rxr
import pandas as pd

### List of relevant fuel types
fueltypes = [3001, 3002, 3003, 3005, 3006, 3007, 3008, 3009, 3010, 3011,
             3012, 3013, 3014, 3015, 3021, 3022, 3023, 3024, 3025, 3026,
             3027, 3028, 3029, 3043, 3047, 3048, 3049, 3050, 3051]

'''
Nearest neighbour remapping with gdalwarp from command line
gdalwarp -r near -tr 90 90 VICSANSW161.tif VICSANSW161_90.tif
'''

### Read in regridded tif 
da = xr.open_rasterio('../data/EVC_fuelType/evc/VICSANSW161_90.tif')

### Change projection for comparison with predictors
da_wgs84 = da.rio.reproject('EPSG:4326')

### Rename dimensions
da_wgs84 = da_wgs84.rename({'x':'lon','y':'lat'})

### Convert to dataframe
df = da_wgs84.to_dataset(name='FT').to_dataframe()

### Drop pixels outside Victoria
df[df['FT'] >= 4000] = np.nan
df[df['FT'] <= 2999] = np.nan

### Drop irrelevant fuel types
df = df[df['FT'].isin(fueltypes)].dropna()

### Drop nan, make dataframe pretty
df.reset_index(inplace=True).drop(columns=['index','band','spatial_ref'],inplace=True)

### Save dataframe
df.to_csv('index.csv',index=False)
