import xarray as xr
import rasterio
import numpy as np
import rioxarray as rxr
import pandas as pd

### Read in regridded tif
da = rxr.open_rasterio('../data/EVC_fuelType/evc/VICSANSW161_90.tif')

### Change projection for comparison with predictors
da_wgs84 = da[0].rio.reproject('EPSG:4326')

### Rename dimensions
da_wgs84 = da_wgs84.rename({'x':'lon','y':'lat'})

### Convert to dataframe
df = da_wgs84.to_dataset(name='FT').to_dataframe()

### Drop fuel types outside Victoria
df[df['FT'] >= 4000] = np.nan
df[df['FT'] <= 2999] = np.nan

df.replace(3000, np.nan, inplace=True)
df.replace(3047, np.nan, inplace=True)
df.replace(3097, np.nan, inplace=True)
df.replace(3098, np.nan, inplace=True)
df.replace(3099, np.nan, inplace=True)

### Drop NaN
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.drop(columns=['band','spatial_ref'],inplace=True)

### Save dataframe
df.to_csv('../cache/index.csv',index=False)
