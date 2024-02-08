import pandas as pd
import numpy as np
import xarray as xr
import argparse

'''
Initialise argument parsing: 
GCM for the different ensemble members: ACCESS1-0 BNU-ESM 
CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M INM-CM4 
IPSL-CM5A-LR MRI-CGCM3
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
GCM = args.GCM
scen = args.scen
timespan = args.timespan

### Set pathway for input data
pathwayIN='/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/dynamics_simulations/CFA/ML/'

### Read in input file from model training for coordinate information
if GCM == 'mode':
    df = pd.read_csv(pathwayIN+'input/cache/pred.ACCESS1-0.'+scen+'_'+timespan+'.csv')
else:
    df = pd.read_csv(pathwayIN+'input/cache/pred.'+GCM+'.'+scen+'_'+timespan+'.csv')
df.rename(columns={'ft': 'FT'},inplace=True)

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

### Read in either result aggregated across GCM ensemble ('mode')
### or result based on individual GCM
if GCM == 'mode':
    df_fut = pd.read_csv(pathwayIN+'output/csv/csv_FT/'+scen+'_'+timespan+
                         '/mode_'+scen+'_'+timespan+'.csv')
    df_dropna['fuel_type']= df_fut['mode'].values.flatten()
else:
    df_fut = pd.read_csv(pathwayIN+'output/csv/csv_FT/'+scen+'_'+timespan+
                         '/fut_'+GCM+'_'+scen+'_'+timespan+'.csv')
    df_dropna['fuel_type']= df_fut[GCM].values.flatten()

### Select relevant columns and convert to xarray dataset
df_sel = df_dropna[['lat','lon','fuel_type']]
df_final = df_sel.set_index(['lat','lon'])
ds = df_final.to_xarray()

### Save to netCDF
if GCM == 'mode':
    ds.to_netcdf(pathwayIN+'/output/netCDF/mode_'+scen+'_'+timespan+'_90.nc',
                encoding={'lat':{'dtype': 'double'},
                          'lon':{'dtype': 'double'},
                          'fuel_type':{'dtype': 'float32'}})

else:
    ds.to_netcdf(pathwayIN+'output/netCDF/GCM_FT/90m_res/'+scen+'_'+
                 timespan+'/fut_'+GCM+'_'+scen+'_'+timespan+'_90.nc',
                 encoding={'lat':{'dtype': 'double'},
                           'lon':{'dtype': 'double'},
                           'fuel_type':{'dtype': 'float32'}})
