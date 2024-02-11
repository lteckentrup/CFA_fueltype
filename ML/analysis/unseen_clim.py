import xarray as xr
import numpy as np

import argparse

'''
Initialise argument parsing: 
GCM for the different ensemble members: ACCESS1-0 BNU-ESM 
CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M INM-CM4 
IPSL-CM5A-LR MRI-CGCM3
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
years (20452060 and 20852100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)
parser.add_argument('--years', type=str, required=True)

args = parser.parse_args()

### Assign variables
GCM = args.GCM
scen= args.scen
timespan = args.timespan
years = args.years

### Set pathway
pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/input/clim/')

'''
I made it so 'unseen climate' is when projected future climate is either 
above or below historical max/ min. Each variable gets a different masking value
(see below: annual_tmax = 1, annual_pr = 10, annual_rh = 100 and 
pr_seasonality = 1000) so that the sum over each possible combination has a 
unique value (so you can see which variable or variable combination is unseen)
'''

def get_new_clim(GCM,scen,timespan,years,var,fill_val):
    global pathway

    ### Read in variables: Compare historial ('hist') vs 
    ### projected future ('fut') climate
    ds_hist = xr.open_dataset(pathway+GCM+'/history/history_20002015_'+var+'.nc')
    ds_fut = xr.open_dataset(pathway+GCM+'/'+
                             scen+'_'+timespan+'/'+scen+'_'+years+'_'+var+'.nc')

    ### Find minimum and maximum value in historical period
    minimum = np.nanmin(ds_hist.layer)
    maximum = np.nanmax(ds_hist.layer)

    ### Mask areas in projected future climate that lower or higher than
    ### minimum and maximum historical values (= unseen climate)
    da = xr.where((ds_fut.layer < minimum) | 
                  (ds_fut.layer  > maximum), 
                  fill_val, 0)
    return(da)

### Add 'unseen climate' values for four climate variables
da = get_new_clim(GCM,scen,timespan,years,'annual_tmax',1) + \
     get_new_clim(GCM,scen,timespan,years,'annual_pr',10) + \
     get_new_clim(GCM,scen,timespan,years,'annual_rh',100) + \
     get_new_clim(GCM,scen,timespan,years,'pr_seasonality',1000)

### Reassign values (easier to plot later on)

### Individual variables
da = xr.where(da == 1, 1, da) # T out of range
da = xr.where(da == 10, 2, da) # pr out of range
da = xr.where(da == 100, 3, da) # rh out of range
da = xr.where(da == 1000, 4, da) # pr seas out of range

### Two variables combined
da = xr.where(da == 11, 5, da)  # T and pr out of range
da = xr.where(da == 101, 6, da) # T and Rh out of range
da = xr.where(da == 1001, 7, da) # T and pr seas out of range

da = xr.where(da == 110, 8, da) # pr and rh out of range
da = xr.where(da == 1010, 9, da) # pr and pr seasonality out of range

da = xr.where(da == 1100, 10, da) # rh and pr seasonality out of range

### Three variables combined
da = xr.where(da == 111, 11, da) # T, pr and rh out of range
da = xr.where(da == 1110, 12, da) # pr, rh and pr seasonality out of range
da = xr.where(da == 1011, 13, da)  # T, pr and pr seasonality out of range
da = xr.where(da == 1101, 14, da)  # T, rh and pr seasonality out of range

### All variables unseen
da = xr.where(da == 1111, 15, da)  

### Convert DataArray to DataSet
ds = da.to_dataset(name = 'out_of_range')

### Save to netCDF
ds.to_netcdf('netCDF/'+GCM+'_'+scen+'_'+timespan+'.nc',
             encoding={'latitude':{'dtype': 'double'},
                       'longitude':{'dtype': 'double'},
                       'out_of_range':{'dtype': 'float32'}})
