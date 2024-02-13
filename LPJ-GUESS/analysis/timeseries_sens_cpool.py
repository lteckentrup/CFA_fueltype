import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig=plt.figure(figsize=(12,9))
ax1=fig.add_subplot(2,2,1)
ax2=fig.add_subplot(2,2,2)
ax3=fig.add_subplot(2,2,3)
ax4=fig.add_subplot(2,2,4)

GCM_list = ['CNRM-CERFACS-CNRM-CM5', 'CSIRO-BOM-ACCESS1-0', 
            'MIROC-MIROC5', 'NOAA-GFDL-GFDL-ESM2M']
pathwayIN = ('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
             'dynamics_simulations/CFA/LPJ-GUESS/')

import argparse

'''
Initialise argument parsing: 
scen for RCP scenario (RCP45 or RCP85)
sens for sensitivity test: 
nofire (fire switched of) 
FHSF  (frequent high severity fire)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--sens', type=str, required=True)

args = parser.parse_args()

### Assign variables
scen=args.scen
sens=args.sens

### Open files and return sensitivity of simulations to no fire/ frequent high severity fire 
def open_data(var,GCM,scen,sens):
    ### Sensitivity experiment
    ds_sens = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+
                              GCM+'_'+scen+'_'+sens+'/'+var+'_1960-2099_fldsum.nc')  

    ### 'Control' simulation  
    ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+
                         GCM+'_'+scen+'/'+var+'_1960-2099_fldsum.nc')     

    ### Grab Total
    da_sens = ds_sens['Total']
    da = ds['Total']

    ### Get sensitivity: we discussed this before 
    if sens == 'nofire':
        da_delta = da - da_sens
    else:
        da_delta = da_sens - da

    ### Return future projections
    return(da_delta.sel(time=slice('2005','2099')).values.flatten())

### Calculate ensemble stats
def get_ens_stat(var,scen,sens):

    ### Open dataframe and include each sens. timeseries as a column
    df = pd.DataFrame()
    for GCM in GCM_list:
        df[GCM] = open_data(var,GCM,scen,sens)

    ### Get ensemble average and standard deviation
    df['Ensemble mean'] = df[GCM_list].mean(axis=1)
    df['Ensemble std'] = df[GCM_list].std(axis=1)

    ### Apply moving average to smooth timeseires
    df = df.rolling(3,center=True).mean()

    ### Include time info
    df['time'] = np.arange(2005,2100,1)
    return(df)

###  Set up plot for timeseries
def plot_timeseries(var, scen, sens, axis):
    ### Get dataframe
    df = get_ens_stat(var,scen,sens)

    #if sens == 'TFInew':
    #    label = '20$\%$ every year'
    #else:
    #    label = '5$\%$ every year'

    ### Plot timeseries of ensemble mean
    axis.plot(df['time'],
              df['Ensemble mean'],
              label='Ensemble mean')
    
    ### Plot shaded area of ensemble mean +- ensemble standard deviation
    axis.fill_between(df['time'],
                      df['Ensemble mean']-df['Ensemble std'],
                      df['Ensemble mean']+df['Ensemble std'],
                      label='Ensemble standard deviation',
                      color='tab:blue',
                      alpha=0.3)

    ### Drop top and right spine
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.axhline(color='tab:grey',lw=0.5)

### Call timeseries plot for four variables
plot_timeseries('cmass',scen,sens,ax1)
plot_timeseries('cmass_leaf',scen,sens,ax2)
plot_timeseries('cmass_wood',scen,sens,ax3)
plot_timeseries('clitter',scen,sens,ax4)

### Remove xticklabels for upper two panels
for a in (ax1,ax2):
    a.set_xticklabels([])

### Set y-labels
ax1.set_ylabel('$\Delta$ Carbon stored in\nvegetation [PgC]')
ax2.set_ylabel('$\Delta$ Carbon stored in\nleaves [PgC]')
ax3.set_ylabel('$\Delta$ Carbon stored in\nwood [PgC]')
ax4.set_ylabel('$\Delta$ Carbon stored in\nlitter [PgC]')

### Set titles
ax1.set_title('Carbon stored in vegetation')
ax2.set_title('Carbon stored in leaves')
ax3.set_title('Carbon stored in wood')
ax4.set_title('Carbon stored in litter')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')

### Include legend
ax1.legend(loc='upper left',frameon=False)

### Align labels
fig.align_ylabels()
plt.tight_layout()
#plt.show()
plt.savefig('figures/timeseries_sens_cpool_'+scen+'_'+sens+'.pdf')
