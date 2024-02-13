import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig=plt.figure(figsize=(12,9))
ax1=fig.add_subplot(3,3,1)
ax2=fig.add_subplot(3,3,2)
ax3=fig.add_subplot(3,3,3)
ax4=fig.add_subplot(3,3,4)
ax5=fig.add_subplot(3,3,5)
ax6=fig.add_subplot(3,3,6)
ax7=fig.add_subplot(3,3,7)
ax8=fig.add_subplot(3,3,8)
ax9=fig.add_subplot(3,3,9)

GCM_list = ['CNRM-CERFACS-CNRM-CM5', 'CSIRO-BOM-ACCESS1-0', 
            'MIROC-MIROC5', 'NOAA-GFDL-GFDL-ESM2M']
pathwayIN = ('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
             'dynamics_simulations/CFA/LPJ-GUESS/')

'''
Initialise argument parsing: 
var for variable (in report we used fpc, cmass, clitter)
scen for RCP scenario (rcp45 or rcp85)
sens for sensitivity test: 
nofire (fire switched of) 
FHSF  (frequent high severity fire)
'''

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--sens', type=str, required=True)

args = parser.parse_args()

### Assign variables
var=args.var
scen=args.scen
sens=args.sens

def open_data(var,PFT,GCM,scen,sens):
    if var == 'fpc':
        ds_sens = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+
                                  scen+'_'+sens+'/'+var+'_1960-2099_fldmean.nc')  
        ### 'Control' simulation  
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+
                             scen+'/'+var+'_1960-2099_fldmean.nc')      
    else:
        ### Sensitivity experiment
        ds_sens = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+
                                  scen+'_'+sens+'/'+var+'_1960-2099_fldsum.nc')  
        ### 'Control' simulation  
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+
                             scen+'/'+var+'_1960-2099_fldsum.nc')     

    ### Show PFT as fraction of Total 
    da_sens = ds_sens[PFT]/ ds_sens['Total']
    da = ds[PFT]/ ds['Total']

    ### Get sensitivity: we discussed this before 
    if sens == 'nofire':
        da_delta = da - da_sens
    else:
        da_delta = da_sens - da

    ### Return future projections
    return(da_delta.sel(time=slice('2005','2099')).values.flatten())

### Calculate ensemble stats
def get_ens_stat(var,PFT,scen,sens):

    ### Open dataframe and include each sens. timeseries as a column
    df = pd.DataFrame()
    for GCM in GCM_list:
        df[GCM] = open_data(var,PFT,GCM,scen,sens)*100

    ### Get ensemble average and standard deviation
    df['Ensemble mean'] = df[GCM_list].mean(axis=1)
    df['Ensemble std'] = df[GCM_list].std(axis=1)
    
    ### Apply moving average to smooth timeseries
    df = df.rolling(3,center=True).mean()

    ### Include time info
    df['time'] = np.arange(2005,2100,1)
    return(df)

###  Set up plot for timeseries
def plot_timeseries(var, PFT, scen, sens, axis):

    ### Get dataframe
    df = get_ens_stat(var,PFT,scen,sens)

    ### Plot timeseries of ensemble mean
    axis.plot(df['time'],
              df['Ensemble mean'],
              label='Ensemble mean')
    
    ### Plot shaded area of ensemble mean +- ensemble standard deviation
    axis.fill_between(df['time'],
                      df['Ensemble mean']-df['Ensemble std'],
                      df['Ensemble mean']+df['Ensemble std'],
                      label='Ensemble STD',
                      color='tab:blue',
                      alpha=0.3)

    ### Drop top and right spine
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    ### Set where horizontal lines should appear
    if var == 'fpc':
        if scen == 'RCP45':
            if sens == 'nofire':
                if axis in (ax6,ax7,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)                    
            elif sens == 'FHSF':
                if axis in (ax3,ax4,ax5,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)
        elif scen == 'RCP85':
            if sens == 'nofire':
                if axis in (ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)
            elif sens == 'FHSF':
                if axis in (ax2,ax3,ax4,ax5,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)

    if var == 'cmass':
        if scen == 'RCP45':
            if sens == 'nofire':
                if axis in (ax1,ax3,ax6,ax7,ax8):
                    axis.axhline(color='tab:grey',lw=0.5)
            elif sens == 'FHSF':
                if axis in (ax2,ax3,ax4,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)
        elif scen == 'RCP85':
            if sens == 'nofire':
                if axis in (ax3,ax6,ax7,ax8):
                    axis.axhline(color='tab:grey',lw=0.5)
            elif sens == 'FHSF':
                if axis in (ax2,ax3,ax4,ax5,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)

    if var == 'clitter':
        if scen == 'RCP45':
            if sens == 'nofire':
                if axis in (ax1,ax3,ax5,ax6,ax7,ax8):
                    axis.axhline(color='tab:grey',lw=0.5)
            elif sens == 'FHSF':
                if axis in (ax1,ax2,ax3,ax4,ax5,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)
        elif scen == 'RCP85':
            if sens == 'nofire':
                if axis in (ax1,ax5,ax6,ax7,ax8):
                    axis.axhline(color='tab:grey',lw=0.5)    
            elif sens == 'FHSF':
                if axis in (ax1,ax2,ax3,ax4,ax5,ax6,ax8,ax9):
                    axis.axhline(color='tab:grey',lw=0.5)

PFT_shortnames=['CRT','ST','SM','SS','SuS','MS','XS','C3','C4']
PFT_longnames = ['Cool rainforest tree',
                 'Tall sclerophyll',
                 'Medium sclerophyll',
                 'Short sclerophyll',
                 'Subalpine tree',
                 'Mesic shrub',
                 'Xeric shrub',
                 'C$_3$ grass',
                 'C$_4$ grass']
title_index=['a)','b)','c)','d)','e)','f)','g)','h)','i)']
axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

for PFTs,PFTl,ti,ax in zip(PFT_shortnames,PFT_longnames,title_index,axes):
    plot_timeseries(var,PFTs,scen,sens,ax)
    ax.set_title(PFTl)
    ax.set_title(ti, loc='left')

for a in (ax1,ax2,ax3,ax4,ax5,ax6):
    a.set_xticklabels([])

for a in (ax1,ax4,ax7):
    if var == 'fpc':
        a.set_ylabel('$\Delta$ Percentage FPC [%]')
    elif var == 'cmass':
        a.set_ylabel('$\Delta$ Percentage carbon\nstored in vegetation [%]')
    elif var == 'clitter':
        a.set_ylabel('$\Delta$ Percentage carbon\nstored in litter [%]')

### Plot legend
ax9.legend(loc='best',frameon=False)

fig.align_ylabels()
plt.tight_layout()
plt.savefig('figures/timeseries_sens_'+var+'_'+scen+'_'+sens+'.pdf')
