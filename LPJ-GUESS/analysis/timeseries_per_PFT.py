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

import argparse

'''
Initialise argument parsing: 
var for variable (in report we used fpc, cmass, clitter)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
args = parser.parse_args()

### Assign variables
var=args.var

def open_data(var,PFT,GCM,exp):
    if var == 'fpc':
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+exp+
                             '/'+var+'_1960-2099_fldmean.nc')    
    else:
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+exp+
                             '/'+var+'_1960-2099_fldsum.nc')  

    ### Show PFT as fraction of Total 
    da = ds[PFT]/ ds['Total']

    ### Only show future projections?
    return(da.sel(time=slice('2005','2099')).values.flatten())

def get_ens_stat(var,PFT,exp):
    global GCM_list

    ### Create dataframe where each column is the timeseries of one GCM
    df = pd.DataFrame()
    for GCM in GCM_list:
        df[GCM] = open_data(var,PFT,GCM,exp)*100 

    ### Get ensemble mean and standard deviation
    df['Ensemble mean'] = df[GCM_list].mean(axis=1)
    df['Ensemble std'] = df[GCM_list].std(axis=1)

    ### Rolling average to smooth timeseries 
    df = df.rolling(3,center=True).mean()
    df['time'] = np.arange(2005,2100,1)
    return(df)
     
def plot_timeseries(var, PFT, exp, axis, color):
    ### Get dataframe with ensemble stats
    df = get_ens_stat(var,PFT,exp)

    ### Plot timeseries of ensemble mean
    axis.plot(df['time'],
              df['Ensemble mean'],
              color=color,
              label=exp[:4]+'.'+exp[4:])
    
    ### Plot shaded area of ensemble mean +- ensemble standard deviation
    axis.fill_between(df['time'],
                      df['Ensemble mean']-df['Ensemble std'],
                      df['Ensemble mean']+df['Ensemble std'],
                      color=color,
                      alpha=0.3)

    ### Drop top and right spine
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.set_title(PFT)

PFT_short_names=['CRT','ST','SM','SS','SuS','MS','XS','C3','C4']
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

### Plot timeseries
exp='RCP45'
color = 'tab:blue'
for PFT_S,ax in zip(PFT_short_names,axes):
    plot_timeseries(var,PFT_S,exp,ax,color)

exp='RCP85'
color = 'tab:red'
for PFT_S,ax in zip(PFT_short_names,axes):
    plot_timeseries(var,PFT_S,exp,ax,color)

### Set title
for ax, PFT_L in zip(axes,PFT_longnames):
    ax.set_title(PFT_L)
    ax.set_title('a)', loc='left')

for ax in (ax1,ax2,ax3,ax4,ax5,ax6):
    ax.set_xticklabels([])

for ax in (ax1,ax4,ax7):
    if var == 'fpc':
        ax.set_ylabel('Percentage FPC [%]')
    elif var == 'cmass':
        ax.set_ylabel('Percentage carbon\nstored in vegetation [%]')
    elif var == 'clitter':
        ax.set_ylabel('Percentage carbon stored in litter [%]')

ax9.legend(loc='upper right',frameon=False)
fig.align_ylabels()
plt.tight_layout()
plt.savefig('figures/timeseries_'+var+'_per_PFT.pdf')
