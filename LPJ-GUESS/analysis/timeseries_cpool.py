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

def open_data(var,GCM,scen):
    ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+
                         GCM+'_'+scen+'/'+var+'_1960-2099_fldsum.nc')  
    da = ds['Total']

    ### Only show future projections?
    return(da.sel(time=slice('2005','2099')).values.flatten())

### Get ensemble 'stats' - such a small ensemble size  ¯\_(ツ)_/¯
def get_ens_stat(var,scen):
    global GCM_list

    ### Create dataframe where each column is the timeseries of one GCM
    df = pd.DataFrame()
    for GCM in GCM_list:
        df[GCM] = open_data(var,GCM,scen)

    ### Get ensemble mean and standard deviation
    df['Ensemble mean'] = df[GCM_list].mean(axis=1)
    df['Ensemble std'] = df[GCM_list].std(axis=1)

    ### Rolling average to smooth timeseries 
    df = df.rolling(3,center=True).mean()
    df['time'] = np.arange(2005,2100,1)
    return(df)
     
def plot_timeseries(var, scen, axis, color):
    ### Get dataframe with ensemble stats
    df = get_ens_stat(var,scen)
    
    ### Plot timeseries of ensemble mean
    axis.plot(df['time'],
              df['Ensemble mean'],
              label=scen[:4]+'.'+scen[4:],
              color=color)
    
    ### Plot shaded area of ensemble mean +- ensemble standard deviation
    axis.fill_between(df['time'],
                      df['Ensemble mean']-df['Ensemble std'],
                      df['Ensemble mean']+df['Ensemble std'],
                      color=color,
                      alpha=0.3)

    ### Drop top and right spine
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

scen='RCP45'
color='tab:blue'
plot_timeseries('cmass',scen,ax1,color)
plot_timeseries('cmass_leaf',scen,ax2,color)
plot_timeseries('cmass_wood',scen,ax3,color)
plot_timeseries('clitter',scen,ax4,color)

scen='RCP85'
color='tab:red'
plot_timeseries('cmass',scen,ax1,color)
plot_timeseries('cmass_leaf',scen,ax2,color)
plot_timeseries('cmass_wood',scen,ax3,color)
plot_timeseries('clitter',scen,ax4,color)

### Hide ticklabels for upper two panels
for a in (ax1,ax2):
    a.set_xticklabels([])

ax1.set_ylabel('Carbon stored in vegetation [PgC]')
ax2.set_ylabel('Carbon stored in leaves [PgC]')
ax3.set_ylabel('Carbon stored in wood [PgC]')
ax4.set_ylabel('Carbon stored in litter [PgC]')

ax1.set_title('Carbon stored in vegetation')
ax2.set_title('Carbon stored in leaves')
ax3.set_title('Carbon stored in wood')
ax4.set_title('Carbon stored in litter')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')

ax4.legend(loc='upper left',frameon=False)
fig.align_ylabels()
plt.tight_layout()
plt.savefig('figures/timeseries_cpool.pdf')
