import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(9,8))

fig.subplots_adjust(hspace=0.17)
fig.subplots_adjust(wspace=0.25)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.9)

ax1=fig.add_subplot(3,2,1)
ax2=fig.add_subplot(3,2,2)
ax3=fig.add_subplot(3,2,3)
ax4=fig.add_subplot(3,2,4)
ax5=fig.add_subplot(3,2,5)
ax6=fig.add_subplot(3,2,6)

### List of GCMs
GCM_list = ['CNRM-CERFACS-CNRM-CM5', 'CSIRO-BOM-ACCESS1-0', 
            'MIROC-MIROC5', 'NOAA-GFDL-GFDL-ESM2M']

### Set pathway where input files are located
pathwayIN = ('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
             'dynamics_simulations/CFA/LPJ-GUESS/')

def get_data(GCM,scen,FT_group,var,first_year,last_year):
    ### FPC is average over Vic, all carbon variables are sums
    if var == 'fpc':
        suffix = 'fldmean'
    else:
        suffix = 'fldsum'

    ### Read in all of Victora vs averages/sums aggregated over fuel groups
    if FT_group == '':
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+
                             GCM+'_'+scen+'/'+var+'_1960-2099_'+suffix+'.nc')
    else:
        ds = xr.open_dataset(pathwayIN+'/output/netCDF/NHP/runs_'+GCM+'_'+scen+
                             '/'+FT_group+'/'+var+'_1960-2099_'+suffix+'.nc')
    
    ### Calculate difference between future projection vs historical
    diff = ds.sel(time=slice(first_year,last_year))['Total'].values - \
           ds.sel(time=slice('2000','2014'))['Total'].values

    return(diff.flatten())

def get_ens_stat(scen,FT_group,var,first_year,last_year):

    ### Create dataframe where each column is the timeseries of one GCM
    df = pd.DataFrame()
    for GCM in GCM_list:
        df[GCM] = get_data(GCM,scen,FT_group,var,first_year,last_year)

    return(df.sum().mean(),df.sum().std())

def plot_barplot(FT_group,var1,var2,axis):

    ### Set up for loop
    scenarios = ['RCP4.5', 'RCP8.5']
    periods = [('2045', '2059'), ('2085', '2099')]
    dict_mean_data = {}
    dict_std_data = {}

    ### Create pandas dataframes for ens avg and standard deviation
    for scenario in scenarios:
        for start_year, end_year in periods:
            ### Drop point to read in file names
            scenario_fname = scenario.replace('.', '')

            ### Set column names
            column_name = f'{scenario} ({start_year} - {end_year})'

            ### Create dictionaries where each entry is the timeseries of one GCM
            dict_mean_data[column_name] = [
                get_ens_stat(scenario_fname, FT_group, var, 
                             start_year, end_year)[0] for var in [var1, var2]
                             ]
            dict_std_data[column_name] = [
                get_ens_stat(scenario_fname, FT_group, var, 
                             start_year, end_year)[1] for var in [var1, var2]
                             ]

    ### Convert dictionaries to dataframes
    df_mean = pd.DataFrame(dict_mean_data)
    df_std = pd.DataFrame(dict_std_data)

    ### Assign scenario names
    df_mean['Scenarios'] = [var1, var2]
    df_std['Scenarios'] = [var1, var2]

    ### Set scenario names as index  
    df_mean.set_index('Scenarios', inplace=True)
    df_std.set_index('Scenarios', inplace=True)

    ### Plot barplot with errorbars
    axis = df_mean.plot(kind='bar',
                        yerr=df_std,
                        color=['#5eccab', '#00678a', '#e6a176', '#984464'],
                        stacked=False,
                        ax=axis,
                        legend=False,
                        error_kw={'linewidth': 1})
    
    ### Plot horizontal line
    axis.axhline(lw=0.5,c='tab:grey')

    ### Remove labels (add later)
    axis.set_xticklabels([])
    axis.set_xlabel('')

    ### Drop right and top spine
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

### Plot barplot
plot_barplot('','cmass','cmass_wood',ax1)
plot_barplot('Wet_forest','cmass','cmass_wood',ax3)
plot_barplot('Mallee','cmass','cmass_wood',ax5)

plot_barplot('','cmass_leaf','clitter',ax2)
plot_barplot('Wet_forest','cmass_leaf','clitter',ax4)
plot_barplot('Mallee','cmass_leaf','clitter',ax6)

### Set xticklabels
ax5.set_xticklabels(['Carbon stored\nin vegetation',
                     'Carbon stored\nin wood'])
ax6.set_xticklabels(['Carbon stored\nin leaves',
                     'Carbon stored\nin litter'])

### Set subplot titles
axes = [ax1,ax2,ax3,ax4,ax5,ax6]
title_labels = ['Victoria','Victoria',
                'Wet forest','Wet forest',
                'Mallee','Malle']
title_indices = ['a)','b)','c)','d)','e)','f)']

for ax, tl, ti in zip(axes,title_labels,title_indices):
    ax.set_title(tl)
    ax.set_title(ti,loc='left')

### Set y label
for ax in (ax1,ax3,ax5):
    ax.set_ylabel('$\Delta$ Carbon pool [PgC]')

### Set legend
ax1.legend(loc='upper center', 
           bbox_to_anchor=(1.1, 1.5),
           frameon=False,
           ncols=2)

fig.align_ylabels()
plt.savefig('figures/barplot_delta_cpool_fuel_group.pdf')
