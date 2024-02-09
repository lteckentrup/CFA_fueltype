import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

### Set absolute pathway
pathwayIN=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
           'dynamics_simulations/CFA/ML/')

### Read in importance score
def get_imp(GCM):
    global pathwayIN
    df = pd.read_csv(pathwayIN+'/output/csv/csv_scores/feature_importance/'+
                     GCM+'_ft_importance.csv')
    return(df.importance)

### Set up dataframe for importance for individual GCMs
df = pd.DataFrame()
df['ACCESS1-0'] = get_imp('ACCESS1-0')
df['BNU-ESM'] = get_imp('BNU-ESM')
df['CSIRO-Mk3-6-0'] = get_imp('CSIRO-Mk3-6-0')
df['GFDL-CM3'] = get_imp('GFDL-CM3')
df['GFDL-ESM2G'] = get_imp('GFDL-ESM2G')
df['GFDL-ESM2M'] = get_imp('GFDL-ESM2M')
df['INM-CM4'] = get_imp('INM-CM4')
df['IPSL-CM5A-LR'] = get_imp('IPSL-CM5A-LR')
df['MRI-CGCM3'] = get_imp('MRI-CGCM3')

### Set up final dataframe
df_imp = pd.DataFrame()

### Grab feature labels
df_imp['Feature'] = pd.read_csv(
    pathwayIN+'/output/csv/csv_scores/'
    'feature_importance/ACCESS1-0_ft_importance.csv')['feature']

### Average importance across ensemble
df_imp['Importance'] = df.mean(axis=1)

### Standard deviation of importance across ensemble
df_imp['Importance (uncertainty)'] = df.std(axis=1)

### Sort dataframe by values
df_imp.sort_values(by=['Importance'],
                   inplace=True)

### Plot barplot with errorbars
fig, ax = plt.subplots(figsize=(9, 7))                        
ax.barh(df_imp['Feature'], 
        df_imp['Importance'], 
        xerr=df_imp['Importance (uncertainty)'],
        color='crimson')

### Update figure labels
ax.yaxis.set_major_locator(FixedLocator(range(len(df_imp))))
ax.set_yticklabels(['Plan curvature', 'Profile curvature','Rad$_{Jul}$',
                    'TWI', 'Thorium\nPotassium ratio', 
                    'Uranium\nPotassium ratio', 'AWC', 'Soil depth', 
                    'Precipitation\nseasonality', 'Clay', 'BDW',
                    'MAP', 'LAI$_{opt}$', 'RH$_{min}$', 'T$_{max}$',
                    'Rad$_{Jan}$'])
ax.set_xlabel('Feature importance')

### Drop right and top spine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/feature_importance.pdf')
