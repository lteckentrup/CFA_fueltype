import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from fueltype_analysis_attributes import \
    Wet_Shrubland, Wet_Shrubland_labels, CM_Wet_Shrubland, \
    Wet_Forest, Wet_Forest_labels, CM_Wet_Forest, \
    Grassland, Grassland_labels, CM_Grassland, \
    Dry_forest, Dry_forest_labels, CM_Dry_forest, \
    Shrubland, Shrubland_labels, CM_Shrubland, \
    High_elevation, High_elevation_labels, CM_High_elevation, \
    Mallee, Mallee_labels, CM_Mallee 

import argparse

'''
Initialise argument parsing: 
var for predictor variable
'''

parser = argparse.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
args = parser.parse_args()

### Set up figure
fig=plt.figure(figsize=(8.27,11.69))
ax1=fig.add_subplot(1,1,1)

### Get predictor variables; climate from SILO dataset
df = pd.read_csv('../../fuelType_ML/cache/ft.ACCESS1-0.history.csv')

'''
Drop Temperate Grassland / Sedgeland (3020) and
Eaten Out Grass when it's NOT on public land
! other irrelevant fuel types are implicitly excluded because they don't occur
in the list of fuel types below
'''
df = df.loc[~((df['FT'] == 3020) & (df['tenure'] == 0)),:]
df = df.loc[~((df['FT'] == 3046) & (df['tenure'] == 0)),:]

### Set inf to Nan
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

### Convert temperature to degrees Celsius
if args.var == 'tmax.mean':
    df[args.var] = df[args.var]-273.15
else:
    df[args.var] = df[args.var]
    
df.reset_index(drop=True,inplace=True)

## Create custom order of fuel types to display individual
## fuel types ordered within their broad fuel groups
FT_list = Wet_Shrubland+Wet_Forest+Grassland+Dry_forest+\
          Shrubland+High_elevation+Mallee

### Order colormaps accordingly
CM_FT = CM_Wet_Shrubland+CM_Wet_Forest+CM_Grassland+CM_Dry_forest+\
        CM_Shrubland+CM_High_elevation+CM_Mallee

### List of yticklabels following custom_order
yticklabels = np.concatenate([Wet_Shrubland_labels,
                              Wet_Forest_labels,
                              Grassland_labels,
                              Dry_forest_labels,
                              Shrubland_labels,
                              High_elevation_labels,
                              Mallee_labels])

### Make boxplot 
ax1 = sns.boxplot(data=df, 
                  x=args.var, 
                  y='FT', 
                  orient='h', 
                  showfliers = False, 
                  palette=CM_FT, 
                  order=FT_list, 
                  ax=ax1)

### Set labels
ax1.set_yticklabels(yticklabels)           
ax1.set_ylabel('')
ax1.set_xlabel('')

if args.var == 'map':
    ax1.set_xlabel('MAP [mm]')
if args.var == 'rad.short.jan':
    ax1.set_xlabel('Rad$_{Jan}$ [MJ m$^{-2}$ day$^{-1}$]')
if args.var == 'rad.short.jul':
    ax1.set_xlabel('Rad$_{Jul}$ [MJ m$^{-2}$ day$^{-1}$]')
if args.var == 'lai.opt.mean':
    ax1.set_xlabel('LAI$_{opt}$ [m$^{2}$ m$^{-2}$]')
if args.var == 'wi':
    ax1.set_xlabel('TWI [-]')
if args.var == 'soil.depth':
    ax1.set_xlabel('Soil depth [m]')
if args.var == 'clay':
    ax1.set_xlabel('Clay [%]')
if args.var == 'awc':
    ax1.set_xlabel('AWC [%]')
if args.var == 'tmax.mean':
    ax1.set_xlabel('T$_{max} [^{\circ}$C]')
if args.var == 'soil.density':
    ax1.set_xlabel('BDW [g $cm^{-3}$]')
if args.var == 'rh.mean':
    ax1.set_xlabel('RH$_{min}$ [%]')
if args.var == 'curvature_profile':
    ax1.set_xlabel('Profile curvature [-]')
if args.var == 'curvature_plan':
    ax1.set_xlabel('Plan curvature [-]')
if args.var == 'pr.seaonality':
    ax1.set_xlabel('Precipitation seasonality [-]')
if args.var == 'thorium_pot':
    ax1.set_xlabel('Thorium Potassium ratio [-]')
if args.var == 'uran_pot':
    ax1.set_xlabel('Uranium Potassium ratio [-]')

### Drop spines
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

### Save figure
plt.tight_layout()
plt.savefig('figures/boxplot_'+args.var+'.pdf')

