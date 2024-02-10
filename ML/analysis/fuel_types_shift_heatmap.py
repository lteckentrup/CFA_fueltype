import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.patches as patches

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

## Create custom order of fuel types to display individual
## fuel types ordered within their broad fuel groups
custom_order = Wet_Forest+High_elevation+Wet_Shrubland+Mallee+\
               Dry_forest+Grassland+Shrubland

### List of xticklabels following custom_order
xticklabels = np.concatenate([Wet_Forest_labels,
                              High_elevation_labels,
                              Wet_Shrubland_labels,
                              Mallee_labels,
                              Dry_forest_labels,
                              Grassland_labels,
                              Shrubland_labels])

### Read in fuel type shifts per fuel type: Does present-day fuel type xy stay
### stay the same or does it shift to a new fuel type?

### Read
df = pd.DataFrame()
df['labels'] = sorted(custom_order)

for ft in custom_order:
    df[str(ft)] = pd.read_csv('csv/fuel_type_shift/transition_mode_'+
                              str(ft)+'_'+scen+'_'+timespan+'.csv')[str(ft)]

df['labels'] = pd.Categorical(df['labels'], 
                              categories=custom_order, 
                              ordered=True)

### Sort values
df.sort_values('labels',inplace=True)
df.set_index('labels',inplace=True)
df[df==0] = np.nan ### if value exactly 0 set to nan for white colors in heatmap

### Set up figure
fig, ax = plt.subplots(figsize=(12,12))

### Heatmap
ax = sns.heatmap(df,cmap='inferno_r',vmin=0,vmax=100,
                 annot=True, fmt='.1f',annot_kws={'fontsize':6})

ax.set_facecolor('#f0eded')

### Set labels
ax.set_xticks(np.arange(len(df.columns)) + 0.5)
ax.set_yticks(np.arange(len(df.index)) + 0.5)
ax.set_xticklabels(xticklabels, rotation=90)
ax.set_yticklabels(xticklabels, rotation=0)

def plot_fuel_group_patch(x_bound,y_bound,CM_FT,color_index):
    x = x_bound
    y = y_bound

    rectangle = patches.Rectangle((x, x), y, y,
                                  linewidth=2, 
                                  edgecolor=CM_FT[color_index], 
                                  fill=False,
                                  zorder=10)
    
    plt.gca().add_patch(rectangle)

plot_fuel_group_patch(0.04, 6.96,CM_Wet_Forest,2)
plot_fuel_group_patch(7,4,CM_High_elevation,2)
plot_fuel_group_patch(11,5,CM_Wet_Shrubland,2)
plot_fuel_group_patch(16,7,CM_Mallee,2)
plot_fuel_group_patch(23,6,CM_Dry_forest,5)
plot_fuel_group_patch(29,4,CM_Grassland,1)
plot_fuel_group_patch(33,2.9,CM_Shrubland,1)

ax.set_xlabel('Observation')
ax.set_ylabel('Future shift')

label_title = scen
label_title = label_title[:4] + '.' + label_title[4:]
label_title = label_title.upper()

if timespan == 'mid':
    label = label_title + ' (2045-2060 vs observed)'
elif timespan == 'long':
    label = label_title + ' (2085-2100 vs observed)'

plt.suptitle(label)

# ax.set_xlabel('RCP8.5 (2045-2060)')
# ax.set_ylabel('RCP8.5 (2085-2100)')

# plt.suptitle('RCP8.5 (2085-2100 vs 2045-2060)')
plt.tight_layout()
plt.savefig('figures/transition_heatmap_'+scen+'_'+timespan+'.pdf')
