import pandas as pd
import numpy as np
from fueltype_analysis_attributes import \
    Wet_Shrubland, Wet_Forest, Grassland, Dry_forest, Shrubland, \
    High_elevation, Mallee

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

fueltypes = Wet_Forest+High_elevation+Wet_Shrubland+Mallee+\
            Dry_forest+Grassland+Shrubland

### Read in present day fuel type distribution - stored in all predictor files for
### historical period ('history') - the same for all GCMs

### Set pathway
pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/')

df_hist = pd.read_csv(pathway+'input/cache/pred.ACCESS1-0.history.csv')

### Drop Temperate Grassland / Sedgeland (3020) and
### Eaten Out Grass when it's NOT on public land
df_hist = df_hist.loc[~((df_hist['FT'] == 3020) & (df_hist['tenure'] == 0)),:]
df_hist = df_hist.loc[~((df_hist['FT'] == 3046) & (df_hist['tenure'] == 0)),:]

### Drop Water, sand, no vegetation (3000)
df_hist.replace(3000, np.nan, inplace=True)

### Drop Non-Combustible (3047)
df_hist.replace(3047, np.nan, inplace=True)

### Drop Orchard / Vineyard (3097),
### Softwood Plantation (3098),
### Hardwood Plantation (3099)
df_hist.replace(3097, np.nan, inplace=True)
df_hist.replace(3098, np.nan, inplace=True)
df_hist.replace(3099, np.nan, inplace=True)

### Set inf to Nan
df_hist.replace([np.inf, -np.inf], np.nan, inplace=True)
df_hist.dropna(inplace=True)

### Read in projected fuel type distribution (aggregated across ensemble)
df_fut = pd.read_csv(pathway+'output/csv/csv_FT/'+scen+'_'+timespan+
                     '/'+GCM+'_'+scen+'_'+timespan+'.csv')

### Combine dataframes for comparison
df_fut['hist'] = df_hist.reset_index()['FT']

### Calculate where present day fuel types remains or shifts to new type
def get_transition(fueltype):

    ### Select fuel type
    df_FT = df_fut[df_fut['hist'] == fueltype]

    ### Count which fuel types and how many occur in grid cells currently 
    ### occupied by selected fuel type
    labels,counts = np.unique(df_FT['mode'],return_counts=True)

    ### Set up dataframe with labels and respective count/ calculate percentage
    df_count = pd.DataFrame()
    df_count['labels'] = labels
    df_count[str(fueltype)] = (counts/len(df_FT))*100

    ### Full list of fuel types based on present day
    labels_hist = df_hist.FT.drop_duplicates()

    ### Get list of projected fuel types - likely mismatch with full list
    labels_count = df_count.labels.drop_duplicates()

    ### Get missing fuel types compared to full list
    miss_labels = list(set(labels_hist).difference(labels_count))

    ### Fill missing fuel types and assign 0 -> easier to process dataframes
    ### when each dataframe has exactly same rows
    try:
        for i in range(0,len(miss_labels)):
            df_count.loc[len(df_count)] = [miss_labels[i],0]
    except IndexError:
        pass

    ### Sort dataframe by labels 
    df_count.sort_values(by='labels',inplace=True)

    ### Round to two decimals, reset index
    df_count[str(fueltype)]=df_count[str(fueltype)].round(2)
    df_count.reset_index(drop=True,inplace=True)

    df_count.to_csv('csv/fuel_type_shift/'
                    'transition_'+GCM+'_'+str(fueltype)+'_'+scen+'_'+timespan+'.csv',
                    index=False)

for i in fueltypes:
    get_transition(i)
