import pandas as pd
import numpy as np
import argparse

'''
Initialise argument parsing: 
scen for RCP scenario (rcp45 or rcp85) 
timespan (mid = 2045 - 2060, long = 2085 - 2100)
'''

parser = argparse.ArgumentParser()
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
scen = args.scen
timespan = args.timespan

### Read in observed pixel count of fuel types: 
### Same in each predictor csv file because it's the target file
pathwayIN='/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/dynamics_simulations/CFA/ML/'
df_obs = pd.read_csv(pathwayIN+'input/cache/ft.ACCESS1-0.history.csv')

### Drop Temperate Grassland / Sedgeland (3020) and
### Eaten Out Grass when it's NOT on public land
df_obs = df_obs.loc[~((df_obs['FT'] == 3020) & (df_obs['tenure'] == 0)),:]
df_obs = df_obs.loc[~((df_obs['FT'] == 3046) & (df_obs['tenure'] == 0)),:]

### Drop Water, sand, no vegetation (3000)
df_obs.replace(3000, np.nan, inplace=True)

### Drop Non-Combustible (3047)
df_obs.replace(3047, np.nan, inplace=True)

### Drop Orchard / Vineyard (3097),
### Softwood Plantation (3098),
### Hardwood Plantation (3099)
df_obs.replace(3097, np.nan, inplace=True)
df_obs.replace(3098, np.nan, inplace=True)
df_obs.replace(3099, np.nan, inplace=True)

### Set inf to NaN
df_obs.replace([np.inf, -np.inf], np.nan, inplace=True)
df_obs.dropna(inplace=True)

### Get labels of observed fuel types and the respective count
obs_labels, obs_count = np.unique(df_obs['FT'],
                                  return_counts=True)

### Set up dataframe
df_count = pd.DataFrame()
df_count['FT'] = obs_labels
df_count['Obs'] = obs_count/len(df_obs) ### Fraction

### Define function to get pixel count of fuel types 
### for future projections
def read_in_df(GCM,scen,timespan):
    df = pd.DataFrame()
    df[GCM] = pd.read_csv(pathwayIN+'/output/csv/csv_FT/'+
                          scen+'_long/fut_'+GCM+'_'+scen+'_'+timespan+'.csv')[GCM]
    
    ### Drop nan
    df.dropna(inplace=True)

    ### Get labels of projected fuel types and the respective count
    FT_labels, FT_count = np.unique(df[GCM],
                                    return_counts=True)

    ### Set up dataframe
    df_count = pd.DataFrame()
    df_count['FT'] = FT_labels
    df_count['Count'] = FT_count/len(df)

    '''
    Under extreme conditions, observed fuel types might 
    disappear in future projections: need to fill gaps to
    be able to merge columns across ensemble
    '''

    ### Full list of fuel types
    FT_full = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 
               3008, 3009, 3010, 3011, 3012, 3013, 3014, 
               3015, 3016, 3017, 3018, 3019, 3020, 3021, 
               3022, 3023, 3024, 3025, 3026, 3027, 3028, 
               3029, 3037, 3043, 3046, 3048, 3049, 3050, 
               3051]
    
    ### If fuel type is not projected, include row with fuel type
    ### and assign 0
    try:
        FT_missing = list(set(FT_full).difference(FT_labels))
        for i in range(0,len(FT_missing)):
            df_count.loc[len(df_count)] = [FT_missing[i],0]

    except IndexError:
        pass
    
    ### Sort values 
    df_count.sort_values(by='FT',inplace=True)
    return(df_count['Count'].values)

### Add column in dataframe for each GCM
df_count['ACCESS1-0'] = read_in_df('ACCESS1-0',scen,timespan)
df_count['BNU-ESM'] = read_in_df('BNU-ESM',scen,timespan)
df_count['CSIRO-Mk3-6-0'] = read_in_df('CSIRO-Mk3-6-0',scen,timespan)
df_count['GFDL-CM3'] = read_in_df('GFDL-CM3',scen,timespan)
df_count['GFDL-ESM2G'] = read_in_df('GFDL-ESM2G',scen,timespan)
df_count['GFDL-ESM2M'] = read_in_df('GFDL-ESM2M',scen,timespan)
df_count['INM-CM4'] = read_in_df('INM-CM4',scen,timespan)
df_count['IPSL-CM5A-LR'] = read_in_df('IPSL-CM5A-LR',scen,timespan)
df_count['MRI-CGCM3'] = read_in_df('MRI-CGCM3',scen,timespan)

### Get ensemble stats of gridcells across ensemble; drop fuel type
### and observation to calculate stats
df_count['Avg'] = df_count.drop(columns=['FT','Obs']).mean(axis=1)
df_count['Std'] = df_count.drop(columns=['FT','Obs']).std(axis=1)

### Save to csv file for figure
df_count.to_csv('csv/count_'+scen+'_'+timespan+'.csv',
                index=False)
