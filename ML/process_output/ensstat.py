import pandas as pd
import numpy as np
import argparse

### Initialise argument parsing: RCP scenario (rcp45 or rcp85) and
### timespan (mid = 2045 - 2060, long = 2085 - 2100)

parser = argparse.ArgumentParser()
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)

args = parser.parse_args()

### Assign variables
scen = args.scen
timespan = args.timespan

### Set pathway for input data
pathwayIN='/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/dynamics_simulations/CFA/ML/output/csv/csv_FT/'

### Function to read in csv 
def get_data(GCM,scen,timespan):
    df = pd.read_csv(pathwayIN+scen+'_'+timespan+'/fut_'+GCM+'_'+scen+'_'+timespan+'.csv',
                     index_col=0)
    return(df[GCM].values)

### Initialise dataframe and read in each GCM
df = pd.DataFrame()
df['ACCESS1-0'] = get_data('ACCESS1-0',scen,timespan)
df['BNU-ESM'] = get_data('BNU-ESM',scen,timespan)
df['CSIRO-Mk3-6-0'] = get_data('CSIRO-Mk3-6-0',scen,timespan)
df['GFDL-CM3'] = get_data('GFDL-CM3',scen,timespan)
df['GFDL-ESM2G'] = get_data('GFDL-ESM2G',scen,timespan)
df['GFDL-ESM2M'] = get_data('GFDL-ESM2M',scen,timespan)
df['INM-CM4'] = get_data('INM-CM4',scen,timespan)
df['IPSL-CM5A-LR'] = get_data('IPSL-CM5A-LR',scen,timespan)
df['MRI-CGCM3'] = get_data('MRI-CGCM3',scen,timespan)

### Convert pandas dataframe to numpy array for faster processing
array = df.to_numpy()

### Calculate mode along the rows, save count of GCMs for mode
modes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=array)
mode_counts = np.sum(array == modes[:, np.newaxis], axis=1)

### Save mode to dataframe (easy to save dataframe to csv)
df_mode = pd.DataFrame()
df_mode['mode'] = modes
df_mode['mode_counts'] = mode_counts
df_mode.to_csv(pathwayIN+scen+'_'+timespan+'/mode_'+scen+'_'+timespan+'.csv',
               index=False)
