### Methods
from sklearn.ensemble import RandomForestClassifier

### Other libraries
import pandas as pd
import random
import numpy as np
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--timespan', type=str, required=True)
args = parser.parse_args()

### Set seed for reproducability
random.seed(42)

def get_data(fuel_group,GCM,scen,timespan):
    ### Read in features + target
    df = pd.read_csv('../input/cache/pred.'+GCM+'.'+scen+'_'+timespan+'.csv')

    ### Drop Temperate Grassland / Sedgeland (3020) and
    ###      Eaten Out Grass when it's NOT on public land
    df = df.loc[~((df['FT'] == 3020) & (df['tenure'] == 0)),:]
    df = df.loc[~((df['FT'] == 3046) & (df['tenure'] == 0)),:]

    ### Drop Water, sand, no vegetation (3000)
    df.replace(3000, np.nan, inplace=True)

    ### Drop Non-Combustible (3047)
    df.replace(3047, np.nan, inplace=True)

    ### Drop Orchard / Vineyard (3097),
    ###      Softwood Plantation (3098) and
    ###      Hardwood Plantation (3099)
    df.replace(3097, np.nan, inplace=True)
    df.replace(3098, np.nan, inplace=True)
    df.replace(3099, np.nan, inplace=True)

    ### Set inf to Nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    ### Drop all NaN
    df.dropna(inplace=True)

    print('Read in data')
    return(df.dropna())

def prep_data(fuel_group,GCM,scen,timespan):
    df = get_data(fuel_group,GCM,scen,timespan)
    print(df)
    ### Select predictors
    X = df[['soil.density', 
            'clay', 
            'rad.short.jan', 
            'rad.short.jul', 
            'wi', 
            'curvature_profile', 
            'curvature_plan', 
            'tmax.mean', 
            'map', 
            'pr.seaonality', 
            'lai.opt.mean', 
            'soil.depth',
            'uran_pot', 
            'thorium_pot', 
            'rh.mean',
            'awc']]

    feature_names = X.columns

    return(X, feature_names)

def predict_fut(fuel_group,GCM,scen,timespan):
    X, feature_names = prep_data(fuel_group,GCM,scen,timespan)
    clf = joblib.load(('pkl/'+GCM+'/RF_'+GCM+'.pkl'))
    print('loaded model')

    batch_size = 10000 
    y_pred = []

    for i in range(0, len(X), batch_size):
        print(i)
        batch_X = X[i:i+batch_size]
        batch_pred = clf.predict(batch_X)
        y_pred.extend(batch_pred)

    y_pred = np.array(y_pred)

    df = pd.DataFrame()
    df[GCM] = y_pred
    return(df)

df = predict_fut('Individual',args.GCM,args.scen,args.timespan)
print('prediction done')
df.to_csv('fut_'+args.GCM+'_'+args.scen+'_'+args.timespan+'.csv')
