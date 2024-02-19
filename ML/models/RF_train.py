### Import relevant sklearn libraries
### Preprocessing
from sklearn.model_selection import train_test_split

### Methods
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

### Interpretation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve

### Class weights
from sklearn.utils.class_weight import compute_class_weight

### Other libraries
import pandas as pd
import random
import numpy as np
import joblib
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--n_samples', type=int, required=True)
args = parser.parse_args()

### Set seed for reproducability
random.seed(42)

def get_data(GCM):
    ### Read in features + target
    df = pd.read_csv('../input/cache/pred.'+GCM+'.history.csv')
    df.rename(columns={'ft': 'FT'},inplace=True)

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

    return(df)        

'''
sample sizes I tried
1161998
813398
10107368
'''

def prep_data(GCM,n_samples,reduce_dim):
    df = get_data(GCM)
    
    n_samples = n_samples
    df_sampled = pd.DataFrame()

    ### Loop through all fuel types
    for value in df['FT'].unique():
        ### Get fuel type
        subset = df[df['FT'] == value]
        
        ### 5 classes have less than 10'000 classes
        n_samples_final = min(n_samples, len(subset))
        
        ### Randomly pull out 10'000 samples !set replace = False to avoid 
        ### that elements are sampled multiple times 
        random_indices = np.random.choice(subset.index, 
                                          n_samples_final,
                                          replace=False)
        
        # Add the selected samples to the result DataFrame
        df_sampled = pd.concat([df_sampled, subset.loc[random_indices]])

    ### Select target
    y = df_sampled['FT']

    ### Select predictors
    X = df_sampled[['soil.density', 
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
    
    ### Split data in to training and test datasets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)   

    ### Get feature names
    feature_names = X.columns

    return(X_train, X_test, y_train, y_test, feature_names)

def ML_function(GCM,n_samples,reduce_dim):
    ### Grab data
    X_train, X_test, y_train, y_test, feature_names = prep_data(GCM,
                                                                n_samples,
                                                                reduce_dim)

    ### Set up model
    clf = RandomForestClassifier(max_depth = 20, 
                                 min_samples_split = 10, 
                                 n_estimators= 500, 
                                 class_weight = 'balanced_subsample',
                                 n_jobs=-1)

    ### Fit model
    clf.fit(X_train, y_train)

    ### Save model for predictions
    joblib.dump(clf, 'pkl/'+GCM+'/RF_'+GCM+'_'+str(n_samples)+'.pkl')
    
    ### Predict on test data
    y_pred = clf.predict(X_test)

    ### You can also save the probability for each class (the probabilities give
    ### some indication on prediction confidence) but CFA wasn't interested in 
    ### those outputs. They can be retried using
    #### y_pred_prob = clf.predict_proba(X_test)

    class_names = clf.classes_  ### Get the class names from the classifier

    ### Dataframe with predicted fuel types
    df_y = pd.DataFrame()
    df_y['win'] = y_pred

    ### Save to csv file
    df_y.to_csv('csv/'+GCM+'/'+'y_pred_'+GCM+'_'+str(n_samples)+'.csv')

    ### Calculate overall accuracy
    accuracy = clf.score(X_test, y_pred)
    weighted_accuracy = balanced_accuracy_score(y_test, y_pred)

    classes = np.unique(y_test)
    accuracies = [accuracy_score(y_test[y_test == c], 
                                 y_pred[y_test == c]) for c in classes]
    accuracies.append(accuracy)
    accuracies.append(accuracy)
    accuracies.append(weighted_accuracy)

    ### Get feature importance
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                  axis=0)

    importances_df = pd.DataFrame({'feature': feature_names,
                                   'importance': importances,
                                   'std': std})
    print(importances_df.sort_values(by=['importance'], ascending=False))
    importances_df.to_csv(GCM+'_'+str(n_samples)+'_importance_individual.csv')

    ### Calculate maximum depth of tree
    depths = [tree.tree_.max_depth for tree in clf.estimators_]
    max_depth = max(depths)

    ### Calculate area under ROC curve for fueltype 1
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,
                                                                    y_pred,
                                                                    pos_label=1)

    ### Grab classification report
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report['accuracy'] = accuracies
    df_report.to_csv('csv/'+GCM+'/'+GCM+'_'+str(n_samples)+'_report.csv')

    return(y_test,y_pred)

ML_function(args.GCM,args.n_samples,False)
