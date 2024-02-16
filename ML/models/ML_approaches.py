### Import relevant sklearn libraries
### Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
'''
I tried a bunch of resampling methods to address the imbalance in the fuel type 
distribution but in the end wrote a function that undersampled all classes to a 
specified threshold. There were a few fuel types where the threshold exceeded 
the sample size but I didn't want to reduce the overall sample size too 
aggressively
'''
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import NeighbourhoodCleaningRule

### Methods
### PCA for dimension reduction
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

### Machine learning approaches
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

### Evaluation of results
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_curve, auc

### Import tools for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import matplotlib as mpl

### Other libraries
import pandas as pd
import random
import numpy as np
import joblib
import argparse

'''
Initialise argument parsing: 
GCM for the different ensemble members: ACCESS1-0 BNU-ESM 
CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M INM-CM4 
IPSL-CM5A-LR MRI-CGCM3
classifier for the machine learning methods: Nearest Neighbor,
Random forest, Neural Network
model_name short for ML method (goes into file name): kNN, RF, MLP
reduce_dim for PCA dimension reduction: 'PCA' or 'None'
'''

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--classifier', type=str, required=True)
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--reduce_dim', type=str, required=True)
args = parser.parse_args()

### Set seed for reproducability (results are not sensitive to seed though)
random.seed(42)

def get_data(GCM):
    ### Read in features + target
    df = pd.read_csv('../input/cache/pred.'+GCM+'.history.csv')

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

def prep_data(GCM,reduce_dim):
    df = get_data(GCM)

    '''
    I tested some sampling approaches, such as random undersampling, 
    random oversampling, neighbourhood cleaning rule, SMOTE - but got best 
    results by randomly selecting 125'190 samples for each class (see below). 
    This was pretty much the maximum size I could use to train the models. 
    Theoretically, data only need to be normalised for MLP but I normalised them
    for all three approaches 
    '''
    
    ### Set threshold for undersampling
    n_samples = 125190

    ### Set up dataframe for resampled data
    df_sampled = pd.DataFrame()

    ### Loop through all fuel types
    for value in df['FT'].unique():
        ### Get fuel type
        subset = df[df['FT'] == value]
        
        ### 5 classes have less than n_samples
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

    ### Reduce dimensions using PCA
    if reduce_dim == 'PCA':
        ### Standardize predictors
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        ### Perform PCA on standardized predictors
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)

        ### Calculate explained variance ratios for each principal component
        explained_variance_ratio = pca.explained_variance_ratio_

        ### Derive number of components: Keep at least 90% of the variance
        target_k = 0.9
        k = (explained_variance_ratio.cumsum() < target_k).sum() + 1

        ### Perform PCA to reduce dimensions
        pca = PCA(n_components=k)
        X_pca = pca.fit_transform(X_scaled)

        ### Split data in to training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, 
                                                            test_size=0.3)

        ### Grab all feature names
        feature_names_full = X.columns

        ### Grab PCA components
        components = pca.components_

        ### Get feature names after PCA
        feature_names = []

        ### Iterate over the first n rows of the components matrix
        for i in range(k):
            # Get i-th row of the components matrix
            row = components[i,:] 

            # Get index of the element with the highest absolute value
            max_index = np.argmax(np.abs(row))
            
            # Get the name of the corresponding feature
            feature_name = feature_names_full[max_index]

            # Print the feature name
            feature_names.append(feature_name)
    
    ### Use all predictors
    elif reduce_dim == 'None':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ### Split data in to training and test datasets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                            test_size=0.3)   

        ### Get feature names
        feature_names = X.columns

    return(X_train, X_test, y_train, y_test, feature_names)

### Hypertuning using GridSearchCV
def hypertuning(classifier, GCM, reduce_dim):
    ### Read in training data
    X_train, X_test, y_train, y_test, feature_names = prep_data(GCM,
                                                                reduce_dim)

    ### Choose parameters for hypertuning. You could probably gain a decimal 
    ### or two adjusting some of those but because that datasets are so big
    ### computational resources were limiting a bit too

    classifiers = {
        'Nearest Neighbor': (KNeighborsClassifier(), 
                             {'n_neighbors': [9,13,17,21,25,29], 
                              'p': [1, 2, 3],
                              'weights': ['uniform', 'distance']
                              }
                             ),
        'Random Forest': (RandomForestClassifier(), 
                          {'n_estimators': [200, 300, 500], 
                           'min_samples_split': [2, 5, 10],
                           ### I kept the class weights in because the data are 
                           ### still not *quite* balanced. See above
                           'class_weight': [None, 'balanced', 'balanced_subsample']
                          }
                         ),                   
        'Neural Network': (MLPClassifier(max_iter=100),
                           {'hidden_layer_sizes': [(10,30,10),(10,10)],
                            'activation': ['tanh', 'relu'],
                            'solver': ['sgd', 'adam'],
                            'alpha': [0.0001, 0.05],
                            'learning_rate': ['constant','adaptive']
                            }
                           )
    }

    ### Get the classifier and hyperparameter grid for the specified classifier
    clf, param_grid = classifiers[classifier]

    if param_grid:
        ### Create the grid search object
        grid_search = GridSearchCV(clf, 
                                   param_grid, 
                                   cv=5, 
                                   n_jobs=-1, 
                                   scoring='accuracy')

        ### Fit the grid search object to the training data
        grid_search.fit(X_train, y_train)

        ### Get the best hyperparameters
        best_params = grid_search.best_params_
        print(best_params)

    else:
        ### Classifier does not have hyperparameters
        best_params = {}

def ML_function(GCM,reduce_dim,classifier,model_name):
    ### Grab data
    X_train, X_test, y_train, y_test, feature_names = prep_data(GCM,
                                                                reduce_dim)

    ### Set hyper parameters based on grid search
    params_by_classifier = {
        'Nearest Neighbor': {
            'PCA': {'n_neighbors': 29, 
                    'p': 1, 
                    'weights': 'distance'}, 
            'None': {'n_neighbors': 25, 
                     'p': 1, 
                     'weights': 'distance'}
                    },
        'Random Forest': {
            'PCA': {'max_depth': 20, 
                    'min_samples_split': 10, 
                    'n_estimators': 500, 
                    'class_weight': 'balanced_subsample', 
                    'n_jobs':-1},
            'None': {'max_depth': 20, 
                     'min_samples_split': 10, 
                     'n_estimators': 500, 
                     'class_weight': 'balanced_subsample', 
                     'n_jobs':-1}
                     },
        'Neural Network': {
            'PCA': {'activation': 'tanh',
                    'alpha': 0.0001, 
                    'hidden_layer_sizes': (10, 30, 10),
                    'learning_rate': 'adaptive',
                    'solver': 'adam'},
            'None': {'activation': 'tanh',
                     'alpha': 0.0001, 
                     'hidden_layer_sizes': (10, 30, 10),
                     'learning_rate': 'constant',
                     'solver': 'adam'},
                     }
                     }

    ### Define classifiers
    classifiers = {
        'Nearest Neighbor': KNeighborsClassifier,
        'Random Forest': RandomForestClassifier,
        'Neural Network': MLPClassifier
    }

    ### Pass optimal paramters
    opt_params = params_by_classifier[classifier][reduce_dim]

    ### Set up model
    clf = classifiers[classifier](**opt_params)

    ### Fit model
    clf.fit(X_train, y_train)

    ### Save model for predictions
    joblib.dump(clf, 'pkl/'+GCM+'/'+model_name+'_'+GCM+'.pkl')
    
    ### Predict on test data
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    class_names = clf.classes_  # Get the class names from the classifier

    df_y = pd.DataFrame(y_pred_prob, columns=class_names)
    df_y['win'] = y_pred

    df_y.to_csv('csv/'+GCM+'/'+'y_pred_'+GCM+'.csv')

    ### Calculate overall accuracy
    accuracy = clf.score(X_test, y_pred)
    weighted_accuracy = balanced_accuracy_score(y_test, y_pred)

    classes = np.unique(y_test)
    accuracies = [accuracy_score(y_test[y_test == c], 
                                 y_pred[y_test == c]) for c in classes]
    accuracies.append(accuracy)
    accuracies.append(accuracy)
    accuracies.append(weighted_accuracy)

    if classifier == 'Random Forest':
        ### Get feature importance
        importances = clf.feature_importances_

        ### Get importance uncertainty
        std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                    axis=0)

        ### Build dataframe
        importances_df = pd.DataFrame({'feature': feature_names,
                                       'importance': importances,
                                       'std': std})

        ### Print and save to csv
        print(importances_df.sort_values(by=['importance'], ascending=False))
        importances_df.to_csv(GCM+'_importance_individual.csv')

    if classifier == 'Random Forest':
        ### Calculate maximum depth of tree
        depths = [tree.tree_.max_depth for tree in clf.estimators_]
        max_depth = max(depths)

    ### Calculate area under ROC curve for fueltype 1
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,
                                                                    y_pred,
                                                                    pos_label=1)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    print('Area under ROC curve: ',roc_auc)

    ### Grab classification report
    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    df_report['accuracy'] = accuracies
    df_report.to_csv('csv/'+GCM+'/'+model_name+'_'+GCM+'_report.csv')

    return(y_test,y_pred)

# hypertuning(args.classifier, args.GCM, args.reduce_dim)
ML_function(args.GCM,args.reduce_dim,args.classifier,args.model_name)
