import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from fueltype_analysis_attributes import \
    Wet_Shrubland, Wet_Shrubland_labels, CM_Wet_Shrubland, \
    Wet_Forest, Wet_Forest_labels, CM_Wet_Forest, \
    Grassland, Grassland_labels, CM_Grassland, \
    Dry_forest, Dry_forest_labels, CM_Dry_forest, \
    Shrubland, Shrubland_labels, CM_Shrubland, \
    High_elevation, High_elevation_labels, CM_High_elevation, \
    Mallee, Mallee_labels, CM_Mallee 

fig=plt.figure(figsize=(12,5))

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0.2)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.05)
fig.subplots_adjust(bottom=0.47)
fig.subplots_adjust(top=0.95)

ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3=fig.add_subplot(1,3,3)

GCM_list = ['ACCESS1-0','BNU-ESM','CSIRO-Mk3-6-0','GFDL-CM3','GFDL-ESM2G',
            'GFDL-ESM2M','INM-CM4','IPSL-CM5A-LR','MRI-CGCM3']
pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/')

def read_score(GCM,ML,score):
    global pathway
    df = pd.read_csv(pathway+'output/csv/csv_scores/scores_report/'+GCM+'/'+
                     ML+'_'+GCM+'_report.csv')   
    return(df[score].values)

def get_score_ML(ML,score):
    global GCM_list

    ### Dataframe with evaluation score for each GCM
    df = pd.DataFrame()
    for GCM in GCM_list:
       df[GCM] = read_score(GCM,ML,score)

    ### Get average score across ensemble and standard deviation
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)

    return(df['Mean'].values.flatten(), 
           df['Std'].values.flatten())

def plot_stripplot(score,axis):
    ### Prepare colormap: First get list of fuel types and respective color
    FT_list = Wet_Shrubland + Wet_Forest + Grassland + Dry_forest + \
              Shrubland + High_elevation + Mallee
    FT_cols = CM_Wet_Shrubland + CM_Wet_Forest + CM_Grassland + CM_Dry_forest + \
              CM_Shrubland + CM_High_elevation + CM_Mallee

    ### Build dataframe
    df_FT = pd.DataFrame()
    df_FT['fuel_types'] = FT_list
    df_FT['colors'] = FT_cols

    ### Sort by fuel type
    df_FT.sort_values(by='fuel_types',inplace=True)

    ### New lists where fuel types and respective colors are ordered by fuel type
    FT_list_sorted = df_FT['fuel_types'].to_list()
    FT_cols_sorted = df_FT['colors'].to_list()

    ### Get scores for each fuel type for three ML methods tested
    kNN_mean, kNN_std = get_score_ML('kNN',score)
    RF_mean, RF_std = get_score_ML('random_forest',score)
    MLP_mean, MLP_std = get_score_ML('MLP',score)

    ### Create dataframe with average score for each method
    df_mean = pd.DataFrame()
    ### Get fuel types and add three extra rows macro average, accuracy, and 
    ### weighted average that come out of report
    df_mean['FT'] = sorted(FT_list_sorted)+['macro avg']+['accuracy']+['weighted avg']

    ### Get scores for each fuel type and each method
    df_mean['k-Nearest\nNeighbor'] = kNN_mean
    df_mean['Random\nForest'] = RF_mean
    df_mean['Multilayer\nPerceptron'] = MLP_mean
    
    ### Melt dataframe to allow swarmplot with hue; select everything except last
    ### three rows (macro average, accuracy, and weighted average)
    df_mean_melt = df_mean[:-3].melt(id_vars =['FT'])

    ### Swarmplot for eval scores for individual fuel types
    sns.swarmplot(data=df_mean_melt,
                  x='variable',
                  y='value',
                  hue='FT',
                  legend=False,
                  palette=FT_cols_sorted,
                  ax=axis,
                  zorder=1
                  )

    ### Overall score in the bottom three lines of the dataframes - 
    ### take last one (weighted average)
    df_avg = pd.DataFrame()
    df_avg['Label'] = df_mean.iloc[-1].index[1:].values
    df_avg['Average'] = df_mean.iloc[-1].values[1:]

    ### Plot weighted average of scores as black diamond marker
    sns.scatterplot(data=df_avg,
                    x='Label',
                    y='Average',
                    marker='D',
                    s=100,
                    color='.2',
                    legend=False,
                    ax=axis,
                    zorder=2
                    )
    
    ### Drop spines
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    ### Remove labels (set later)
    axis.set_xlabel('')
    axis.set_ylabel('')

    ### Set same y-lims for each plot
    axis.set_ylim(-0.05,1)

scores = ['accuracy','precision','recall']
title_labels = ['Accuracy','Precision','Recall']
title_indices = ['a)','b)','c)']
axes=[ax1,ax2,ax3]

for s,tl,ti,ax in zip(scores,title_labels,title_indices,axes):
    plot_stripplot(s,ax)
    ax.set_title(tl)
    ax.set_title(ti, loc='left')

### Plot custom legend: Group into broad fuel groups
### Define labels
labels = {
    'Wet_Forest': Wet_Forest_labels,
    'High_elevation': High_elevation_labels,
    'Wet_Shrubland': Wet_Shrubland_labels,
    'Mallee': Mallee_labels,
    'Dry_forest': Dry_forest_labels,
    'Grassland': Grassland_labels,
    'Shrubland': Shrubland_labels,    
}

### Define colors
colors = {
    'Wet_Forest': CM_Wet_Forest,
    'High_elevation': CM_High_elevation,
    'Wet_Shrubland': CM_Wet_Shrubland,
    'Mallee': CM_Mallee,
    'Dry_forest': CM_Dry_forest,
    'Grassland': CM_Grassland,   
    'Shrubland': CM_Shrubland,
}

### Create separate legend handels for each fuel group. 
legend_handles = {}
for key in labels.keys():
    legend_handles[key] = [
        Line2D([0], [0], marker='o', color='w', label=label, 
               markerfacecolor=color, markersize=8, linestyle='')
        for label, color in zip(labels[key], colors[key])
    ]
    
### Define location of individual legends
x_offsets = [-0.15, 0.35, 0.8, 1.35, 1.9, 2.37, 2.9]
y_offsets = [-0.83, -0.74, -0.68, -0.95, -0.99, -0.625, -0.53]

### Plot all legends - need to store legend in variable, calling ax.legend
### multiple times overwrites previous legend
for i, key in enumerate(labels.keys()):
    legend = ax1.legend(handles=legend_handles[key], 
                        ncols=1,
                        loc='lower left',
                        bbox_to_anchor=(x_offsets[i], y_offsets[i]),
                        fontsize=9,
                        frameon=False)
    ax1.add_artist(legend)

### Save figure
plt.savefig('figures/scatter_ML_scores.pdf')
