import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fueltype_analysis_attributes import \
    Wet_Shrubland, Wet_Shrubland_labels, CM_Wet_Shrubland, \
    Wet_Forest, Wet_Forest_labels, CM_Wet_Forest, \
    Grassland, Grassland_labels, CM_Grassland, \
    Dry_forest, Dry_forest_labels, CM_Dry_forest, \
    Shrubland, Shrubland_labels, CM_Shrubland, \
    High_elevation, High_elevation_labels, CM_High_elevation, \
    Mallee, Mallee_labels, CM_Mallee 

fig=plt.figure(figsize=(9,4))

ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3=fig.add_subplot(1,3,3)

FT_list = Wet_Shrubland + Wet_Forest + Grassland + Dry_forest + \
          Shrubland + High_elevation + Mallee
FT_cols = CM_Wet_Shrubland + CM_Wet_Forest + CM_Grassland + CM_Dry_forest + \
          CM_Shrubland + CM_High_elevation + CM_Mallee
FT_labels = Wet_Shrubland_labels + Wet_Forest_labels + Grassland_labels + \
            Dry_forest_labels + Shrubland_labels + High_elevation_labels + \
            Mallee_labels

pathway=('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
         'dynamics_simulations/CFA/ML/')

def read_score(GCM,ML,score):
    global pathway
    df = pd.read_csv(pathway+'output/csv/csv_scores/scores_report/'+GCM+'/'+
                     ML+'_'+GCM+'_report.csv')
    
    return(df[score].values)

def get_score_ML(ML,score):
    df = pd.DataFrame()
    df['ACCESS1-0'] = read_score('ACCESS1-0',ML,score)
    df['BNU-ESM'] = read_score('BNU-ESM',ML,score)
    df['CSIRO-Mk3-6-0'] = read_score('CSIRO-Mk3-6-0',ML,score)
    df['GFDL-CM3'] = read_score('GFDL-CM3',ML,score)
    df['GFDL-ESM2G'] = read_score('GFDL-ESM2G',ML,score)
    df['GFDL-ESM2M'] = read_score('GFDL-ESM2M',ML,score)
    df['INM-CM4'] = read_score('INM-CM4',ML,score)
    df['IPSL-CM5A-LR'] = read_score('IPSL-CM5A-LR',ML,score)

    ### Get average score across ensemble and standard deviation
    df['Mean'] = df.mean(axis=1)
    df['Std'] = df.std(axis=1)

    return(df['Mean'].values.flatten(), 
           df['Std'].values.flatten())

def stripplot_FG(score,FG,FG_col,axis):

    ### Get scores for three methods tested
    kNN_mean, kNN_std = get_score_ML('kNN',score)
    RF_mean, RF_std = get_score_ML('random_forest',score)
    MLP_mean, MLP_std = get_score_ML('MLP',score)

    ### Create dataframe with average score for each method
    df_mean = pd.DataFrame()
    df_mean['FT'] = sorted(FT_list)
    df_mean['k-Nearest\nNeighbor'] = kNN_mean[:-3]
    df_mean['Random\nForest'] = RF_mean[:-3]
    df_mean['Multilayer\nPerceptron'] = MLP_mean[:-3]

    df_mean['FT'] = pd.Categorical(df_mean['FT'], 
                                   categories=FT_list, 
                                   ordered=True)

    df_mean.sort_values('FT',inplace=True)

    df_std = pd.DataFrame()
    df_std['FT'] = sorted(FT_list)
    df_std['kNN'] = kNN_std[:-3]
    df_std['RF'] = RF_std[:-3]
    df_std['MLP'] = MLP_std[:-3]

    df_mean_melt = df_mean.melt(id_vars =['FT'])
    df_std_melt = df_std.melt(id_vars =['FT'])

    df_mean_melt = df_mean_melt[df_mean_melt['FT'].isin(FG)]

    sns.swarmplot(data=df_mean_melt,
                  x='variable',
                  y='value',
                  hue='FT',
                  legend=False,
                  palette=FG_col,
                  ax=axis,
                  zorder=1
                  )

    df_avg = pd.DataFrame()
    df_avg['Label'] = df_mean.iloc[-1].index[1:].values
    df_avg['Average'] = df_mean.iloc[-1].values[1:]

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
        
    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.set_xlabel('')
    axis.set_ylabel('')
    axis.set_ylim(-0.05,1)

stripplot_FG('accuracy',FT_list,FT_cols,ax1)
stripplot_FG('precision',FT_list,FT_cols,ax2)
stripplot_FG('recall',FT_list,FT_cols,ax3)

ax1.set_title('a)', loc='left')
ax1.set_title('Accuracy')
ax2.set_title('b)', loc='left')
ax2.set_title('Precision')
ax3.set_title('c)', loc='left')
ax3.set_title('Recall')

plt.tight_layout()
# plt.show()
plt.savefig('figures/scatter_score.pdf')
