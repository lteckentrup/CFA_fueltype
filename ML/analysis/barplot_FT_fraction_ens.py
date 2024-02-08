import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import seaborn as sns

'''
Set up dataframe for fuel type fractions, and read in averages and 
standard deviation for all scenarios and timeslices
'''

df = pd.DataFrame()
df['Labels'] = pd.read_csv('csv/count_rcp45_mid.csv')['FT'] # Get fuel type labels
df['Observation'] = pd.read_csv('csv/count_rcp45_mid.csv')['Obs'] # Get observed fuel type fraction

### Projected fuel type fraction averaged across GCM ensemble
df['RCP4.5 (2045-2060)'] = pd.read_csv('csv/count_rcp45_mid.csv')['Avg']
df['RCP4.5 (2085-2100)'] = pd.read_csv('csv/count_rcp45_long.csv')['Avg']
df['RCP8.5 (2045-2060)'] = pd.read_csv('csv/count_rcp85_mid.csv')['Avg']
df['RCP8.5 (2085-2100)'] = pd.read_csv('csv/count_rcp85_long.csv')['Avg']

df['Observation Std'] = 0 # dummy 
### Standard deviation of projected fuel type fraction across GCM ensemble
df['RCP4.5 (2045-2060) Std'] = pd.read_csv('csv/count_rcp45_mid.csv')['Std']
df['RCP4.5 (2085-2100) Std'] = pd.read_csv('csv/count_rcp45_long.csv')['Std']
df['RCP8.5 (2045-2060) Std'] = pd.read_csv('csv/count_rcp85_mid.csv')['Std']
df['RCP8.5 (2085-2100) Std'] = pd.read_csv('csv/count_rcp85_long.csv')['Std']

'''
Group individual fuel types into discussed broader group fuels - 
this is consistently quite clunky I'm afraid
'''

### Wet shrubland
Wet_Shrubland = [3001,3003,3014,3023,3029]
Wet_Shrubland_tick_labels = ['Moist whrubland','Low flammable shrubs',
                             'Riparian shrubland','Wet heath',
                             'Ephemeral grass/\nsedge/ herbs']

### Wet forest
Wet_Forest = [3002,3006,3007,3011,3012,3013,3015]
Wet_Forest_tick_labels = ['Moist woodland','Forest with shrub',
                          'Forest herb-rich', 'Wet forest shrub &\nwiregrass',
                          'Damp forest shrub', 'Riparian forest shrub', 
                          'Rainforest']

### Grassland
Grassland = [3004,3020,3037,3046] 
Grassland_tick_labels = ['Moist sedgeland/\ngrassland',
                         'Temperate grassland/\nsedgeland',
                         'Wet herbland','Eaten out grass'] 

### Dry forest
Dry_forest = [3005,3008,3009,3022,3028,3043]
Dry_forest_tick_labels = ['Woodland heath',
                          'Dry open forest\nshrubs/ herbs',
                          'Woodland grass/\nherb-rich', 
                          'Woodland bracken/\nshrubby',
                          'Woodland Callitris/\nBelah',
                          'Gum woodland\ngrass/ herbs']

### Shrubland
Shrubland = [3010,3021,3024]
Shrubland_tick_labels = ['Sparse shrubland',
                         'Broombush/ Shrubland/\nTea-tree',
                         'Dry Heath']

### High elevation
High_elevation = [3016,3017,3018,3019]
High_elevation_tick_labels = ['High elevation\ngrassland',
                              'High elevation\nshrubland/ heath',
                              'High elevation\nwoodland shrub',
                              'High elevation\nwoodland grass']

### Mallee
Mallee = [3025,3026,3027,3048,3049,3050,3051] 
Mallee_tick_labels = ['Mallee shrub/ heath','Mallee spinifex',
                      'Mallee chenopod','Mallee dry heath',
                      'Mallee shrub/\nheath (costata)',
                      'Mallee spinifex\n(costata)',
                      'Mallee shrub\nheath (discontinuous)'] 

### Create custom order of fuel types to display individual
### fuel types ordered within their broad fuel groups
custom_order = Wet_Forest+High_elevation+Wet_Shrubland+Mallee+\
               Dry_forest+Grassland+Shrubland

### List of xticklabels following custom_order
xticklabels = np.concatenate([Wet_Forest_tick_labels,
                              High_elevation_tick_labels,
                              Wet_Shrubland_tick_labels,
                              Mallee_tick_labels,
                              Dry_forest_tick_labels,
                              Grassland_tick_labels,
                              Shrubland_tick_labels])

### Include label column in dataframe
df['Labels'] = pd.Categorical(df['Labels'], 
                              categories=custom_order, 
                              ordered=True)

### Sort the DataFrame based on the custom order
df.sort_values('Labels',inplace=True)

### Plot grouped barplot
fig, ax = plt.subplots(figsize=(16,9))
df.set_index('Labels')[['Observation',
                        'RCP4.5 (2045-2060)', 
                        'RCP4.5 (2085-2100)', 
                        'RCP8.5 (2045-2060)', 
                        'RCP8.5 (2085-2100)']].plot.bar(
                            yerr=df[['Observation Std',
                                     'RCP4.5 (2045-2060) Std', 
                                     'RCP4.5 (2085-2100) Std', 
                                     'RCP8.5 (2045-2060) Std', 
                                     'RCP8.5 (2085-2100) Std']].values.T, 
                        color=['tab:grey', '#5eccab', '#00678a', '#e6a176', '#984464'], ### bar colors
                        width=0.7,
                        ax=ax,
                        error_kw=dict(ecolor='black',linewidth=0.5, lolims=False, capsize=0), ### errorbar
                        zorder=2
                        )

### Update figure labels
ax.set_xlabel('Fuel type',fontsize=13)
ax.set_ylabel('Fraction of full domain [-]',fontsize=13)
ax.set_ylim(-0.001,0.28)

ax.set_xticklabels(xticklabels, rotation=90, fontsize=13)      
ax.tick_params(axis='y', labelsize=13)
ax.set_xlabel('')

### Include figure legend
ax.legend(['Observation',
           'RCP4.5 (2045-2060)', 'RCP4.5 (2085-2100)',
           'RCP8.5 (2045-2060)', 'RCP8.5 (2085-2100)'],
           frameon=False,
           fontsize=13)

### Drop right and top spine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

### For easier interpretation, include patches with colors according 
### to broad fuel group. First get colors (consistent across report),
### then plot patch according to defined position

def plot_fuel_group_patch(x_start,x_end,colormap,n_fueltypes):
    pal = sns.color_palette(colormap,n_fueltypes)
    CM_FT = pal.as_hex()

    x_left = x_start
    x_right = x_end
    y_bottom = 0
    y_top = 0.28

    rectangle = patches.Rectangle((x_left, y_bottom), 
                                  x_right - x_left, 
                                  y_top - y_bottom, 
                                  linewidth=1, 
                                  edgecolor=CM_FT[0], 
                                  facecolor=CM_FT[0],
                                  zorder=1)
    plt.gca().add_patch(rectangle)

plot_fuel_group_patch(-0.35,6.5,'Greens',7)
plot_fuel_group_patch(6.5,10.5,'Greys',4)
plot_fuel_group_patch(10.5,15.5,'Blues',5)
plot_fuel_group_patch(15.5,22.5,'Oranges',7)
plot_fuel_group_patch(22.5,28.5,'Reds',6)
plot_fuel_group_patch(28.5,32.5,'pink_r',4)
plot_fuel_group_patch(32.5,35.5,'Purples',3)

# ### Wet Forest
# pal = sns.color_palette('Greens',7)
# CM_Wet_Forest = pal.as_hex()

# x_start = -0.35
# x_end = 6.5

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Wet_Forest[0], 
#                               facecolor=CM_Wet_Forest[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### High Elevation
# pal = sns.color_palette('Greys',4)
# CM_High_elevation = pal.as_hex()

# x_start = 6.5
# x_end = 10.5

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_High_elevation[0], 
#                               facecolor=CM_High_elevation[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### Wet shrubland
# pal = sns.color_palette('Blues',5)
# CM_Wet_Shrubland = pal.as_hex()

# x_start = 10.5
# x_end = 15.5
# y_start = 0
# y_end = 0.28

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Wet_Shrubland[0], 
#                               facecolor=CM_Wet_Shrubland[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### Mallee
# pal = sns.color_palette('Oranges',7)
# CM_Mallee = pal.as_hex()

# x_start = 15.5
# x_end = 22.5

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Mallee[0], 
#                               facecolor=CM_Mallee[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### Dry Forest
# pal = sns.color_palette('Reds',6)
# CM_Dry_forest = pal.as_hex()

# x_start = 22.5
# x_end = 28.5

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Dry_forest[0], 
#                               facecolor=CM_Dry_forest[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### Grasslands
# pal = sns.color_palette('pink_r',4)
# CM_Grassland = pal.as_hex()

# x_start = 28.5
# x_end = 32.5
# y_start = 0
# y_end = 0.28

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Grassland[0], 
#                               facecolor=CM_Grassland[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)

# ### Shrubland
# pal = sns.color_palette('Purples',3)
# CM_Shrubland = pal.as_hex()

# x_start = 32.5
# x_end = 35.5

# rectangle = patches.Rectangle((x_start, y_start), 
#                               x_end - x_start, 
#                               y_end - y_start, 
#                               linewidth=1, 
#                               edgecolor=CM_Shrubland[0], 
#                               facecolor=CM_Shrubland[0],
#                               zorder=1)
# plt.gca().add_patch(rectangle)


### Save figure
plt.tight_layout()
plt.savefig('figures/barplot_FT_fraction_ens.pdf')
