import seaborn as sns

'''
Group individual fuel types into discussed broader group fuels - 
this is consistently quite clunky I'm afraid
'''

### Wet shrubland
Wet_Shrubland = [3001,3003,3014,3023,3029]
Wet_Shrubland_labels = ['Moist whrubland','Low flammable shrubs',
                        'Riparian shrubland','Wet heath',
                        'Ephemeral grass/\nsedge/ herbs']
pal = sns.color_palette('Blues',len(Wet_Shrubland))
CM_Wet_Shrubland = pal.as_hex()

### Wet forest
Wet_Forest = [3002,3006,3007,3011,3012,3013,3015]
Wet_Forest_labels = ['Moist woodland','Forest with shrub',
                     'Forest herb-rich', 'Wet forest shrub &\nwiregrass',
                     'Damp forest shrub', 'Riparian forest shrub', 
                     'Rainforest']
pal = sns.color_palette('Greens',len(Wet_Forest))
CM_Wet_Forest = pal.as_hex()

### Grassland
Grassland = [3004,3020,3037,3046] 
Grassland_labels = ['Moist sedgeland/\ngrassland',
                    'Temperate grassland/\nsedgeland',
                    'Wet herbland','Eaten out grass'] 
pal = sns.color_palette('pink',len(Grassland))
CM_Grassland = pal.as_hex()

Dry_forest = [3005,3008,3009,3022,3028,3043]
Dry_forest_labels = ['Woodland heath',
                     'Dry open forest\nshrubs/ herbs',
                     'Woodland grass/\nherb-rich', 
                     'Woodland bracken/\nshrubby',
                     'Woodland Callitris/\nBelah',
                     'Gum woodland\ngrass/ herbs']
pal = sns.color_palette('Reds',len(Dry_forest))
CM_Dry_forest = pal.as_hex()

### Shrubland
Shrubland = [3010,3021,3024]
Shrubland_labels = ['Sparse shrubland',
                    'Broombush/ Shrubland/\nTea-tree',
                    'Dry Heath']
pal = sns.color_palette('Purples',len(Shrubland))
CM_Shrubland = pal.as_hex()

### High elevation
High_elevation = [3016,3017,3018,3019]
High_elevation_labels = ['High elevation\ngrassland',
                         'High elevation\nshrubland/ heath',
                         'High elevation\nwoodland shrub',
                         'High elevation\nwoodland grass']
pal = sns.color_palette('Greys',len(High_elevation))
CM_High_elevation = pal.as_hex()

### Mallee
Mallee = [3025,3026,3027,3048,3049,3050,3051] 
Mallee_labels = ['Mallee shrub/ heath','Mallee spinifex',
                 'Mallee chenopod','Mallee dry heath',
                 'Mallee shrub/\nheath (costata)',
                 'Mallee spinifex\n(costata)',
                 'Mallee shrub\nheath (discontinuous)'] 
pal = sns.color_palette('Oranges',len(Mallee))
CM_Mallee = pal.as_hex()
