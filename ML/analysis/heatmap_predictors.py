import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Read in data
df = pd.read_csv('/data/hiestorage/WorkingData/MEDLYN_GROUP/PROJECTS/'
                 'dynamics_simulations/CFA/ML/input/cache/'
                 'ft.ACCESS1-0.history.csv')

### Create custom order of predictors: Group according to report (climate, 
### topography rad, soil properties, topography curvature, uranium + thorium)
order = ['tmax.mean', 'map', 'rh.mean', 'pr.seaonality', 'lai.opt.mean',
         'rad.short.jan', 'rad.short.jul', 'soil.depth', 'soil.density', 
         'clay', 'awc', 'wi', 'curvature_plan', 
         'curvature_profile', 'uran_pot', 'thorium_pot'] 

### Clean dataframe and drop irrelevant columns
df.drop(columns=['x','y','lon','lat','ft','uranium','thorium',
                 'potassium','elevation'],
                 inplace=True)

### Order dataframe following custom order
df = df[order]

### Calculate pearson correlation
corr = df.corr()

### Only show lower triangle in final figure: create mask upper half
mask = np.triu(np.ones_like(corr, dtype=bool))

### Set up figure
fig, ax = plt.subplots(figsize=(11, 9))

### Plot heatmap
sns.heatmap(corr, 
            mask=mask, 
            cmap='seismic', 
            vmin=-1,
            vmax=1, 
            center=0,
            square=True, 
            linewidths=.5,
            annot=True,
            fmt='.1f',
            cbar_kws={'shrink': .8},
            ax=ax)

### Assign more readable predictor labels for plot
labels = ['T$_{max}$', 'MAP', 'RH$_{min}$', 'Rainfall\nseasonality', 
          'LAI$_{opt}$', 'Rad$_{Jan}$', 'Rad$_{Jul}$', 'Soil depth', 
          'Soil density', 'Clay fraction', 'AWC', 'Wetness index', 
          'Plan curvature', 'Profile curvature', 'Uranium-Potassium\nratio',
          'Thorium-Potassium\nratio', 
          ]

### Set labels
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.tight_layout()
# plt.show()
plt.savefig('figures/heatmap_predictors.pdf')
