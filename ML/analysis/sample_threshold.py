import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('../input/cache/pred.ACCESS1-0.history.csv')

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

### Get fuel type indices and their respective counts
ft, counts = np.unique(df.FT.values,return_counts=True)

### Dataframe for fuel types and counts
df_count = pd.DataFrame()
df_count['FT'] = ft
df_count['count'] = counts

### Sort by count 
df_count_sort = df_count.sort_values(by='count')

### Plot barplot to show count for each fuel type
ax = df_count_sort.plot.bar(x='FT',
                            y='count',
                            color='crimson',
                            rot=90,
                            legend=False,
                            figsize=(10, 6))

### Log y-scale
ax.set_yscale('log')

### Get median value of fuel count - will be used as sampling threshold in ML
ax.axhline(df_count['count'].median(),lw=0.5,c='tab:grey')
print(df_count['count'].median())

### Drop right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

### Set labels
ax.set_ylabel('Fuel type count')
ax.set_xlabel('Fuel type ID')

plt.tight_layout()
plt.savefig('figures/sample_threshold.pdf')
