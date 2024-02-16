import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def read_in(scen_time,var):
   ds = xr.open_dataset('../input/clim/'+scen_time+'_'+var+'.nc')
   if var == 'annual_tmax':
      return(ds.layer.values.flatten()-273.15)
   else:
      return(ds.layer.values.flatten())

fig=plt.figure(figsize=(7.5,9))
ax1=fig.add_subplot(5,2,1)
ax2=fig.add_subplot(5,2,2)
ax3=fig.add_subplot(5,2,3)
ax4=fig.add_subplot(5,2,4)
ax5=fig.add_subplot(5,2,5)
ax6=fig.add_subplot(5,2,6)
ax7=fig.add_subplot(5,2,7)
ax8=fig.add_subplot(5,2,8)
ax9=fig.add_subplot(5,2,9)
ax10=fig.add_subplot(5,2,10)

fig.subplots_adjust(hspace=0.45)
fig.subplots_adjust(wspace=0.3)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.1)
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.95)
    
def make_kde_plot(var,axis1,axis2):
   # Create DataFrame for RCP4.5 scenario
   df_rcp45 = pd.DataFrame({
      'Historical': read_in('history_20002015', var),
      'RCP4.5 (2045-2060)': read_in('rcp45_20452060', var),
      'RCP4.5 (2085-2100)': read_in('rcp45_20852100', var)
   })

   # Create DataFrame for RCP8.5 scenario
   df_rcp85 = pd.DataFrame({
      'Historical': read_in('history_20002015', var),
      'RCP8.5 (2045-2060)': read_in('rcp85_20452060', var),
      'RCP8.5 (2085-2100)': read_in('rcp85_20852100', var)
   })

   df_rcp45.plot.kde(color=['tab:grey', '#5eccab', '#00678a'], 
                     legend=False, ax=axis1)
   df_rcp85.plot.kde(color=['tab:grey', '#e6a176', '#984464'], 
                     legend=False, ax=axis2)

   for ax in [axis1, axis2]:
      ax.spines['right'].set_visible(False)
      ax.spines['top'].set_visible(False)
   
   # Dictionary to map variable to limits and labels
   var_info = {
      'annual_tmax': {'limits': (7, 32),
                      'label': 'T$_{max}$ [$^\circ$C]'
                     },
      'annual_pr': {'limits': (0, 2200),
                    'label': 'MAP [mm]'
                    },
      'pr_seasonality': {'limits': (0, 0.1),
                         'label': 'Precipitation seasonality [-]'
                         },
      'annual_rh': {'limits': (20, 100),
                    'label': 'RH$_{min}$ [%]'
                   },
      'lai_jan_5km': {'limits': (-0.05, 4),
                      'label': 'LAI$_{opt}$ [m$^2$ m$^{-2}$]'
                      }
                      }

   # Set limits and label based on var
   limits = var_info[var]['limits']
   label = var_info[var]['label']

   for ax in [axis1, axis2]:
      ax.set_xlim(limits)
      ax.set_xlabel(label)
   
make_kde_plot('annual_tmax',ax1,ax2)
make_kde_plot('annual_pr',ax3,ax4)
make_kde_plot('pr_seasonality',ax5,ax6)
make_kde_plot('annual_rh',ax7,ax8)
make_kde_plot('lai_jan_5km',ax9,ax10)

axes = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,ax10]
title_indices = ['a)','b)','c)','d)','e)','f)','g)','h)','i)','j)']

for ax,ti in zip(axes,title_indices):
   ax.set_title(ti, loc='left')

ax1.set_title('RCP4.5', loc='center')
ax2.set_title('RCP8.5', loc='center')

legend_elements = [
    Line2D([0], [0], color='tab:grey', lw=2, label='Historical'),
    Line2D([0], [0], color='white', lw=2, label=''),
    Line2D([0], [0], color='#5eccab', lw=2, label='RCP4.5 (2045-2060)'),
    Line2D([0], [0], color='#00678a', lw=2, label='RCP4.5 (2085-2100)'),
    Line2D([0], [0], color='#e6a176', lw=2, label='RCP8.5 (2045-2060)'),
    Line2D([0], [0], color='#984464', lw=2, label='RCP8.5 (2085-2100)'),
]

# Create a custom legend and place it outside the figure at the bottom center
legend = ax10.legend(handles=legend_elements, 
                     loc='upper center', 
                     bbox_to_anchor=(-0.2, -0.45), 
                     ncol=3,
                     frameon=False)

fig.align_ylabels()
# plt.show()
plt.savefig('figures/clim_pdf.pdf')
