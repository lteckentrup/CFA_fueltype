# CFA_fueltype

The ```prep_data``` directory contains all scripts used to prepare the prediction files for the machine learning.

```get_index.py``` selects the coordinates and fuel types from the current fuel type distribution (provided by CFA).

```prep_static.R``` maps all static predictors to the coordinates of the fuel type map.

```prep_CMIP_hist.R``` maps the historical (2000-2015) GCM simulations to the coordinates of the fuel type map and combines the static and climate forcing.

```prep_CMIP_fut.R``` maps the future GCM projections to the coordinates of the fuel type map and combines the static and climate forcing (mid century: 2045-2060; end century: 2085-2100).

```prep_CMIP_hist.R``` and ```prep_CMIP_fut.R``` take arguments to run. They can either be run individually from command line but there are also two bash scripts that loop through all GCMs, scenarios and timespans (```get_hist_predictors.sh``` and ```get_fut_predictors.sh```).
