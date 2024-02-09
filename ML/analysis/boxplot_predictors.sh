### Loop through all variables; single command would be something like
### python3.9 boxplot_predictors.py --var map

### List of predictor variables
variables=('map' 'rad.short.jan' 'rad.short.jul' 'lai.opt.mean' 'wi' \
           'soil.depth' 'clay' 'awc' 'tmax.mean' 'soil.density' 'rh.mean' \
           'curvature_profile' 'curvature_plan' 'pr.seaonality' 'thorium_pot' \
           'uran_pot')

### Loop through all variables and run boxplot script
for var in "${variables[@]}"; do
    echo ${var}
    python3.9 boxplot_predictors.py --var ${var}
done
