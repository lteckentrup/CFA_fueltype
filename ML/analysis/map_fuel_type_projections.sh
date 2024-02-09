### For report, I only plotted the ensemble aggregate but script should also run
### on individual GCMs. E.g.:
### python3.9 map_mode_projections.py --GCM ACCESS1-0 --scen rcp45 --timespan mid

for scen in rcp45 rcp85; do
    echo ${scen}
    for timespan in mid long; do
        echo ${timespan}
        python3.9 map_mode_projections.py --GCM 'mode' --scen ${scen} \
                                          --timespan ${timespan}
    done
done
