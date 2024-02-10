for scen in rcp45 rcp85; do
    echo ${scen}
    for timespan in mid long; do
        echo ${timespan}
        python3.9 map_fuel_types_uncertainty.py --scen ${scen} \
                                                --timespan ${timespan}
    done
done
