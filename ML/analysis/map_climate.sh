for scen in rcp45 rcp85; do
    echo ${scen}
    for timespan in mid long; do
        echo ${timespan}
        python3.9 map_climate.py --ensstat '' \
                                 --scen ${scen} \
                                 --timespan ${timespan}

        python3.9 map_climate.py --ensstat 'CV' \
                                 --scen ${scen} \
                                 --timespan ${timespan}
    done
done
