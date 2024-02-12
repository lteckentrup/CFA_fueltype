for scen in rcp45 rcp85; do
    echo ${scen}
    for timespan in mid long; do
        echo ${timespan}
        python3.9 map_unseen_clim.py --scen ${scen} --timespan ${timespan}
    done
done
