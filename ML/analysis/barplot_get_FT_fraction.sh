### Get fuel type fraction for each RCP scenario and timespan

for RCP in rcp45 rcp85; do
    echo ${RCP}
    for timespan in mid long; do
        echo ${timespan}
        python3.9 barplot_get_FT_fraction.py --scen ${RCP} --timespan ${timespan}
    done
done
