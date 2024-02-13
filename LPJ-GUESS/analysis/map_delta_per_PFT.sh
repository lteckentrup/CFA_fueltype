for var in fpc cmass clitter; do
    for scen in RCP45 RCP85; do
        python3.9 map_delta_per_PFT.py --var ${var} --scen ${scen} \
                                       --first_year 2045 --last_year 2059
        python3.9 map_delta_per_PFT.py --var ${var} --scen ${scen} \
                                       --first_year 2085 --last_year 2099
    done
done
