for var in fpc cmass clitter; do
    echo ${var}
    for scen in RCP45 RCP85; do
        echo ${scen}
        for sens in nofire FHSF; do
            echo ${sens}
            python3.9 map_sens_per_PFT.py --var ${var} --scen ${scen} \
                                          --sens ${sens} --first_year '2045' \
                                          --last_year '2059'
            python3.9 map_sens_per_PFT.py --var ${var} --scen ${scen} \
                                          --sens ${sens} --first_year '2085' \
                                          --last_year '2099'
        done
    done
done
