for scen in RCP45 RCP85; do
    echo ${scen}
    for sens in nofire FHSF; do
        echo ${sens}
        python3.9 timeseries_sens_cpool.py --sens ${sens} --scen ${scen}
    done
done
