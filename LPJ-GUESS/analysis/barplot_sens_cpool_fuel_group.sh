for sens in nofire FHSF; do
    echo ${sens}
    python3.9 barplot_sens_cpool_fuel_group.py --sens ${sens}
done
