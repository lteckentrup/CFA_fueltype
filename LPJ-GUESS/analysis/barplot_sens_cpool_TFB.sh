for sens in nofire FHSF; do
    echo ${sens}
    python3.9 barplot_sens_cpool_TFB.py --sens ${sens}
done
