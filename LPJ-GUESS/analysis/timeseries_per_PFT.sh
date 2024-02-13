for var in fpc cmass clitter; do
    echo ${var}
    python3.9 timeseries_per_PFT.py --var ${var}
done
