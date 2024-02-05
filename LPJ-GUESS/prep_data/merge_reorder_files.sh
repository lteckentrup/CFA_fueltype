for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
    echo ${GCM}
    for scen in rcp45 rcp85; do
        echo ${scen}
        for var in temp prec insol rh tmax tmin wind; do
            echo ${var}
            cdo -L -b F64 -invertlat -mergetime \
                ${var}/${GCM}_historical_r1i1p1_r240x120-MRNBC-AWAP_${var}.nc \
                ${var}/${GCM}_${scen}_r1i1p1_r240x120-MRNBC-AWAP_${var}.nc \
                ${var}/${GCM}_${scen}_r1i1p1_r240x120-MRNBC-AWAP_${var}_1960-2100.nc
        done
    done
done


for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
    echo ${GCM}
    for scen in rcp45 rcp85; do
        echo ${scen}
        for var in temp prec insol rh tmax tmin wind; do
            python reorder.py --GCM 'CNRM-CERFACS-CNRM-CM5' --scen ${scen} --var ${var}
            python reorder.py --GCM 'CSIRO-BOM-ACCESS1-0' --scen ${scen} --var ${var}
            python reorder.py --GCM 'MIROC-MIROC5' --scen ${scen} --var ${var}
            python reorder.py --GCM 'NOAA-GFDL-GFDL-ESM2M' --scen ${scen} --var ${var}
        done
    done
done
