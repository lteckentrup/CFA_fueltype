for GCM in ACCESS1-0 BNU-ESM CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G INM-CM4 IPSL-CM5A-LR MRI-CGCM3; do
    echo ${GCM}
    for scen in rcp45 rcp85; do
        echo ${scen}
        Rscript make_CMIP_fut.R --GCM ${GCM} --scen ${scen} --time mid --years 20452060
        Rscript make_CMIP_fut.R --GCM ${GCM} --scen ${scen} --time long --years 20852100
    done
done
