for GCM in ACCESS1-0 BNU-ESM CSIRO-Mk3-6-0 GFDL-CM3 GFDL-ESM2G GFDL-ESM2M INM-CM4 IPSL-CM5A-LR MRI-CGCM3; do
    echo ${GCM}
    for scen in rcp45 rcp85; do
        echo ${scen}
        python3.9 unseen_clim.py --GCM ${GCM} --scen ${scen} \
                                 --timespan 'mid' --years '20452060'
        python3.9 unseen_clim.py --GCM  ${GCM} --scen ${scen} \
                                 --timespan 'long' --years '20852100'
    done
done
