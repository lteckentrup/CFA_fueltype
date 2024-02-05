### See Frost et al. 2018 for reference (AWRA documentation)
# scen='historical'
# scen='rcp45'
scen='rcp85'

real='r1i1p1'

BC='r240x120-MRNBC-AWAP'
#BC='CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP' 

for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
    cdo -L -b F64 -expr,"pe=610.8*(17.27*(tmin-273.15)/(tmin-273.15+237.3));" \
        tmin/${GCM}_${scen}_r1i1p1_${BC}_tmin.nc \
        pe/${GCM}_${scen}_r1i1p1_${BC}_pe.nc

    cdo -L -b F64 -expr,"pes=610.8*(17.27*(temp-273.15)/(temp-273.15+237.3));" \
        temp/${GCM}_${scen}_r1i1p1_${BC}_temp.nc \
        pes/${GCM}_${scen}_r1i1p1_${BC}_pes.nc

    cdo -L -b F64 -setattribute,rhum@unit='1' -chname,pe,rhum \
        -setrtoc,1,100000,1 -div \
        pe/${GCM}_${scen}_r1i1p1_${BC}_pe.nc \
        pes/${GCM}_${scen}_r1i1p1_${BC}_pes.nc \
        rh/${GCM}_${scen}_r1i1p1_${BC}_rh.nc
    ncatted -O -a standard_name,rhum,c,c,relative_humidity rh/${GCM}_${scen}_r1i1p1_${BC}_rh.nc
    ncatted -O -a units,rhum,c,c,1 rh/${GCM}_${scen}_r1i1p1_${BC}_rh.nc
done
