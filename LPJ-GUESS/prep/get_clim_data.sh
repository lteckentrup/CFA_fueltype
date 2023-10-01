pathway=pathway
real='r1i1p1'

### Select simulation
scen='historical'
# scen='rcp45'
# scen='rcp85'

### Select bias correction type
# BC='r240x120-MRNBC-AWAP'
BC='CSIRO-CCAM-r3355-r240x120-ISIMIP2b-AWAP' 


for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
    ### Get temperature files
    cdo -L -b F64 -chname,tasmax,tmax -sellonlatbox,140,150.1,-33,-40 \
                ${pathway}${GCM}/${scen}/${real}/${BC}/*/*/tasmax/* \
                tmax/${GCM}_${scen}_${real}_${BC}_tmax.nc
    cdo -L -b F64 -chname,tasmin,tmin -sellonlatbox,140,150.1,-33,-40 \
                ${pathway}${GCM}/${scen}/${real}/${BC}/*/*/tasmin/* \
                tmin/${GCM}_${scen}_${real}_${BC}_tmin.nc
    cdo -L -b F64 -chname,tmax,temp -divc,2 -add \
                tmax/${GCM}_${scen}_${real}_${BC}_tmax.nc \
                tmin/${GCM}_${scen}_${real}_${BC}_tmin.nc \
                temp/${GCM}_${scen}_${real}_${BC}_temp.nc

    ncatted -O -a standard_name,temp,c,c,air_temperature temp/${GCM}_${scen}_${real}_${BC}_temp.nc
    ncatted -O -a standard_name,tmax,c,c,air_temperature tmax/${GCM}_${scen}_${real}_${BC}_tmax.nc
    ncatted -O -a standard_name,tmin,c,c,air_temperature tmin/${GCM}_${scen}_${real}_${BC}_tmin.nc

    ### Get precipitation
    cdo -L -b F64 -setrtoc,-1000,0,0 -chname,pr,prec -sellonlatbox,140,150.1,-33,-40 \
                ${pathway}${GCM}/${scen}/${real}/${BC}/*/*/pr/* \
                prec/${GCM}_${scen}_${real}_${BC}_prec.nc

    ### Get incoming SW radiation
    cdo -L -b F64 -chname,rsds,insol -sellonlatbox,140,150.1,-33,-40 \
                ${pathway}${GCM}/${scen}/${real}/${BC}/*/*/rsds/* \
                insol/${GCM}_${scen}_${real}_${BC}_insol.nc
    ncatted -O -a standard_name,insol,c,c,surface_downwelling_shortwave_flux \
                                    insol/${GCM}_${scen}_${real}_${BC}_insol.nc

    ### Get wind
    cdo -L -b F64 -chname,sfcWind,wind -sellonlatbox,140,150.1,-33,-40 \
                ${pathway}${GCM}/${scen}/${real}/${BC}/*/*/sfcWind/* \
                wind/${GCM}_${scen}_${real}_${BC}_wind.nc
    ncatted -O -a standard_name,wind,c,c,wind_speed wind/${GCM}_${scen}_${real}_${BC}_wind.nc
done
