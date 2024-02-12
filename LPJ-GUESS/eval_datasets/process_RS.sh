cdo -L -b F64 -div -remapnn,01_mask.nc \
    -sellonlatbox,140,150,-40,-33 \
    /scratch/pt17/lt0205/AusEFlux/AusEFlux_GPP_2003_2022_5km_quantiles_v1.1.nc \
    01_mask.nc AusEFlux_GPP.nc

cdo -L -b F64 -div -remapnn,01_mask.nc \
    -sellonlatbox,140,150,-40,-33 \
    /scratch/pt17/lt0205/AusEFlux/AusEFlux_ER_2003_2022_5km_quantiles_v1.1.nc \
    01_mask.nc AusEFlux_ER.nc

cdo -L -b F64 -div -remapnn,01_mask.nc \
    -divc,10 -sellonlatbox,140,150,-40,-33 \
    /g/data/w97/lt0205/research/La_Nina/MODIS/GPP/combined_annual.nc \
    01_mask.nc MODIS_GPP.nc

cdo -L -b F64 -div -remapnn,01_mask.nc \
    -sellonlatbox,140,150,-40,-33 \
    /g/data/w97/lt0205/research/VOD/Global_annual_mean_ABC_lc2001_1993_2012_20150331_reorder.nc \
    01_mask.nc VOD_ABC.nc

cdo -L -b F64 -chname,Band46,mlai -div -remapnn,01_mask.nc \
    -sellonlatbox,140,150,-40,-33 \
    /g/data/w97/lt0205/HIE/LAI/MODIS/lai_2002-2022_annual.nc \
    01_mask.nc MODIS_LAI.nc

cdo -L -b F64 -div -remapnn,01_mask.nc \
    -sellonlatbox,140,150,-40,-33 \
    /g/data/w97/lt0205/research/Fire_CCI5.1/2001-2020-ESACCI-L4_FIRE-BA-MODIS-fv5.1.nc \
    01_mask.nc Fire_CCI_BA.nc

cdo -L -chunit,m2,% -mulc,100 -div Fire_CCI_BA.nc -gridarea Fire_CCI_BA.nc Fire_CCI_BF.nc

cdo -L -b F64 -div -remapnn,01_mask.nc -yearsum \
    -sellonlatbox,140,150,-40,-33 \
    /g/data/w97/lt0205/research/CAMS_GFAS/CMAS_GFAS_CO2_monthly_unit.nc \
    01_mask.nc CAMS-GFAS_fire_CO2.nc

### Field sums and averages
cdo -L -b F64 -chname,GPP_median,agpp -divc,1e+12 -fldsum -mul \
    -yearsum -divc,1000 -selvar,GPP_median \
    AusEFlux_GPP.nc -gridarea AusEFlux_GPP.nc AusEFlux_GPP_fldsum.nc

cdo -L -b F64 -chname,ER_median,er -divc,1e+12 -fldsum -mul \
    -yearsum -divc,1000 -selvar,ER_median \
    AusEFlux_ER.nc -gridarea AusEFlux_ER.nc AusEFlux_ER_fldsum.nc

cdo -L -b F64 -chname,__xarray_dataarray_variable__,agpp -divc,1e+12 \
    -fldsum -mul MODIS_GPP.nc -gridarea \
    MODIS_GPP.nc MODIS_GPP_fldsum.nc

cdo -L -b F64 -fldmean MODIS_LAI.nc MODIS_LAI_fldmean.nc

cdo -L -b F64 -divc,1e+12 -fldsum -mul \
    CAMS-GFAS_fire_CO2.nc -gridarea CAMS-GFAS_fire_CO2.nc \
    CAMS-GFAS_fire_CO2_fldsum.nc

cdo -L -b F64 -divc,1e+12 -divc,10 -fldsum -mul \
    VOD_ABC.nc -gridarea VOD_ABC.nc VOD_ABC_fldsum.nc

### Time average
cdo -L -b F64 -chname,GPP_median,agpp -timmean -yearsum -divc,1000 \
    -selyear,2003/2021 -selvar,GPP_median AusEFlux_GPP.nc \
    AusEFlux_GPP_timmean.nc

cdo -L -b F64 -chname,ER_median,er -timmean -yearsum -divc,1000 \
    -selyear,2003/2021 -selvar,ER_median AusEFlux_ER.nc \
    AusEFlux_ER_timmean.nc

cdo -L -b F64 -chname,__xarray_dataarray_variable__,agpp \
    -timmean -selyear,2003/2021 MODIS_GPP.nc MODIS_GPP_timmean.nc

cdo -L -b F64 -timmean -selyear,2003/2021 MODIS_LAI.nc MODIS_LAI_timmean.nc
