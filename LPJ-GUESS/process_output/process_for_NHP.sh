### Messed up saving the wood carbon so calculated it from other outputs
for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
        for exp in RCP45 RCP45_FHSF RCP45_nofire RCP85 RCP85_FHSF RCP85_nofire; do
		cdo -L -b F64 -add ../runs_${GCM}_${exp}/cmass_sap_1960-2099.nc \
				   ../runs_${GCM}_${exp}/cmass_heart_1960-2099.nc \
				   ../runs_${GCM}_${exp}/cmass_prelim_1960-2099.nc 
		cdo -L -b F64 -sub ../runs_${GCM}_${exp}/cmass_prelim_1960-2099.nc\
				   ../runs_${GCM}_${exp}/cmass_debt_1960-2099.nc \
				   ../runs_${GCM}_${exp}/cmass_wood_1960-2099.nc 
		rm ../runs_${GCM}_${exp}/cmass_prelim_1960-2099.nc 
		done
	done


for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
        for exp in RCP45 RCP45_FHSF RCP45_nofire RCP85 RCP85_FHSF RCP85_nofire; do
                ### Calculates annual average and area weighted average over Vic
		for var in mlai mwcont_upper; do
			cdo -L -b F64 -fldmean -yearmean ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
							 ../runs_${GCM}_${exp}/${var}_1960-2099_fldmean.nc
		done
                ### Calculates area weighted average over Vic
		for var in lai fpc height dens; do
			cdo -L -b F64 -fldmean -yearmean ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
							 ../runs_${GCM}_${exp}/${var}_1960-2099_fldmean.nc
		done

                ### Calculates area weighted sum over Vic
		for var in agpp anpp cflux cmass cmass_root cmass_leaf clitter cmass_wood; do
			cdo -L -b F64 -divc,1e+12 -fldsum -mul ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
							       -gridarea \
							       ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
							       ../runs_${GCM}_${exp}/${var}_1960-2099_fldsum.nc
		done
	done
done

### Sorry this is me being absolutely lazy but I saved the masked area sums and averages to new 
### files
for GCM in CNRM-CERFACS-CNRM-CM5 CSIRO-BOM-ACCESS1-0 MIROC-MIROC5 NOAA-GFDL-GFDL-ESM2M; do
        for exp in RCP45 RCP45_FHSF RCP45_nofire RCP85 RCP85_FHSF RCP85_nofire; do
                for var in mlai mwcont_upper; do
                        cdo -L -b F64 -fldmean -yearmean -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                                              Mallee_mask.nc \
                                                              ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099_fldmean.nc
                        cdo -L -b F64 -fldmean -yearmean -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                                              Wet_forest_mask.nc \
                                                              ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099_fldmean.nc
                done
                for var in lai fpc height dens; do
                        cdo -L -b F64 -fldmean -yearmean -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                                              Mallee_mask.nc \
                                                              ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099_fldmean.nc
                        cdo -L -b F64 -fldmean -yearmean -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                                              Wet_forest_mask.nc \
                                                              ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099_fldmean.nc

                done

                for var in agpp anpp cflux cmass cmass_root cmass_leaf clitter cmass_wood; do
                        cdo -L -b F64 -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                           Mallee_mask.nc \
                                           ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099.nc
                        cdo -L -b F64 -div ../runs_${GCM}_${exp}/${var}_1960-2099.nc \
                                           Wet_forest_mask.nc \
                                           ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099.nc
                        cdo -L -b F64 -divc,1e+12 -fldsum -mul ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099.nc \
                                                               -gridarea \
                                                               ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099.nc \
                                                               ../runs_${GCM}_${exp}/Mallee/${var}_1960-2099_fldsum.nc
                        cdo -L -b F64 -divc,1e+12 -fldsum -mul ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099.nc \
                                                               -gridarea \
                                                               ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099.nc \
                                                               ../runs_${GCM}_${exp}/Wet_forest/${var}_1960-2099_fldsum.nc
                done
        done
done

