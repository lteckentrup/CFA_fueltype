for exp in LPJ_default LPJ_CFA; do
	for var in mlai lai agpp anpp cflux cmass cmass_root fpc; do
		cdo -L -b F64 -setgrid,grid.txt ${exp}/${var}_1950-2022.nc \
						   ${exp}/${var}_1950-2022_mask.nc
	done
done

for exp in Vic_default Vic_final; do
	cdo -L -b F64 -timmean -yearmean -selyear,2003/2021 \
				  ${exp}/mlai_1950-2022_mask.nc \
				  ${exp}/mlai_timmean.nc
	cdo -L -b F64 -fldmean -yearmean ${exp}/mlai_1950-2022_mask.nc \
									 ${exp}/mlai_fldmean.nc

	for var in lai agpp anpp cflux fpc; do
		cdo -L -b F64 -timmean -selyear,2003/2021 \
					  ${exp}/${var}_1950-2022_mask.nc \
					  ${exp}/${var}_timmean.nc
		cdo -L -b F64 -divc,1e+12 -fldsum -mul ${exp}/${var}_1950-2022_mask.nc \
											   -gridarea \
                                               ${exp}/${var}_1950-2022_mask.nc \
											   ${exp}/${var}_fldsum.nc
    done
    for var in cmass cmass_root; do
		cdo -L -b F64 -timmean -selyear,1993/2012 \
					  ${exp}/${var}_1950-2022_mask.nc \
					  ${exp}/${var}_timmean.nc
		cdo -L -b F64 -selyear,1993/2012 -divc,1e+12 -fldsum -mul \
											  ${exp}/${var}_1950-2022_mask.nc \
											   -gridarea \
                                               ${exp}/${var}_1950-2022_mask.nc \
											   ${exp}/${var}_fldsum.nc
	done
done
