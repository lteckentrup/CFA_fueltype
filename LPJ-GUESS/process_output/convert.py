from write_netcdf import (
    convert_ascii_netcdf_monthly,
    convert_ascii_netcdf_annual
    )

'''
convert_ascii_netcdf_monthly(var)
convert_ascii_netcdf_annual(var)
'''

config='Vic_default'
res=0.1

convert_ascii_netcdf_annual('agpp',config,res)
convert_ascii_netcdf_annual('cmass_leaf',config,res)
convert_ascii_netcdf_annual('cmass_sap',config,res)
convert_ascii_netcdf_annual('cmass_heart',config,res)
convert_ascii_netcdf_annual('cmass_debt',config,res)
convert_ascii_netcdf_annual('cmass',config,res)
convert_ascii_netcdf_annual('clitter',config,res)
convert_ascii_netcdf_annual('dens',config,res)
convert_ascii_netcdf_annual('fpc',config,res)
convert_ascii_netcdf_annual('cflux',config,res)
convert_ascii_netcdf_annual('annual_burned_area',config,res)
convert_ascii_netcdf_annual('cflux',config,res)
convert_ascii_netcdf_annual('agpp',config,res)
convert_ascii_netcdf_annual('lai',config,res) 
convert_ascii_netcdf_monthly('mlai',config,res)
convert_ascii_netcdf_monthly('mwcont_upper',config,res)
convert_ascii_netcdf_monthly('mwcont_lower',config,res)
convert_ascii_netcdf_monthly('monthly_burned_area',config,res)
