import pandas as pd
import xarray as xr
import os
import numpy as np
from datetime import date
from decimal import Decimal

### Read in PFT names: shortnames in ASCII output files, 
### longnames are more descriptive
from vars_global import (
    PFT_shortnames,
    PFT_longnames,
    cflux_shortnames,
    cflux_longnames,
    cpool_shortnames,
    cpool_longnames,
    nflux_shortnames,
    nflux_longnames,
    ngases_shortnames,
    ngases_longnames,
    npool_shortnames,
    npool_longnames,
    nsources_shortnames,
    nsources_longnames,
    tot_runoff_shortnames,
    tot_runoff_longnames
    )

### Which date
date_created = date.today()

### Convert annual *out files to netcdf
def convert_ascii_netcdf_annual(var,config,res):
    ### Read in out file
    df = pd.read_csv('/g/data/w97/lt0205/research/lpj_guess_4.1/runs/runs_HIE/'+
                     config+'/'+var+'.out',header=0,delim_whitespace=True)
    
    ### Get years, first and last year is used for file name
    years = np.unique(df.Year)
    first_year = str(int(years[0]))
    last_year = str(int(years[-1]))

    ### Set up file name for netCDF
    fileOUT = (config+'/'+var+'_'+first_year+'-'+last_year+'.nc')

    ### Print last year
    print(last_year)

    ### Rename columns to more traditional dimension names
    df.rename(columns={'Year': 'time', 
                       'Lat': 'lat',
                       'Lon': 'lon'}, inplace=True)
    
    ### Convert years to time information
    df.time = pd.to_datetime(df.time, format = '%Y')

    ### Convert dataframe to xarray DataSet
    ds = df.set_index(['time', 
                       'lat', 
                       'lon']).to_xarray()

    ### LPJ runs with land pixels only so need to fill in NaN values for non-land
    ### (facilitates comparison with other non-LPJ datasets)

    ### Choose decimal package to avoid drama with precision

    ### Get longitudes for Vic
    start = Decimal('140')
    stop = Decimal('150.1')
    resolution = Decimal(res)
    target_lon = np.arange(start, stop + resolution, resolution, dtype=object)
    target_lon = np.array(target_lon, dtype=float)

    ### Get latitudes for Vic
    start = Decimal('33')
    stop = Decimal('40')
    resolution = Decimal(res)
    target_lat = np.arange(start, stop + resolution, resolution, dtype=object) 
    target_lat = np.array(target_lat, dtype=float)* (-1)

    ### Pull out time dimension from prelim DataSet
    target_time = ds['time']

    ### Create an empty target dataset with target dimensions
    ds_target = xr.Dataset(
        {'value': (['time', 'lat', 'lon'], 
                   np.nan*np.ones((len(target_time), 
                                   len(target_lat), 
                                   len(target_lon))))},
        coords={'time': target_time, 
                'lat': target_lat[::-1], 
                'lon': target_lon}
    )

    ### Use broadcast_like to fill up ds
    ds = ds.broadcast_like(ds_target)

    ### time encoding firlefanz
    ds.time.encoding['units'] = 'Seconds since 1950-01-01 00:00:00'
    ds.time.encoding['long_name'] = 'Time'
    ds.time.encoding['calendar'] = '365_day'
    
    ### Global attributes
    ds.attrs={'Conventions':'CF-1.6',
              'Model':'LPJ-GUESS version 4.0.1.',
              'Set-up': 'Stochastic and fire disturbance active',
              'Date_Created':str(date_created)}

    ### Set up list of dimension and data type in prep to save to netCDF later 
    dim = ['time','lat','lon']
    dim_dtype = ['double','double','double']

    ### Set variable units
    if var in ('aaet','agpp','anpp','clitter','cmass','cmass_wood','cmass_leaf',
               'cmass_root','cton_leaf','dens','fpc','height','lai','nlitter',
               'nmass','nuptake','vmaxnlim','wscal_mean'):
        if var == 'aaet':
            unit='mm/year'
        elif var in ('agpp','anpp'):
            unit='kgC/m2/year'
        elif var in ('clitter','cmass','cmass_wood','cmass_leaf','cmass_root'):
            unit='kgC/m2'
        elif var == 'cton_leaf':
            unit='ckgC/kgN'
        elif var == 'dens':
            unit='indiv/m2'
        elif var in ('fpc','lai'):
            unit='m2/m2'
        elif var in ('nlitter','nmass'):
            unit='kgN/m2'
        elif var == 'nuptake':
            unit='kgN/m2/year'
        elif var == 'vmaxlim':
             unit='-'
        elif var == 'height':
            unit='m'
        elif var == 'wscal_mean':
            unit='-'

        ### for each PFT, set unit, long_name, dimensions and data type
        for PFT_short, PFT_long in zip(PFT_shortnames, PFT_longnames):
            ds[PFT_short].attrs={'units':unit,
                                 'long_name':PFT_long}
            dim.append(PFT_short)
            dim_dtype.append('float32')

        ### Variables that also have all PFTs aggregated to 'Total'
        if var in ('aaet','agpp','anpp','clitter','cmass','cmass_wood',
                   'cmass_leaf','cmass_root','cton_leaf','dens','fpc','lai',
                   'nlitter','nmass','nuptake','vmaxnlim'):
            ds['Total'].attrs={'units':unit,
                               'long_name':'Total'}
            dim.append('Total')
            dim_dtype.append('float32')
        else:
            pass

    elif var == 'cflux':
        for cflux_short, cflux_long in zip(cflux_shortnames, cflux_longnames):
            ds[cflux_short].attrs={'units':'kgC/m2/year',
                                   'long_name':cflux_long}

            dim.append(cflux_short)
            dim_dtype.append('float32')

    elif var == 'cpool':
        for cpool_short, cpool_long in zip(cpool_shortnames, cpool_longnames):
            ds[cpool_short].attrs={'units':'kgC/m2',
                                   'long_name':'cpool_long'}

            dim.append(cpool_short)
            dim_dtype.append('float32')

    elif var == 'firert':
        ds['FireRT'].attrs={'units':'yr',
                            'long_name':'Fire return time'}

        dim.append('FireRT')
        dim_dtype.append('float32')

    elif var == 'doc':
        ds['Total'].attrs={'units':'kgC/m2r',
                           'long_name':'Total dissolved organic carbon'}

        dim.append('Total')
        dim_dtype.append('float32')

    elif var == 'nflux':
        for nflux_short, nflux_long in zip(nflux_shortnames, nflux_longnames):
            ds[nflux_short].attrs={'units':'kgN/ha/year',
                                   'long_name':nflux_long}

            dim.append(nflux_short)
            dim_dtype.append('float32')

    elif var == 'ngases':
        for ngases_short, ngases_long in zip(ngases_shortnames, ngases_longnames):
            ds[ngases_short].attrs={'units':'kgN/ha/year',
                                    'long_name':ngases_long}

            dim.append(ngases_short)
            dim_dtype.append('float32')

    elif var == 'npool':
        for npool_short, npool_long in zip(npool_shortnames, npool_longnames):
            ds[npool_short].attrs={'units':'kgN/m2',
                                   'long_name':npool_long}
            dim.append(npool_short)
            dim_dtype.append('float32')

    elif var == 'nsources':
        for nsources_short, nsources_long in zip(nsources_shortnames,
                                                 nsources_longnames):
            ds[nsources_short].attrs={'units':'gN/ha',
                                      'long_name':nsources_long}

            dim.append(nsources_short)
            dim_dtype.append('float32')

    elif var == 'tot_runoff':
        for tot_runoff_short, tot_runoff_long in zip(tot_runoff_shortnames,
                                                     tot_runoff_longnames):
            ds[tot_runoff_short].attrs={'units':'mm/year',
                                        'long_name':tot_runoff_long}

            dim.append(tot_runoff_short)
            dim_dtype.append('float32')
    else:
        pass

    dtype_fill = ['dtype']*len(dim)
    encoding_dict = {a: {b: c} for a, b, c in zip(dim, dtype_fill, dim_dtype)}

    # save to netCDF
    ds.to_netcdf(fileOUT, encoding=encoding_dict)

def convert_ascii_netcdf_monthly(var,config,res):
    ### Read in out file
    df = pd.read_csv('/g/data/w97/lt0205/research/lpj_guess_4.1/runs/runs_HIE/'+
                     config+'/'+var+'.out',header=0,delim_whitespace=True)

    ### Rename columns to more traditional dimension names
    df.rename(columns={'Year': 'year', 
                       'Lat': 'lat',
                       'Lon': 'lon'}, inplace=True)
    
    ### Get years, first and last year is used for file name
    years = np.unique(df.year)

    first_year=str(int(years[0]))
    last_year=str(int(years[-1]))

    ### Set up file name for netCDF
    fileOUT = (config+'/'+var+'_'+first_year+'-'+last_year+'.nc')

    # Create a pandas datetime object by combining 'time' and 'month' columns    
    df = pd.melt(df, id_vars=['year', 'lat', 'lon'],
                 var_name='month', 
                 value_name=var)

    ### Link month names with numbers - helps creating time axis
    month2num = {'Jan': '01', 'Feb': '02', 
                 'Mar': '03', 'Apr': '04', 
                 'May': '05', 'Jun': '06',
                 'Jul': '07', 'Aug': '08', 
                 'Sep': '09', 'Oct': '10', 
                 'Nov': '11', 'Dec': '12'
                 }
    
    ### Include time info in dataframe
    df['time'] = pd.to_datetime(df['year'].astype(str) + df['month'].map(month2num), 
                                format='%Y%m')

    ### Convert dataframe to xarray DataSet
    ds = df[[var,'time','lat','lon']].set_index(['time','lat','lon']).to_xarray()

    ### LPJ runs with land pixels only so need to fill in NaN values for non-land
    ### (facilitates comparison with other non-LPJ datasets)

    ### Choose decimal package to avoid drama with precision

    ### Get longitudes for Vic
    start = Decimal('140')
    stop = Decimal('150.1')
    resolution = Decimal(res)
    target_lon = np.arange(start, stop + resolution, resolution, dtype=object)
    target_lon = np.array(target_lon, dtype=float)

    ### Get latitudes for Vic
    start = Decimal('33')
    stop = Decimal('40')
    resolution = Decimal(res)
    target_lat = np.arange(start, stop + resolution, resolution, dtype=object) 
    target_lat = np.array(target_lat, dtype=float)* (-1)

    ### Pull out time dimension from prelim DataSet
    target_time = ds['time']

    ### Create an empty target dataset with target dimensions
    ds_target = xr.Dataset(
        {'value': (['time', 'lat', 'lon'], 
                   np.nan*np.ones((len(target_time), 
                                   len(target_lat), 
                                   len(target_lon))))},
        coords={'time': target_time, 
                'lat': target_lat[::-1], 
                'lon': target_lon}
    )

    ### Use broadcast_like to fill up ds
    ds = ds.broadcast_like(ds_target)

    ### time encoding firlefanz
    ds.time.encoding['units'] = 'Seconds since 1901-01-01 00:00:00'
    ds.time.encoding['long_name'] = 'Time'
    ds.time.encoding['calendar'] = '365_day'

    ### Global attributes
    ds.attrs={'Conventions':'CF-1.6',
              'Model':'LPJ-GUESS version 4.0.1.',
              'Set-up': 'Stochastic and fire disturbance active',
              'Date_Created':str(date_created)}

    ### Monthly Total
    if var == 'maet':
        ds[var].attrs={'units':'mm/month',
                       'long_name':'Monthly actual Evapotranspiration'}
    elif var == 'mevap':
        ds[var].attrs={'units':'mm/month',
                       'long_name':'Monthly Evapotranspiration'}
    elif var == 'mgpp':
        ds[var].attrs={'units':'kgC/m2/month',
                       'long_name':'Monthly GPP'}
    elif var == 'mintercep':
        ds['mintercep'].attrs={'units':'mm/month',
                               'long_name':'Monthly interception Evaporation'}
    elif var == 'miso':
        ds[var].attrs={'units':'kg/month',
                       'long_name':'Monthly isopene emissions'}
    elif var == 'mmon':
        ds[var].attrs={'units':'kg/month',
                       'long_name':'Monthly monterpene emissions'}
    elif var == 'mnee':
        ds[var].attrs={'units':'kgC/m2/month', 'long_name':'Monthly NEE'}
    elif var == 'mpet':
        ds[var].attrs={'units':'mm/month',
                       'long_name':'Monthly potential evapotranspiration'}
    elif var == 'mra':
        ds[var].attrs={'units':'kgC/m2/month',
                       'long_name':'Monthly autotrophic respiration'}
    elif var == 'mrh':
        ds[var].attrs={'units':'kgC/m2/month',
                       'long_name':'Monthly heterotrophic respiration'}
    elif var == 'mlai':
        ds[var].attrs={'units':'m2/m2',
                       'long_name':'Monthly LAI'}
    elif var == 'mrunoff':
        ds[var].attrs={'units':'mm/month',
                       'long_name':'Monthly runoff'}
    elif var == 'mwcont_lower':
        ds[var].attrs={'units':'fraction of available water-holding capacity',
                       'long_name':'Monthly water in content in lower soil layer'
                       '(50 - 150 cm)'}
    elif var == 'mwcont_upper':
        ds[var].attrs={'units':'fraction of available water-holding capacity',
                       'long_name':'Monthly water in content in upper soil layer'
                       '(0 - 50 cm)'}


    ds.to_netcdf(fileOUT, encoding={'time':{'dtype': 'double'},
                                    'lat':{'dtype': 'double'},
                                    'lon':{'dtype': 'double'},
                                     var:{'dtype': 'float32'}})
