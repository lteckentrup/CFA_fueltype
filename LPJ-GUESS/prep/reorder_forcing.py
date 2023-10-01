import xarray as xr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--GCM', type=str, required=True)
parser.add_argument('--scen', type=str, required=True)
parser.add_argument('--var', type=str, required=True)

args = parser.parse_args()

def reorder(GCM,scen,var):
    ds = xr.open_dataset(var+'/'+GCM+'_'+scen+'_r1i1p1_r240x120-MRNBC-AWAP_'+
                         var+'_1960-2100.nc')

    ds = ds.drop_dims('bnds').transpose('lat','lon','time')
    print('reordered')
    ds_chunked = ds.chunk({'time': 1})
    ds_chunked.to_netcdf(var+'/'+GCM+'_'+scen+'_r1i1p1_'+var+'_reorder.nc')
    print('wrote file')

reorder(args.GCM,args.scen,args.var)
