library(raster)
library(data.table)

### Read in coordinates
static.df <- fread('ft.static.csv')

GCM='ACCESS1-0'
# GCM='BNU-ESM'
# GCM='CSIRO-Mk3-6-0'
# GCM='GFDL-CM3'
# GCM='GFDL-ESM2G'
# GCM='GFDL-ESM2M'
# GCM='INM-CM4'
# GCM='IPSL-CM5A-LR'
# GCM='MRI-CGCM3'

scen='history'
years='20002015'

### Read in clim forcing
tmax.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_tmax.rds',sep=''))
prcp.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_pr.rds',sep=''))
rh.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_rh.rds',sep=''))
lai.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_lai_jan_5km.rds',sep=''))
seas.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_pr_seasonality.rds',sep=''))

met.stack = stack(list(tmax = tmax.ra,
                       prcp = prcp.ra, 
                       pr.seaonality = seas.ra, 
                       rh = rh.ra,
                       lai = lai.ra))

met.df <- as.data.frame(extract(met.stack,
                        cbind(static.df$lon,static.df$lat)))

static.df$tmax.mean <- met.df$tmax
static.df$map <- met.df$prcp
static.df$rh.mean <- met.df$rh
static.df$lai.opt.mean <- met.df$lai
static.df$pr.seaonality <- met.df$pr.seaonality

fwrite(static.df, paste('../cache/ft.',GCM,'.history.csv',sep=''))
