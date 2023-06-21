library(raster)
library(data.table)

### Read in coordinates
static.df <- fread('ft.static.csv')

### Define GCM, scenarion, timeperiod
GCM='ACCESS1-0'
scen='rcp45'
time='mid'
years='20452060'


### Read in clim forcing
tmax.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_tmax.rds',sep=''))
prcp.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_pr.rds',sep=''))
rh.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_rh.rds',sep=''))
lai.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_lai_jan_5km.rds',sep=''))
seas.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_pr_seasonality.rds',sep=''))

### Stack variables
met.stack = stack(list(tmax = tmax.ra,
                       prcp = prcp.ra, 
                       pr.seaonality = seas.ra, 
                       rh = rh.ra,
                       lai = lai.ra))

### Convert to dataframe
met.df <- as.data.frame(extract(met.stack,
                        cbind(static.df$lon,static.df$lat)))

### Combine static and climate features
static.df$tmax.mean <- met.df$tmax
static.df$map <- met.df$prcp
static.df$rh.mean <- met.df$rh
static.df$lai.opt.mean <- met.df$lai
static.df$pr.seaonality <- met.df$pr.seaonality

fwrite(static.df, paste('../cache/ft.',GCM,'.',scen,'_',time,'.csv',sep=''))
