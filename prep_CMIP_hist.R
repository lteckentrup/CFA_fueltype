library(raster)
library(data.table)
library(argparse)

### Read in coordinates
df.pred <- fread('pred.static.csv')

parser <- ArgumentParser(description = 'Generate historical climate forcing for GCM of interest')
parser$add_argument('--GCM', 
                    dest='GCM', 
                    help='Pass GCM name')

args <- parser$parse_args()
GCM <- args$GCM

scen='history'
years='20002015'

### Read in clim forcing
tmax.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_tmax.rds',sep=''))
prcp.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_pr.rds',sep=''))
rh.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_annual_rh.rds',sep=''))
lai.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_lai_jan_5km.rds',sep=''))
seas.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'/',scen,'_',years,'_pr_seasonality.rds',sep=''))

### Stack met files
met.stack = stack(list(tmax = tmax.ra,
                       prcp = prcp.ra, 
                       pr.seaonality = seas.ra, 
                       rh = rh.ra,
                       lai = lai.ra))

### Convert to dataframe
df.met <- as.data.frame(extract(met.stack,
                        cbind(df.pred$lon,df.pred$lat)))

### Combine met and static predictors
df.pred$tmax.mean <- df.met$tmax
df.pred$map <- df.met$prcp
df.pred$rh.mean <- df.met$rh
df.pred$lai.opt.mean <- df.met$lai
df.pred$pr.seaonality <- df.met$pr.seaonality

fwrite(df.pred, paste('../cache/pred.',GCM,'.history.csv',sep=''))
