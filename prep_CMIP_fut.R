library(raster)
library(data.table)
library(argparse)

### Set up arg parse
parser <- ArgumentParser(description = 'Generate future projection predictor file for GCM of interest')
parser$add_argument('--GCM', 
                    dest='GCM', 
                    help='Pass GCM name')
parser$add_argument('--scen', 
                    dest='scen', 
                    help='Pass RCP scenario')
parser$add_argument('--time', 
                    dest='time', 
                    help='Pass time span (mid or long)')
parser$add_argument('--years', 
                    dest='years', 
                    help='Pass years (20452060 for mid;  20852100 for long)')

### Parse arguments
args <- parser$parse_args()
GCM <- args$GCM
scen <- args$scen
time <- args$time
years <- args$years

### Read in coordinates
df.pred <- fread('pred.static.csv')

### Read in clim forcing
tmax.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_tmax.rds',sep=''))
prcp.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_pr.rds',sep=''))
rh.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_annual_rh.rds',sep=''))
lai.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_lai_jan_5km.rds',sep=''))
seas.ra <- readRDS(paste('../data/met/future/',GCM,'/',scen,'_',time,'/',scen,'_',years,'_pr_seasonality.rds',sep=''))

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

### Write CSV file
fwrite(df.pred, paste('../cache/pred.',GCM,'.',scen,'_',time,'.csv',sep=''))
