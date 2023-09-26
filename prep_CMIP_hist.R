library(raster)
library(data.table)
library(argparse)

### Read in coordinates
static.df <- fread('ft.static.csv')

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
