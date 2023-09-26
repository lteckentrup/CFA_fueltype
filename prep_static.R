library(raster)
library(data.table)

### Coordinates of 90m fuel type map; see get_index.py
sample.df <- fread('../cache/index.csv')

# Get soil density
soil.density.ra <- raster('../data/soil/BDW_000_005_EV_N_P_AU_TRN_N_20140801.tif')
sample.df$soil.density <- extract(x = soil.density.ra,
                                  y = cbind(sample.df$lon,
                                            sample.df$lat))

# Get clay fraction
soil.clay.ra <- raster('../data/soil/CLY_000_005_EV_N_P_AU_TRN_N_20140801.tif')
sample.df$clay <- extract(x = soil.clay.ra,
                          y = cbind(sample.df$lon,
                                    sample.df$lat))

# Get awailable water content
soil.awc.ra <- raster('../data/soil/AWC_000_005_EV.tif')
sample.df$awc <- extract(x = soil.awc.ra,
                         y = cbind(sample.df$lon,
                                   sample.df$lat))

# Get minerals
# Blue = Band 1 = Uranium
mineral.uranium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                             band=1)
sample.df$uranium <- extract(x = mineral.uranium.ra,
                             y = cbind(sample.df$lon,
                                       sample.df$lat))

# Green = Band 2 = Thorium
mineral.thorium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                             band=2)
sample.df$thorium <- extract(x = mineral.thorium.ra,
                             y = cbind(sample.df$lon,
                                       sample.df$lat))

# Red = Band 3 = Potassium
mineral.potassium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                               band=3)
sample.df$potassium <- extract(x = mineral.potassium.ra,
                               y = cbind(sample.df$lon,
                                         sample.df$lat))

# Use ratios
sample.df$uran_pot <- sample.df$uranium/ sample.df$potassium
sample.df$thorium_pot <- sample.df$thorium/ sample.df$potassium

### Get topography
# Shortwave radiation january
rad.jan.ra <- raster('../data/topo/rad/SRADTotalShortwaveSlopingSurf_0115_lzw.tif')
sample.df$rad.short.jan <- extract(x = rad.jan.ra,
                                   y = cbind(sample.df$lon,
                                             sample.df$lat))
rm(rad.jan.ra)

rad.jul.ra <- raster('../data/topo/rad/SRADTotalShortwaveSlopingSurf_0715_lzw.tif')
sample.df$rad.short.jul <- extract(x = rad.jul.ra,
                                   y = cbind(sample.df$lon,
                                             sample.df$lat))
rm(rad.jul.ra)

# get wi#####
wi.ra <- raster('../data/topo/wetness_index/90m/twi_3s.tif')
sample.df$wi <- extract(x = wi.ra, 
                        y = cbind(sample.df$lon,
                                  sample.df$lat))
rm(wi.ra)

# get profile_c
fn <- '../data/topo/curvature_profile/90m/profile_curvature_3s.tif'
c_profile.ra <- raster(fn)
sample.df$curvature_profile <- extract(x = c_profile.ra,
                                       y = cbind(sample.df$lon,
                                                 sample.df$lat))

fn.c.plan <- '../data/topo/curvature_plan/90m/plan_curvature_3s.tif'
c_plan.ra <- raster(fn.c.plan)
sample.df$curvature_plan <- extract(x = c_plan.ra,
                                    y = cbind(sample.df$lon,
                                              sample.df$lat))

soil.depth.ra <- raster('../data/soil/DER_000_999_EV_N_P_AU_NAT_C_20150601.tif')
sample.df$soil.depth <- extract(x = soil.depth.ra,
                                y =  cbind(sample.df$lon,
                                           sample.df$lat))

### Get elevation/ relief data
elevation.ra <- raster('../data/relief/Relief_dems_3s_mosaic1.tif')
sample.df$elevation <- extract(x = elevation.ra,
                               y = cbind(sample.df$lon,
                                         sample.df$lat))

### Get mask for public vs private land
tenure.ra <- raster('../data/tenure/tenure.tif')
sample.df$tenure <- extract(x = tenure.ra,
                            y = cbind(sample.df$lon,
                                      sample.df$lat))

fwrite(sample.df, 'ft.static.csv')
