library(raster)
library(data.table)

### Coordinates of 90m fuel type map; see get_index.py
df.pred <- fread('../cache/index.csv')

# Get soil density
soil.density.ra <- raster('../data/soil/BDW_000_005_EV_N_P_AU_TRN_N_20140801.tif')
df.pred$soil.density <- extract(x = soil.density.ra,
                                y = cbind(df.pred$lon,
                                          df.pred$lat))

# Get clay fraction
soil.clay.ra <- raster('../data/soil/CLY_000_005_EV_N_P_AU_TRN_N_20140801.tif')
df.pred$clay <- extract(x = soil.clay.ra,
                        y = cbind(df.pred$lon,
                                  df.pred$lat))

# Get awailable water content
soil.awc.ra <- raster('../data/soil/AWC_000_005_EV.tif')
df.pred$awc <- extract(x = soil.awc.ra,
                       y = cbind(df.pred$lon,
                                 df.pred$lat))

# Get minerals
# Blue = Band 1 = Uranium
mineral.uranium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                             band=1)
df.pred$uranium <- extract(x = mineral.uranium.ra,
                           y = cbind(df.pred$lon,
                                     df.pred$lat))

# Green = Band 2 = Thorium
mineral.thorium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                             band=2)
df.pred$thorium <- extract(x = mineral.thorium.ra,
                           y = cbind(df.pred$lon,
                                     df.pred$lat))

# Red = Band 3 = Potassium
mineral.potassium.ra <- raster('../data/minerals/radmap_v4_2019_filtered_ternary_image.tif', 
                               band=3)
df.pred$potassium <- extract(x = mineral.potassium.ra,
                             y = cbind(df.pred$lon,
                                       df.pred$lat))

# Use ratios
df.pred$uran_pot <- df.pred$uranium/ df.pred$potassium
df.pred$thorium_pot <- df.pred$thorium/ df.pred$potassium

### Get topography
# Shortwave radiation january
rad.jan.ra <- raster('../data/topo/rad/SRADTotalShortwaveSlopingSurf_0115_lzw.tif')
df.pred$rad.short.jan <- extract(x = rad.jan.ra,
                                 y = cbind(df.pred$lon,
                                           df.pred$lat))
rm(rad.jan.ra)

rad.jul.ra <- raster('../data/topo/rad/SRADTotalShortwaveSlopingSurf_0715_lzw.tif')
df.pred$rad.short.jul <- extract(x = rad.jul.ra,
                                 y = cbind(df.pred$lon,
                                           df.pred$lat))
rm(rad.jul.ra)

# get wi#####
wi.ra <- raster('../data/topo/wetness_index/90m/twi_3s.tif')
df.pred$wi <- extract(x = wi.ra, 
                      y = cbind(df.pred$lon,
                                df.pred$lat))
rm(wi.ra)

# get profile_c
fn <- '../data/topo/curvature_profile/90m/profile_curvature_3s.tif'
c_profile.ra <- raster(fn)
df.pred$curvature_profile <- extract(x = c_profile.ra,
                                     y = cbind(df.pred$lon,
                                               df.pred$lat))

fn.c.plan <- '../data/topo/curvature_plan/90m/plan_curvature_3s.tif'
c_plan.ra <- raster(fn.c.plan)
df.pred$curvature_plan <- extract(x = c_plan.ra,
                                  y = cbind(df.pred$lon,
                                            df.pred$lat))

soil.depth.ra <- raster('../data/soil/DER_000_999_EV_N_P_AU_NAT_C_20150601.tif')
df.pred$soil.depth <- extract(x = soil.depth.ra,
                              y =  cbind(df.pred$lon,
                                         df.pred$lat))

### Get elevation/ relief data
elevation.ra <- raster('../data/relief/Relief_dems_3s_mosaic1.tif')
df.pred$elevation <- extract(x = elevation.ra,
                             y = cbind(df.pred$lon,
                                       df.pred$lat))

### Get mask for public vs private land
tenure.ra <- raster('../data/tenure/tenure.tif')
df.pred$tenure <- extract(x = tenure.ra,
                          y = cbind(df.pred$lon,
                                    df.pred$lat))

fwrite(df.pred, 'ft.static.csv')
