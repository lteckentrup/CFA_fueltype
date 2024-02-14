# CFA_fueltype

# LPJ-GUESS
```LPJ-GUESS``` has all the stuff for the LPJ-GUESS simulations. It has the subdirectories

```process_input``` - this directory contains scripts to process the input data. ```grid_vic.py``` is a script to write a grid file specific for the CFA domain, ```get_clim_data.sh``` is a bash script to grab future climate projections (uses CDO and NCO so it can only run on gadi), ```calc_RH.sh``` calculates relative humidity following the protocol in AWRA (see also CFA report). ```reorder.py``` reorders the dimensions in the climate forcing for more efficient run times in LPJ-GUESS but Drew said the chunk sizes weren't yet ideal so there is probably room for improvement. Lastly, ```merge_reorder_files.sh``` merges the files across the time axis for each scenario, GCM, and variable, and also runs the ```reorder.py``` script once the files are merged.

```process_output``` contains two scripts to convert the raw ASCII LPJ-GUESS output to netCDF. The function to do this itself is stored in ```ascii2netCDF.py``` and then called in ```convert.py``` where simulation and variable can be specified. ```process_for_eval_RS.sh``` is a bash script I used to process the LPJ-GUESS netCDF to compare with remote sensing datasets. This one uses CDO so it won't run on HIE storage.

```eval_datasets``` contains the bash script ```process_RS.sh``` that processes the remote sensing data for comparison with LPJ-GUESS. As with ```process_for_eval_RS.sh```, it uses CDO so it won't run on HIE storage.

```analysis``` contains all the analysis scripts that went into the report. 

All timeseries plots are done with ```timeseries*py```. ```timeseries_cpool.py``` plots the absolute timeseries of carbon pools, and ```timeseries_per_PFT.py``` plots the absolute timeseries of a variable that can be specified in the argument (see also ```timeseries_per_PFT.sh``` to run script) per PFT. Similar to this, all ```timeseries_sens_<suffix>.py``` plot the sensitivity of the variables to the experiment where fire is switched off (```nofire```) or where every 10 years 10% of each gridcell are burned ('frequent high severity fires', ```FHSF```). These timeseries scripts take an argument for the sensitivity experiment of interest (see also bash scripts with identical name).

All maps are plotted using ```map*py```, all barplots are plotted using ```barplot_*py```. The structure of the script names is consistent across the different types of plots (so identical to what I described for `timeseries*py```). Again, when python scripts take arguments, there is always an accompanying bash script that loops through all arguments.

# ML
```LPJ-GUESS``` has all the stuff for the machine learning bit. It has the subdirectories

```process_input``` has all the scripts that I used for the data prep for the machine learning. ```get_index.py``` elects the coordinates and fuel types from the current fuel type distribution (provided by CFA). ```prep_static.R``` generates a csv file with a dataframe of all static predictors, ```prep_CMIP_hist.R``` generates a csv file where historical CMIP projections are added to the static predictors and saved for each GCM individually. ```prep_CMIP_fut.R``` generates a csv file where future CMIP projections are added to the static predictors and saved for each GCM individually, and also each RCP scenario and timeslice are saved in separate files. Both ```prep_CMIP*R``` take the arguments GCM, scen, and time identifier (mid, long) and the associated years (20452060, 20852100). ```get_hist_predictors.sh``` and ```get_fut_predictors.sh``` run all ```prep_CMIP_hist.R``` and ```prep_CMIP_fut.R```, respectively. Jim did some of the initial processing and you'd need to look that up on what he left.

```process_output``` contains scripts to calculate ensemble statitics across the ensemble (```ensstat.py```), ```csv2netCDF.py``` converts the machine learning prediction csv to netCDF. They both use arguments, they are in the comment on top of the python script.

```models``` has the scripts for the actual machine learning models. ```ML_approaches.py``` tests three different approaches (see report for more info), ```RF_train.py``` trains the random forest and saves the resulting model as a pickle file, and ```RF_predict.py``` reads in the trained model and applies it to the future projections. Both ```RF_train.py``` and ```RF_predict.py``` take arguments which are defined in the comment on top of the respective python script.

```analysis``` contains all analysis scripts. They file naming should be somewhat self-explanatory. There are also two scripts that prepare and save an interim dataset into a csv directory: ```fuel_type_shifts.py``` and ```unseen_clim.py```. These calculate how fuel types might shift in the future, and which gridcells might experience novel (unseen) climates. In the analysis directory, most python scripts have a twin bash script which can be used to run the respective analysis script for all timeslices and RCP scenarios (and/or individual GCMs).

