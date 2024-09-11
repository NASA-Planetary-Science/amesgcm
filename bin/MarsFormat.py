#!/usr/bin/env python3

#Load generic Python Modules
import argparse   # parse arguments
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os
from amescap.Script_utils import prPurple,prCyan,prLightPurple,prRed,read_variable_dict_amescap_profile,prYellow,filter_vars,get_longname_units
from amescap.FV3_utils import daily_to_average, daily_to_diurn,layers_mid_point_to_boundary
from amescap.Ncdf_wrapper import Ncdf, Fort
xr.set_options(keep_attrs=True)

#---
# MarsFormat.py
# Routine to Transform Model Input (variable names, dimension names, array order)
# to expected configuration CAP

# ======================================================
#                  ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(description="""\033[93m MarsFormat, Used to convert model output to FV3 format \n \033[00m""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_file', nargs='+',  # sys.stdin
                    help='***.nc file or list of ***.nc files')

parser.add_argument('-t', '--type', type=str, default='legacy',
                    help=""">  --type can be 'openmars', 'marswrf' or 'legacy' [DEFAULT is legacy] \n"""
                    """>  Usage: MarsFormat.py ****.nc \n"""
                    """          MarsFormat.py ****.nc -t openmars \n""")

parser.add_argument('-std', '--standard', action='store_true', default=False,
                    help=' Replace dimension and variable names with the standard FV3 convention \n'
                    '> Usage: MarsFormat.py file.nc -model_flag -std \n')

# ===========================
path2data = os.getcwd()

def main():

   if not (parser.parse_args().type):
         prYellow(''' ***Notice***  No operation requested. Use '-type' and specify openmars, marswrf, legacy ''')
         exit()  # Exit cleanly

   path2data = os.getcwd()
   
   # Load all of the netcdf files
   file_list    = parser.parse_args().input_file
   model_type  = parser.parse_args().type  # e.g. 'legacy'
   for filei in file_list:
      #Add path unless full path is provided
      if not ('/' in filei):
         fullnameIN = path2data + '/' + filei
      else:
         fullnameIN=filei
      fullnameOUT = fullnameIN[:-3]+'_atmos_daily.nc'

      print('Processing...')
      #Load model variables,dimensions
      fNcdf=Dataset(fullnameIN,'r')
      model=read_variable_dict_amescap_profile(fNcdf)
      fNcdf.close()
      
      exit()
      prCyan('Reading model attributes from ~.amescap_profile:')
      prCyan(vars(model)) #Print attribute
      #dataDIR = path+filename+'.nc'
      DS = xr.open_dataset(fullnameIN, decode_times=False)

      #=================================================================
      # ===================MarsWRF Specific Processing==================
      #=================================================================
      if model_type == 'marswrf':
         #TODO longname is 'description' for MarsWRF
         '''
         print('Input File content (description) and (description) attibutes:')
         print('------')
         for ivar in  DS.keys():
            print(ivar,DS[ivar].attrs['description'],DS[ivar].attrs['units'])
         print('------')
         '''
         #==================================================================
         # First save all variable descriptions in attrs longname
         #================================================================== 
         for var_name in DS.data_vars:
            var = DS[var_name]
            if 'description' in var.attrs:
               var.attrs['long_name'] = var.attrs['description'] 
         
         #==================================================================
         # Reformat Dimension Variables/Coords as Needed
         #================================================================== 
         time        = DS[model.time]/ 60/ 24         # minutes since simulation start [m]
         lat = DS[model.lat][0,:,0]
         lon2D = DS[model.lon][0,:]
         lon = np.squeeze(lon2D[0,:])
         DS[model.lon],DS[model.lat],DS[model.time]=lon,lat,time
         
         # Derive half and full reference pressure levels (Pa)
         pfull = DS.P_TOP[0]+ DS.ZNU[0,:]* DS.P0 
         phalf= DS.P_TOP[0]+ DS.ZNW[0,:]* DS.P0
         DS = DS.assign_coords(phalf=phalf,pfull=pfull)
         DS.phalf.attrs['long_name'] = '(ADDED IN POST PROCESSING) pressure at layer interfaces'
         DS.phalf.attrs['description'] = '(ADDED IN POST PROCESSING) pressure at layer interfaces'
         DS.phalf.attrs['units'] = 'Pa'
 
         # Update dimensions
         DS = DS.assign_coords(dimensions='phalf')

         # Update Variable Description, Longname and Unit
         DS[model.lon].attrs['long_name'] = '(MODIFIED IN POST PROCESSING) ' + DS[model.lon].attrs['description']
         DS[model.lat].attrs['long_name'] = '(MODIFIED IN POST PROCESSING) ' + DS[model.lat].attrs['description']
        
         DS[model.time].attrs['long_name'] = '(MODIFIED IN POST PROCESSING) days since simulation start, time/60/24'
         DS[model.lon].attrs['description'] = '(MODIFIED IN POST PROCESSING) ' + DS[model.lon].attrs['description']
         DS[model.lat].attrs['description'] = '(MODIFIED IN POST PROCESSING) ' + DS[model.lat].attrs['description']

         DS[model.time].attrs['description'] = '(MODIFIED IN POST PROCESSING) days since simulation start, time/60/24'
         DS[model.time].attrs['units'] = '(MODIFIED IN POST PROCESSING) days '
        

         #==================================================================
         # Derive ak, bk
         #==================================================================
         ak  = np.zeros(len(DS.phalf))
         bk = np.zeros(len(DS.phalf))
         ak[-1]=DS.P_TOP[0]  #MarsWRF comes with pressure increasing with N
         bk[:]=DS.ZNW[0,:]

         DS['ak'],DS['bk'] = ak,bk

         # Update Variable Description & Longname
         DS['ak'].attrs['long_name'], DS['bk'].attrs['long_name'] = '(ADDED IN POST PROCESSING)', '(ADDED IN POST PROCESSING)'
         DS['ak'].attrs['description'], DS['bk'].attrs['description'] = '(ADDED IN POST PROCESSING)', '(ADDED IN POST PROCESSING)'
        
         #==================================================================
         # Calculate *Level* Heights above the Surface (i.e. above topo)
         #==================================================================
         zagl_lvl = (DS.PH[:,:-1,:,:] + DS.PHB[0,:-1,:,:]) / DS.G - DS.HGT[0,:,:]

         #==================================================================
         # Find Layer Pressures [Pa]
         #=================================================================
         try:
            pfull3D = DS.P_TOP + DS.PB[0,:] # perturb. pressure + base state pressure, time-invariant
         except NameError:
            pfull3D = DS[model.ps][:,:-1,:-1] * DS.ZNU[:,:-1]

         #==================================================================
         # Derive attmospherice temperature [K]
         #==================================================================
         gamma = DS.CP / (DS.CP - DS.R_D)
         temp = (DS.T + DS.T0) * (pfull3D / DS.P0)**((gamma-1.) / gamma)
         DS = DS.assign(temp=temp)
         DS['temp'].attrs['description'] = '(ADDED IN POST PROCESSING) Temperature'
         DS['temp'].attrs['long_name'] = '(ADDED IN POST PROCESSING) Temperature'
         DS['temp'].attrs['units'] = 'K'

         #==================================================================
         # Interpolate U, V, W, Zfull onto Regular Mass Grid (from staggered)
         #==================================================================
         # For variables staggered x (lon) [t,z,y,x'] -> regular mass grid [t,z,y,x]:
         # Step 1: Identify variables with the dimension 'west_east_stag' and _U not in the variable name (these are staggered grid identifiers like the staggered latitude, I think
         variables_with_west_east_stag = [var for var in DS.variables if 'west_east_stag' in DS[var].dims and '_U' not in var]

         prCyan('Interpolating Staggered Variables to Standard Grid')
         # Loop through, and unstagger. the dims_list process finds the dimensoins of the variable and replaces west_east_stag with west_east
         prLightPurple('     From west_east_stag to west_east: ' + ', '.join(variables_with_west_east_stag))
         for var_name in variables_with_west_east_stag:
            var = getattr(DS, var_name)
            dims = var.dims
            # replace west_east_stag in dimensions with west_east
            dims_list = list(dims)
            for i, dim in enumerate(dims_list):
               if dim == 'west_east_stag':
                  dims_list[i]='west_east'
                  break # Stop the loop once the replacement is made
            new_dims = tuple(dims_list)
            
            transformed_var = 0.5 * (var.sel(west_east_stag=slice(None, -1)) + var.sel(west_east_stag=slice(1, None)))
            DS[var_name] = xr.DataArray(transformed_var, dims=new_dims, coords={'XLAT':DS['XLAT']})
           
            DS[var_name].attrs['description'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']
            DS[var_name].attrs['long_name'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']
            DS[var_name].attrs['stagger'] = 'USTAGGERED IN POST-PROCESSING'
           
        
         # For variables staggered y (lat) [t,z,y',x] -> regular mass grid [t,z,y,x]: 
         variables_with_south_north_stag = [var for var in DS.variables if 'south_north_stag' in DS[var].dims and '_V' not in var]
         prLightPurple('     From south_north_stag to south_north: '  + ', '.join(variables_with_south_north_stag))
         for var_name in variables_with_south_north_stag:
            var = getattr(DS, var_name)
            dims = var.dims
            # replace west_east_stag in dimensions with west_east
            dims_list = list(dims)
            for i, dim in enumerate(dims_list):
               if dim == 'south_north_stag':
                  dims_list[i]='south_north'
                  break # Stop the loop once the replacement is made
            new_dims = tuple(dims_list)
            
            transformed_var = 0.5 * (var.sel(south_north_stag=slice(None, -1)) + var.sel(south_north_stag=slice(1, None)))
            DS[var_name] = xr.DataArray(transformed_var, dims=new_dims, coords={'XLONG':DS['XLONG']})
           
            DS[var_name].attrs['description'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']

            DS[var_name].attrs['long_name'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']
            DS[var_name].attrs['stagger'] = 'USTAGGERED IN POST-PROCESSING'
         
         #DS[model.vcomp] = 0.5 * (DS[model.vcomp][:,:,:-1,:] + DS[model.vcomp][:,:,1:,:])

         # For variables staggered p/z (height) [t,z',y,x] -> regular mass grid [t,z,y,x]:
         variables_with_bottom_top_stag = [var for var in DS.variables if 'bottom_top_stag' in DS[var].dims and 'ZNW' not in var and 'phalf' not in var]
         prLightPurple('     From bottom_top_stag to bottom_top: '  + ', '.join(variables_with_bottom_top_stag))
         for var_name in variables_with_bottom_top_stag:
            var = getattr(DS, var_name)
            dims = var.dims
            # replace bottom_top_stag
            dims_list = list(dims)
            for i, dim in enumerate(dims_list):
               if dim == 'bottom_top_stag':
                  dims_list[i]='bottom_top'
                  break # Stop the loop once the replacement is made
            new_dims = tuple(dims_list)
            transformed_var = 0.5 * (var.sel(bottom_top_stag=slice(None, -1)) + var.sel(bottom_top_stag=slice(1, None)))
            DS[var_name] = xr.DataArray(transformed_var, dims=new_dims)
            DS[var_name].attrs['description'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']
            DS[var_name].attrs['long_name'] = '(UNSTAGGERED IN POST-PROCESSING) ' + DS[var_name].attrs['description']
            DS[var_name].attrs['stagger'] = 'USTAGGERED IN POST-PROCESSING'
         

         #DS[model.w] = 0.5 * (DS[model.w][:,:-1,:,:] + DS[model.w][:,1:,:,:])

         # ALSO INTERPOLATE TO FIND *LAYER* HEIGHTS ABOVE THE SURFACE (i.e., above topography; m)
         zfull3D = 0.5 * (zagl_lvl[:,:-1,:,:] + zagl_lvl[:,1:,:,:])

      #=================================================================
      # ===================OpenMars Specific Processing==================
      #=================================================================
      elif model_type == 'openmars':
         '''
         print('Input File content (FIELDNAM) and (UNITS) attibutes:')
         print('------')
         for ivar in  DS.keys():
            print(ivar,DS[ivar].attrs['FIELDNAM'],DS[ivar].attrs['UNITS'])
         print('------')
         '''
         #==================================================================
         # First save all variable FIELDNAM in attrs longname
         #==================================================================
         for var_name in DS.data_vars:
            var = DS[var_name]
            if 'FIELDNAM' in var.attrs:
               var.attrs['long_name'] = var.attrs['FIELDNAM']

         #==================================================================
         # Define Coordinates for New DataFrame
         #==================================================================
         ref_press=720 #TODO this is added on to create ak/bk
         time        = DS[model.dim_time]         # minutes since simulation start [m]
         lat = DS[model.dim_lat]  #Replace DS.lat
         lon = DS[model.dim_lon]

         # Derive half and full reference pressure levels (Pa)
         DS = DS.assign(pfull=DS[model.dim_pfull]*ref_press)
         DS['pfull'].attrs['FIELDNAM']='(MODIFIED IN POST-PROCESSING) ' + DS['pfull'].attrs['FIELDNAM']
         DS['pfull'].attrs['long_name']='(MODIFIED IN POST-PROCESSING) ' + DS['pfull'].attrs['FIELDNAM']

         #==================================================================
         # add ak,bk as variables
         # add p_half dimensions as vertical grid coordinate 
         #==================================================================

         #Compute sigma values. Swap the sigma array upside down twice  with [::-1] since the layers_mid_point_to_boundary() needs sigma[0]=0, sigma[-1]=1) and then to reorganize the array in the original openMars format with sigma[0]=1, sigma[-1]=0
         bk = layers_mid_point_to_boundary(DS[model.dim_pfull][::-1],1.)[::-1]
         ak = np.zeros(len(DS[model.dim_pfull]) + 1)

         DS['phalf']= ak + ref_press*bk
         DS.phalf.attrs['long_name'] = '(ADDED IN POST PROCESSING) pressure at layer interfaces'
         DS.phalf.attrs['description'] = '(ADDED IN POST PROCESSING) pressure at layer interfaces'
         DS.phalf.attrs['units'] = 'Pa'
          
         DS = DS.assign(bk=(model.dim_phalf, np.array(bk)))
         DS = DS.assign(ak=(model.dim_phalf, np.zeros(len(DS[model.dim_pfull]) + 1)))
 
         # Update Variable Description & Longname
         DS['ak'].attrs['long_name'], DS['bk'].attrs['long_name'] = '(ADDED IN POST PROCESSING)', '(ADDED IN POST PROCESSING)'
         DS['ak'].attrs['FIELDNAM'], DS['bk'].attrs['FIELDNAM'] = '(ADDED IN POST PROCESSING)', '(ADDED IN POST PROCESSING)'
        


      #==================================================================
      # START PROCESSING FOR ALL MODELS
      #==================================================================

      #==================================================================
      # check that vertical grid starts at toa with highest level at surface
      #==================================================================

      if DS[model.dim_pfull][0] != DS[model.dim_pfull].min(): # if toa, lev = 0 is surface then flip
          DS = DS.isel(**{model.dim_pfull: slice(None, None, -1)})
          DS=DS.isel(**{model.dim_phalf: slice(None, None, -1)}) #Also flip phalf,ak, bk
          prRed('NOTE: all variables flipped along vertical dimension, so that the top of the atmosphere is now index 0')

      #==================================================================
      # reorder dimensions
      #==================================================================
      prCyan('Transposing variable dimensions to match order expected in CAP') 
      DS = DS.transpose(model.dim_time, model.dim_pfull ,model.dim_lat,model.dim_lon, ...)

      #==================================================================
      # change longitude from -180-179 to 0-360
      #==================================================================
      if min(DS[model.dim_lon]) < 0:      
            tmp = np.array(DS[model.dim_lon])
            tmp = np.where(tmp<0,tmp+360,tmp)
            DS[model.dim_lon] = tmp
#DS=DS.assign_coords({model.dim_lon:(model.dim_lon,tmp,DS[model.dim_lon].attrs)})
            DS = DS.sortby(model.dim_lon)
            prRed('NOTE: Longitude changed to 0-360E and all variables appropriately reindexed')

      #==================================================================
      # add scalar axis to areo [time, scalar_axis])
      #==================================================================
      inpt_dimlist = DS.dims
      # first check if dimensions are correct and don't need to be modified
      if 'scalar_axis' not in inpt_dimlist:           # first see if scalar axis is a dimension
            scalar_axis = DS.assign_coords(scalar_axis=1)
      if DS[model.areo].dims != (model.time,scalar_axis):
            DS[model.areo] = DS[model.areo].expand_dims('scalar_axis', axis=1)
            DS[model.areo].attrs['long_name'] = '(SCALAR AXIS ADDED IN POST-PROCESSING) ' + DS[model.areo].attrs['long_name']
           
            prRed('NOTE: scalar axis added to aerocentric longitude')

      #==================================================================
      # CREATE ATMOS_DAILY, ATMOS_AVERAGE, & ATMOS_DIURN FILES
      #==================================================================
      
      if parser.parse_args().standard: 
         
         print(rev_dict)
         DS.rename_vars(rev_dict)
      #==================================================================
      # Output Processed Data to New **atmos_daily.nc File
      #==================================================================
      DS.to_netcdf(fullnameOUT)
      prCyan(fullnameOUT +' was created')

      '''
      #==================================================================
      # Output Binned Data to New **atmos_average.nc file
      #==================================================================
      fullnameOUT = fullnameIN[:-3]+'_atmos_average'+'.nc'

      # Figure out number of timesteps per 5 sol
      dt_in = DS[model.time][1]-DS[model.time][0]
      iperday = int(np.round(1/dt_in))
      combinedN = int(iperday*5)
      time = model.dim_time
      
      # Coarsen the 'time' dimension by a factor of 5 and average over each window
      DS_average = DS.coarsen(**{model.dim_time:combinedN}).mean()

      # Update the time coordinate attribute
      DS_average[model.time].attrs['long_name'] = 'time averaged over 5 sols'

      # Create New File
      DS_average.to_netcdf(fullnameOUT)
      prCyan(fullnameOUT +' was created')

      #==================================================================
      # Output Binned Data to New  **atmos_diurn.nc file
      #==================================================================
      fullnameOUT = fullnameIN[:-3]+'_atmos_diurn'+'.nc'

      # create a new time of day dimension
      tod_name = 'time_of_day_%02d' %(iperday)
      days = len(DS[model.time])/iperday

      # initialize the new dataset
      DS_diurn = None

      # loop through coarsened grid, slicing time dimension in 5 sol groups 
      for i in range(0, int(days/5)):
       
         # slice original dataset in 5 sol periods 
         downselect = DS.isel(**{model.time:slice(i*combinedN, i*combinedN+combinedN)})
   
         # rename the time dimension to the time of day and find the LT equivalent
         downselect = downselect.rename({model.time: tod_name})
         downselect[tod_name] = np.mod(downselect[tod_name]*24, 24).values
    
         # Group up all instances of the same LT & take the mean
         idx = downselect.groupby(tod_name).mean()
    
         # add back in the time dimensionn
         idx = idx.expand_dims({model.time: [i]})  # Add 'time' dimension with integer values
    
         # concatenate into new diurn array with a LT and time dimension (stack along time)
         if DS_diurn is None:
            DS_diurn = idx
         else:
            DS_diurn = xr.concat([DS_diurn, idx], dim=model.time)

      # replace the time dimension with the time dimension from DS_average
      DS_diurn[model.time] = DS_average[model.time]
 
      # Update the time coordinate attribute
      DS_diurn[model.time].attrs['long_name'] = 'time averaged over 5 sols'
  
      # Create New File
      DS_diurn.to_netcdf(fullnameOUT)
      prCyan(fullnameOUT +' was created')
      '''
if __name__ == '__main__':
    main()
