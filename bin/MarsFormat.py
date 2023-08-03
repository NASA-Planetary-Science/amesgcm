#!/usr/bin/env python3

#Load generic Python Modules
import argparse   # parse arguments
import numpy as np
import xarray as xr
from netCDF4 import Dataset
import os
from amescap.Script_utils import prCyan,prRed,read_variable_dict_amescap_profile,prYellow
from amescap.FV3_utils import layers_mid_point_to_boundary

xr.set_options(keep_attrs=True)

#---
# fit2FV3.py
# Routine to Transform Model Input (variable names, dimension names, array order)
# to expected configuration CAP

parser = argparse.ArgumentParser(description="""\033[93m fit2FV3.py  Used to convert model output to FV3 format  \n \033[00m""",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_file', nargs='+',  # sys.stdin
                    help='***.nc file or list of ***.nc files ')


parser.add_argument('-openmars', '--openmars', nargs='+',
                    help="""Produce a FV3-like daily file \n"""
                    """> Usage: MarsFormat.py fileIN*.nc --model \n"""
                    """> Available options are:                     \n"""
                    """(-openmars)  --openmars  daily \n"""
                    """(-marswrf)   --marswrf   daily \n"""
                    """\n""")

parser.add_argument('-marswrf', '--marswrf', nargs='+',
                    help=argparse.SUPPRESS)

parser.add_argument('-legacy', '--legacy', nargs='+',
                    help=argparse.SUPPRESS)

def main():

   if not (parser.parse_args().marswrf or parser.parse_args().openmars or parser.parse_args().legacy):
         prYellow(''' ***Notice***  No operation requested. Use '-marswrf', '-openmars' or  '-legacy ''')
         exit()  # Exit cleanly

   path2data = os.getcwd()
   # Open a single File
   file_list=parser.parse_args().input_file
   #path_inpt
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
      prCyan('Reading model attributes from ~.amescap_profile:')
      prCyan(vars(model)) #Print attribute
      #dataDIR = path+filename+'.nc'
      DS = xr.open_dataset(fullnameIN, decode_times=False)

      #=================================================================
      # ===================OpenMars Specific Processing==================
      #=================================================================
      if parser.parse_args().marswrf:
         #TODO longname is 'description' for MarsWRF
         '''
         print('Input File content (description) and (description) attibutes:')
         print('------')
         for ivar in  DS.keys():
            print(ivar,DS[ivar].attrs['description'],DS[ivar].attrs['units'])
         print('------')
         '''
         #==================================================================
         # Find Shape of Coordinates
         #==================================================================
         # [t,z,y,x] = 100,43,90,180
         ppt_dims = np.shape(DS.T)
         lmax = ppt_dims[3] # x = 180
         jmax = ppt_dims[2] # y = 90
         tmax = ppt_dims[0] # t = 100
         pmax = ppt_dims[1] # z = 43 (layer)

         #==================================================================
         # Define Coordinates for New DataFrame
         #==================================================================
         time        = DS.XTIME/ 60/ 24         # minutes since simulation start [m]
         lat = DS[model.lat][0,:,0]
         lon2D = DS[model.lon][0,:]
         lon = np.squeeze(lon2D[0,:])

         # Derive half and full reference pressure levels (Pa)
         pfull = DS.P_TOP[0]+ DS.ZNU[0,:]* DS.P0
         phalf = DS.P_TOP[0]+ DS.ZNW[0,:]* DS.P0

         #==================================================================
         # Calculate *Level* Heights above the Surface (i.e. above topo)
         #==================================================================
         zagl_lvl = (DS.PH[:,:pmax,:,:] + DS.PHB[0,:pmax,:,:]) / DS.G - DS.HGT[0,:,:]

         #==================================================================
         # Find Layer Pressures [Pa]
         #==================================================================
         try:
            pfull3D = DS.P_TOP + DS.PB[0,:] # perturb. pressure + base state pressure, time-invariant
         except NameError:
            pfull3D = DS[model.ps][:,:jmax,:lmax] * DS.ZNU[:,:pmax]

         #==================================================================
         # Interpolate U, V, W, Zfull onto Regular Mass Grid (from staggered)
         #==================================================================
         # For variables staggered x (lon) [t,z,y,x'] -> regular mass grid [t,z,y,x]:
         ucomp = 0.5 * (DS[model.ucomp][..., :-1] + DS[model.ucomp][..., 1:])

         # For variables staggered y (lat) [t,z,y',x] -> regular mass grid [t,z,y,x]:
         vcomp = 0.5 * (DS[model.vcomp][:,:,:-1,:] + DS[model.vcomp][:,:,1:,:])

         # For variables staggered p/z (height) [t,z',y,x] -> regular mass grid [t,z,y,x]:
         w = 0.5 * (DS[model.w][:,:-1,:,:] + DS[model.w][:,1:,:,:])

         # ALSO INTERPOLATE TO FIND *LAYER* HEIGHTS ABOVE THE SURFACE (i.e., above topography; m)
         zfull3D = 0.5 * (zagl_lvl[:,:-1,:,:] + zagl_lvl[:,1:,:,:])

         #==================================================================
         # Derive attmospherice temperature [K]
         #==================================================================
         gamma = DS.CP / (DS.CP - DS.R_D)
         temp = (DS.T + DS.T0) * (pfull3D / DS.P0)**((gamma-1.) / gamma)

         #==================================================================
         # Derive ak, bk
         #==================================================================
         ak  = np.zeros(len(phalf))
         bk = np.zeros(len(phalf))
         ak[-1]=DS.P_TOP[0]  #MarsWRF comes with pressure increasing with N
         bk[:]=DS.ZNW[0,:]

         #==================================================================
         # Create New DataFrame
         #==================================================================
         coords = {model.dim_time: np.array(time), model.dim_phalf:np.array(phalf),model.dim_pfull: np.array(pfull), model.dim_lat: np.array(lat), model.dim_lon: np.array(lon)}  # Coordinates dictionary

         archive_vars = {
            'ak' :         [ak, [model.dim_phalf],'pressure part of the hybrid coordinate','Pa'],
            'bk' :         [bk, [model.dim_phalf],'vertical coordinate sigma value','none'],
            model.areo :   [DS[model.areo], [model.dim_time],'solar longitude','degree'],
            model.ps :     [DS[model.ps], [model.dim_time,model.dim_lat,model.dim_lon],'surface pressure','Pa'],
            model.zsurf:   [DS[model.zsurf][0,:],['lat','lon'],'surface height','m'],
            model.ucomp:   [ucomp, [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'zonal winds','m/sec'],
            model.vcomp:   [vcomp, [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'meridional winds','m/sec'],
            model.w:       [w, [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'vertical winds','m/s'],
            'pfull3D':     [pfull3D, [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'pressure','Pa'],
            model.temp:    [temp, [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'temperature','K'],
            'h2o_ice_sfc': [DS.H2OICE, [model.dim_time,model.dim_lat,model.dim_lon],'Surface H2O Ice','kg/m2'],
            'co2_ice_sfc': [DS.CO2ICE, [model.dim_time,model.dim_lat,model.dim_lon],'Surface CO2 Ice','kg/m2'],
            model.ts:      [DS[model.ts], [model.dim_time,model.dim_lat,model.dim_lon],'surface temperature','K'],
         }


      #=================================================================
      # ===================OpenMars Specific Processing==================
      #=================================================================
      elif parser.parse_args().openmars:
         '''
         print('Input File content (FIELDNAM) and (UNITS) attibutes:')
         print('------')
         for ivar in  DS.keys():
            print(ivar,DS[ivar].attrs['FIELDNAM'],DS[ivar].attrs['UNITS'])
         print('------')
         '''
         #==================================================================
         # Define Coordinates for New DataFrame
         #==================================================================
         ref_press=720 #TODO this is added on to create ak/bk
         time        = DS[model.dim_time]         # minutes since simulation start [m]
         lat = DS[model.dim_lat]  #Replace DS.lat
         lon = DS[model.dim_lon]
         # Derive half and full reference pressure levels (Pa)
         pfull = DS[model.dim_pfull]*ref_press


         #==================================================================
         # add p_half dimensions and ak, bk vertical grid coordinates
         #==================================================================

         #DS.expand_dims({'p_half':len(pfull)+1})
         #Compute sigma values. Swap the sigma array upside down twice  with [::-1] since the layers_mid_point_to_boundary() needs sigma[0]=0, sigma[-1]=1) and then to reorganize the array in the original openMars format with sigma[0]=1, sigma[-1]=0
         DS['bk'] = layers_mid_point_to_boundary(DS[model.dim_pfull][::-1],1.)[::-1]
         DS['ak'] = np.zeros(len(pfull)+1) #Pure sigma model, set bk to zero
         phalf= np.array(DS['ak']) + ref_press*np.array(DS['bk'])  #compute phalf


         #==================================================================
         # Make New DataFrame
         #==================================================================

         coords = {model.dim_time: np.array(time), model.dim_phalf: np.array(phalf), model.dim_pfull: np.array(pfull), model.dim_lat:np.array(lat), model.dim_lon: np.array(lon)}

         #Variable to archive [name, values, dimensions, longname,units]
         archive_vars = {
         'bk' :          [DS.bk, [model.dim_phalf],'vertical coordinate sigma value','none'],
         'ak' :          [DS.ak, [model.dim_phalf], 'pressure part of the hybrid coordinate','Pa'],
         model.areo :       [DS[model.areo], [model.dim_time],'solar longitude','degree'],
         model.ps :          [DS[model.ps], [model.dim_time,model.dim_lat,model.dim_lon],'surface pressure','Pa'],
         model.ucomp:        [DS[model.ucomp], [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'zonal winds','m/sec'],
         model.vcomp:        [DS[model.vcomp], [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'meridional wind','m/sec'],
         model.temp:         [DS[model.temp], [model.dim_time,model.dim_pfull,model.dim_lat,model.dim_lon],'temperature','K'],
         'dust_mass_col':    [DS.dustcol, [model.dim_time,model.dim_lat,model.dim_lon],'column integration of dust','kg/m2'],
         'co2_ice_sfc':      [DS.co2ice, [model.dim_time,model.dim_lat,model.dim_lon],'surace CO2 ice','kg/m2'],
         model.ts:           [DS[model.ts], [model.dim_time,model.dim_lat,model.dim_lon],'Surface Temperature','K']
         }



      #==================================================================
      #========Create output file (common to all models)=================
      #==================================================================
      archive_coords = {
      model.time:  ['time','days'],
      model.pfull: ['ref full pressure level','Pa'],
      model.lat:   ['latitudes' ,'degrees_N'],
      model.lon:   ['longitudes','degrees_E'],
      model.phalf: ['ref pressure at layer boundaries','Pa']
      }
      prYellow(archive_coords)
      # Empty xarray dictionary
      data_vars = {}
      # Assign description and units attributes to the xarray dictionary
      for ivar in archive_vars.keys():
         data_vars[ivar] = xr.DataArray(np.array(archive_vars[ivar][0]), dims=archive_vars[ivar][1])
         data_vars[ivar].attrs['long_name'] = archive_vars[ivar][2]
         data_vars[ivar].attrs['units'] =  archive_vars[ivar][3]

      # Create the dataset with the data variables and assigned attributes
      DF = xr.Dataset(data_vars, coords=coords)

      #Add longname and units attibutes to the coordiate variables
      prCyan(DF)
      for ivar in archive_coords.keys():
         print(ivar)
         DF[ivar].attrs['long_name']=archive_coords[ivar][0]
         DF[ivar].attrs['units']=archive_coords[ivar][1]



      #==================================================================
      # check that vertical grid starts at toa with highest level at surface
      #==================================================================
      #TODO comment: this poses an issue

      if DF[model.dim_pfull][0] != DF[model.dim_pfull].min(): # if toa, lev = 0 is surface then flip
          DF=DF.isel(pfull=slice(None, None, -1)) # regrids DS based on pfull
          DF=DF.isel(phalf=slice(None, None, -1)) #Also flip phalf,ak, bk


      #==================================================================
      # reorder dimensions
      #==================================================================
      DF = DF.transpose(model.dim_phalf,model.dim_time, model.dim_pfull ,model.dim_lat,model.dim_lon)


      #==================================================================
      # change longitude from -180-179 to 0-360
      #==================================================================
      if min(DF.lon)<0:
            tmp = np.array(DF.lon)
            tmp = np.where(tmp<0,tmp+360,tmp)
            DF=DF.assign_coords({model.dim_lon:(model.dim_lon,tmp,DF.lon.attrs)})
            DF = DF.sortby(model.dim_lon)

      #==================================================================
      # add scalar axis to areo [time, scalar_axis])
      #==================================================================
      inpt_dimlist = DF.dims
      # first check if dimensions are correct and don't need to be modified
      if 'scalar_axis' not in inpt_dimlist:           # first see if scalar axis is a dimension
            scalar_axis = DF.assign_coords(scalar_axis=1)
      if DF[model.areo].dims != (model.time,scalar_axis):
            DF[model.areo] = DF[model.areo].expand_dims('scalar_axis', axis=1)




      #==================================================================
      # Output Processed Data to New NC File
      #==================================================================
      DF.to_netcdf(fullnameOUT)
      prCyan(fullnameOUT +' was created')
if __name__ == '__main__':
    main()
