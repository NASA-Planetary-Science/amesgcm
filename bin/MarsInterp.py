#!/usr/bin/env python3

# Load generic Python Modules
import argparse   # parse arguments
import os         # access operating systems function
import subprocess  # run command
import sys        # system command
import time       # monitor interpolation time
import re         # string matching module to handle time_of_day_XX

# ==========
from amescap.FV3_utils import fms_press_calc, fms_Z_calc, vinterp, find_n, polar2XYZ, interp_KDTree, axis_interp
from amescap.Script_utils import check_file_tape, prYellow, prRed, prCyan, prGreen, prPurple, print_fileContent
from amescap.Script_utils import read_variable_dict_amescap_profile
from amescap.Script_utils import section_content_amescap_profile, find_tod_in_diurn, filter_vars, find_fixedfile, ak_bk_loader
from amescap.Ncdf_wrapper import Ncdf
# ==========

# Attempt to import specific scientic modules that may or may not
# be included in the default Python installation on NAS.
try:
    import matplotlib
    matplotlib.use('Agg') # Force matplotlib NOT to use any Xwindows backend
    import numpy as np
    from netCDF4 import Dataset, MFDataset

except ImportError as error_msg:
    prYellow("Error while importing modules")
    prYellow('You are using Python version '+str(sys.version_info[0:3]))
    prYellow('Please source your virtual environment, e.g.:')
    prCyan('    source envPython3.7/bin/activate.csh \n')
    print("Error was: " + error_msg.message)
    exit()
except Exception as exception:
    # Output unexpected Exceptions
    print(exception, False)
    print(exception.__class__.__name__ + ": " + exception.message)
    exit()

# ======================================================
#                  ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(description="""\033[93m MarsInterp, pressure interpolation on fixed layers\n \033[00m""",
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_file', nargs='+',  # sys.stdin
                    help='***.nc file or list of ***.nc files')
parser.add_argument('-t', '--type', type=str, default='pstd',
                    help=""">  --type can be 'pstd', 'zstd' or 'zagl' [DEFAULT is pstd, 36 levels] \n"""
                    """>  Usage: MarsInterp.py ****.atmos.average.nc \n"""
                    """          MarsInterp.py ****.atmos.average.nc -t zstd \n""")

parser.add_argument('-l', '--level', type=str, default=None,
                    help=""">  Layer IDs as defined in the ~/.amescap_profile hidden file. \n"""
                    """(For first time use, copy ~/.amescap_profile to ~/amesCAP, e.g.: \n"""
                    """\033[96mcp ~/amesCAP/mars_templates/amescap_profile ~/.amescap_profile\033[00m) \n"""
                    """>  Usage: MarsInterp.py ****.atmos.average.nc -t pstd -l p44 \n"""
                    """          MarsInterp.py ****.atmos.average.nc -t zstd -l phalf_mb \n""")

parser.add_argument('-include', '--include', nargs='+',
                    help="""Only include the listed variables. Dimensions and 1D variables are always included. \n"""
                    """> Usage: MarsInterp.py *.atmos_daily.nc --include ps ts temp \n"""
                         """\033[00m""")

parser.add_argument('-e', '--ext', type=str, default=None,
                    help="""> Append an extension (_ext.nc) to the output file instead of replacing the existing file. \n"""
                    """>  Usage: MarsInterp.py ****.atmos.average.nc -ext B \n"""
                    """   This will produce ****.atmos.average_pstd_B.nc files \n""")

parser.add_argument('-g', '--grid', action='store_true',
                    help="""> Output current grid information to standard output. This will not run the interpolation. """
                    """>  Usage: MarsInterp.py ****.atmos.average.nc -t pstd -l p44 -g \n""")

parser.add_argument('--debug',  action='store_true',
                    help='Debug flag: release the exceptions.')


# =====================================================================
# =====================================================================
# =====================================================================
# TODO: If only one time step, reshape from (lev,lat,lon) to (time, lev, lat, lon).

# Fill values for NaN. Do not use np.NaN - it is deprecated and will raise issues when using runpinterp
fill_value = 0.

# Define constants
rgas  = 189.   # J/(kg-K) -> m2/(s2 K)
g     = 3.72   # m/s2
R     = 8.314  # J/ mol. K
Cp    = 735.0  # J/K
M_co2 = 0.044  # kg/mol

# ===========================
filepath = os.getcwd()

def main():
    start_time   = time.time()
    debug        = parser.parse_args().debug
    # Load all of the netcdf files
    file_list    = parser.parse_args().input_file
    interp_type  = parser.parse_args().type  # e.g. 'pstd'
    custom_level = parser.parse_args().level # e.g. 'p44'
    grid_out     = parser.parse_args().grid

    # PRELIMINARY DEFINITIONS
    # =========================== pstd ===========================
    if interp_type == 'pstd':
        longname_txt    = 'standard pressure'
        units_txt       = 'Pa'
        need_to_reverse = False
        interp_technic  = 'log'
        if custom_level:
            content_txt = section_content_amescap_profile(
                'Pressure definitions for pstd')
            # print(content_txt)
            exec(content_txt)  # Load all variables in that section
            # Copy requested variable
            lev_in = eval('np.array('+custom_level+')')
        else:
            # Default levels, this is size 36
            lev_in = np.array([1.0e+03, 9.5e+02, 9.0e+02, 8.5e+02, 8.0e+02, 7.5e+02, 7.0e+02,
                               6.5e+02, 6.0e+02, 5.5e+02, 5.0e+02, 4.5e+02, 4.0e+02, 3.5e+02,
                               3.0e+02, 2.5e+02, 2.0e+02, 1.5e+02, 1.0e+02, 7.0e+01, 5.0e+01,
                               3.0e+01, 2.0e+01, 1.0e+01, 7.0e+00, 5.0e+00, 3.0e+00, 2.0e+00,
                               1.0e+00, 5.0e-01, 3.0e-01, 2.0e-01, 1.0e-01, 5.0e-02, 3.0e-02,
                               1.0e-02])

    # =========================== zstd ===========================
    elif interp_type == 'zstd':
        longname_txt    = 'standard altitude'
        units_txt       = 'm'
        need_to_reverse = True
        interp_technic  = 'lin'
        if custom_level:
            content_txt = section_content_amescap_profile(
                'Altitude definitions for zstd')
            exec(content_txt)  # Load all variables in that section
            # Copy requested variable
            lev_in = eval('np.array('+custom_level+')')
        else:
            # Default levels, this is size 45
            lev_in = np.array([-7000, -6000, -5000, -4500, -4000, -3500, -3000, -2500, -2000, -1500, -1000,
                               -500, 0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                               6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000,
                               20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
                               60000, 70000, 80000, 90000, 100000])

        # The fixed file is necessary if pk, bk are not in the requested file, or
        # to load the topography if zstd output is requested.
        name_fixed = find_fixedfile(file_list[0])
        try:
            f_fixed = Dataset(name_fixed, 'r')
            model=read_variable_dict_amescap_profile(f_fixed)
            zsurf = f_fixed.variables[model.zsurf][:]
            f_fixed.close()
        except FileNotFoundError:
            prRed('***Error*** Topography (zsurf) is required for interpolation to zstd, but the')
            prRed('file %s cannot be not found' % (name_fixed))
            exit()

    # =========================== zagl ===========================
    elif interp_type == 'zagl':
        longname_txt    = 'altitude above ground level'
        units_txt       = 'm'
        need_to_reverse = True
        interp_technic  = 'lin'
        if custom_level:
            content_txt = section_content_amescap_profile(
                'Altitude definitions for zagl')
            # print(content_txt)
            exec(content_txt)  # Load all variables in that section
            # Copy requested variable
            lev_in = eval('np.array('+custom_level+')')
        else:
            # Default levels, this is size 45
            lev_in = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                               6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 18000,
                               20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000,
                               60000, 70000, 80000, 90000, 100000, 110000])
    else:
        prRed("Interpolation type '%s' is not supported, use  'pstd','zstd' or 'zagl'" % (
            interp_type))
        exit()
    # Only print grid content and exit the code
    if grid_out:
        print(*lev_in)
        exit()

    # For all the files:
    for ifile in file_list:
        # First check if file is present on the disk (Lou only)
        check_file_tape(ifile)

        # Append extension, if any
        if parser.parse_args().ext:
            newname = filepath+'/'+ifile[:-3]+'_' + \
                interp_type+'_'+parser.parse_args().ext+'.nc'
        else:
            newname = filepath+'/'+ifile[:-3]+'_'+interp_type+'.nc'

        # =================================================================
        # ======================== Interpolation ==========================
        # =================================================================

        fNcdf = Dataset(ifile, 'r', format='NETCDF4_CLASSIC')
        # Load pk, bk, and ps for 3D pressure field calculation.
        # Read the pk and bk for each file in case the vertical resolution has changed.
        model=read_variable_dict_amescap_profile(fNcdf)
        ak, bk = ak_bk_loader(fNcdf)

        ps = np.array(fNcdf.variables[model.ps])

        if len(ps.shape) == 3:
            do_diurn = False
            tod_name = 'not_used'
            # Put vertical axis first for 4D variable, e.g (time, lev, lat, lon) >>> (lev, time, lat, lon)
            permut = [1, 0, 2, 3]
            # ( 0 1 2 3 ) >>> ( 1 0 2 3 )
        elif len(ps.shape) == 4:
            do_diurn = True
            # Find 'time_of_day' variable name
            tod_name = find_tod_in_diurn(fNcdf)
            # Same for 'diurn' files, e.g (time, time_of_day_XX, lev, lat, lon) >>> (lev, time_of_day_XX, time, lat, lon)
            permut = [2, 1, 0, 3, 4]
            # ( 0 1 2 3 4) >>> ( 2 1 0 3 4 )

        # Compute levels in the file, these are permutted arrays
        # Suppress "divide by zero" error
        with np.errstate(divide='ignore', invalid='ignore'):
            if interp_type == 'pstd':
                # Permute by default dimension, e.g lev is first
                L_3D_P = fms_press_calc(ps, ak, bk, lev_type='full')

            elif interp_type == 'zagl':
                temp = fNcdf.variables[model.temp][:]
                L_3D_P = fms_Z_calc(ps, ak, bk, temp.transpose(
                    permut), topo=0., lev_type='full')

            elif interp_type == 'zstd':
                temp = fNcdf.variables[model.temp][:]
                # Expand the 'zsurf' array to the 'time' dimension
                zflat = np.repeat(zsurf[np.newaxis, :], ps.shape[0], axis=0)
                if do_diurn:
                    zflat = np.repeat(
                        zflat[:, np.newaxis, :, :], ps.shape[1], axis=1)

                L_3D_P = fms_Z_calc(ps, ak, bk, temp.transpose(
                    permut), topo=zflat, lev_type='full')

        fnew = Ncdf(newname, 'Pressure interpolation using MarsInterp.py')

        # Copy existing DIMENSIONS other than pfull
        # Get all variables in the file
        # var_list=fNcdf.variables.keys()
        var_list = filter_vars(
            fNcdf, parser.parse_args().include)  # Get the variables

        fnew.copy_all_dims_from_Ncfile(fNcdf, exclude_dim=['pfull'])
        # Add new vertical dimension
        fnew.add_dim_with_content(interp_type, lev_in, longname_txt, units_txt)

        if 'tile' in ifile:
            fnew.copy_Ncaxis_with_content(fNcdf.variables['grid_xt'])
            fnew.copy_Ncaxis_with_content(fNcdf.variables['grid_yt'])
        else:
            fnew.copy_Ncaxis_with_content(fNcdf.variables[model.lon])
            fnew.copy_Ncaxis_with_content(fNcdf.variables[model.lat])

        fnew.copy_Ncaxis_with_content(fNcdf.variables[model.time])

        if do_diurn:
            fnew.copy_Ncaxis_with_content(fNcdf.variables[tod_name])

        # Re-use the indices for each file, this speeds up the calculation
        compute_indices = True
        for ivar in var_list:
            if (fNcdf.variables[ivar].dimensions == ('time', 'pfull', 'lat', 'lon') or
                fNcdf.variables[ivar].dimensions == ('time', tod_name, 'pfull', 'lat', 'lon') or
                    fNcdf.variables[ivar].dimensions == ('time', 'pfull', 'grid_yt', 'grid_xt')):
                if compute_indices:
                    prCyan("Computing indices ...")
                    index = find_n(
                        L_3D_P, lev_in, reverse_input=need_to_reverse)
                    compute_indices = False

                prCyan("Interpolating: %s ..." % (ivar))
                varIN = fNcdf.variables[ivar][:]
                # This with the loop suppresses "divide by zero" errors
                with np.errstate(divide='ignore', invalid='ignore'):
                    varOUT = vinterp(varIN.transpose(permut), L_3D_P,
                                     lev_in, type_int=interp_technic, reverse_input=need_to_reverse,
                                     masktop=True, index=index).transpose(permut)

                long_name_txt = getattr(fNcdf.variables[ivar], 'long_name', '')
                units_txt = getattr(fNcdf.variables[ivar], 'units', '')
                # long_name_txt=fNcdf.variables[ivar].long_name
                # units_txt=fNcdf.variables[ivar].units)

                if not do_diurn:
                    if 'tile' in ifile:
                        fnew.log_variable(ivar, varOUT, ('time', interp_type, 'grid_yt', 'grid_xt'),
                                          long_name_txt, units_txt)
                    else:
                        fnew.log_variable(ivar, varOUT, ('time', interp_type, 'lat', 'lon'),
                                          long_name_txt, units_txt)
                else:
                    if 'tile' in ifile:
                        fnew.log_variable(ivar, varOUT, ('time', tod_name, interp_type, 'grid_yt', 'grid_xt'),
                                          long_name_txt, units_txt)
                    else:
                        fnew.log_variable(ivar, varOUT, ('time', tod_name, interp_type, 'lat', 'lon'),
                                          long_name_txt, units_txt)
            else:

                if ivar not in [model.time, model.pfull, model.lat, model.lon, 'phalf', 'ak', 'pk', 'bk', model.pstd, model.zstd, model.zagl, tod_name, 'grid_xt', 'grid_yt']:
                    #print("\r Copying over: %s..."%(ivar), end='')
                    prCyan("Copying over: %s..." % (ivar))
                    fnew.copy_Ncvar(fNcdf.variables[ivar])

        print('\r ', end='')
        fNcdf.close()
        fnew.close()
        print("Completed in %.3f sec" % (time.time() - start_time))


if __name__ == '__main__':
    main()
