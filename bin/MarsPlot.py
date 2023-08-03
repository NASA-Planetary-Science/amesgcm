#!/usr/bin/env python3

from warnings import filterwarnings
filterwarnings('ignore', category = DeprecationWarning)

# Load generic Python modules
import argparse   # parse arguments
import os         # access operating systems function
import subprocess # run command
import sys        # system command

# ==========
from amescap.Script_utils import check_file_tape, prYellow, prRed, prCyan, prGreen, prPurple
from amescap.Script_utils import section_content_amescap_profile, print_fileContent, print_varContent, FV3_file_type, find_tod_in_diurn,read_variable_dict_amescap_profile
from amescap.Script_utils import wbr_cmap, rjw_cmap, dkass_temp_cmap, dkass_dust_cmap
from amescap.FV3_utils import lon360_to_180, lon180_to_360, UT_LTtxt, area_weights_deg
from amescap.FV3_utils import add_cyclic, azimuth2cart, mollweide2cart, robin2cart, ortho2cart
# ==========

# Attempt to import specific scientic modules that may or may not
# be included in the default Python installation on NAS.
try:
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib NOT to use any Xwindows backend
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import (
        LogFormatter, NullFormatter, LogFormatterSciNotation, MultipleLocator)  # Format ticks
    from netCDF4 import Dataset, MFDataset
    from numpy import sqrt, exp, max, mean, min, log, log10, sin, cos, abs
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import LogFormatter

except ImportError as error_msg:
    prYellow("Error while importing modules")
    prYellow('You are using Python version '+str(sys.version_info[0:3]))
    prYellow('Please source your virtual environment, e.g.:')
    prCyan('    source envPython3.7/bin/activate.csh \n')
    print("Error was: " + error_msg.message)
    exit()
except Exception as exception:
    # Output unexpected Exceptions.
    print(exception.__class__.__name__ + ": ", exception)
    exit()

degr = u"\N{DEGREE SIGN}"
global current_version
current_version = 3.3

# ======================================================
#                  ARGUMENT PARSER
# ======================================================
parser = argparse.ArgumentParser(description="""\033[93mAnalysis Toolkit for the MGCM, V%s\033[00m """ % (current_version),
                                 formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('custom_file', nargs='?', type=argparse.FileType('r'), default=None,  # sys.stdin
                    help='Use optional input file Custom.in to create the graphs. \n'
                    '> Usage: MarsPlot Custom.in  [other options]\n'
                    'Update CAP as needed with \033[96mpip install git+https://github.com/NASA-Planetary-Science/AmesCAP.git --upgrade\033[00m \n'
                    'Tutorial: \033[93mhttps://github.com/NASA-Planetary-Science/AmesCAP\033[00m')

parser.add_argument('-i', '--inspect_file', default=None,
                    help="""Inspect netcdf file content. Variables are sorted by dimensions. \n"""
                    """> Usage: MarsPlot -i 00000.atmos_daily.nc\n"""
                    """Options: use --dump (variable content) and --stat (min, mean,max) jointly with --inspect \n"""
                    """>  MarsPlot -i 00000.atmos_daily.nc -dump pfull 'temp[6,:,30,10]'  (quotes '' necessary for browsing dimensions)\n"""
                    """>  MarsPlot -i 00000.atmos_daily.nc -stat 'ucomp[5,:,:,:]' 'vcomp[5,:,:,:]'\n""")

# These two options are to be used jointly with --inspect
parser.add_argument('--dump', '-dump', nargs='+', default=None,
                    help=argparse.SUPPRESS)

parser.add_argument('--stat', '-stat', nargs='+', default=None,
                    help=argparse.SUPPRESS)

parser.add_argument('-d', '--date', nargs='+', default=None,
                    help='Specify the files to use. Default is the last file created. \n'
                    '> Usage: MarsPlot Custom.in -d 700     (one file) \n'
                    '         MarsPlot Custom.in -d 350 700 (start file end file)')

parser.add_argument('--template', '-template', action='store_true',
                    help="""Generate a template (Custom.in) for creating the plots.\n """
                    """(Use '--temp' to create a Custom.in file without these instructions)\n""")

parser.add_argument('-temp', '--temp', action='store_true',
                    help=argparse.SUPPRESS)  # Creates a Custom.in template without the instructions

parser.add_argument('-do', '--do', nargs=1, type=str, default=None,  # sys.stdin
                    help='(Re)use a template file (e.g. my_custom.in). Searches in ~/amesCAP/mars_templates/ first, \n'
                    'then in /u/mkahre/MCMC/analysis/working/shared_templates/ \n'
                    '> Usage: MarsPlot -do my_custom [other options]')

parser.add_argument('-sy', '--stack_year', action='store_true', default=False,
                    help='Stack consecutive years in 1D time series plots (recommended). Otherwise plots in monotonically increasing format.\n'
                    '> Usage: MarsPlot Custom.in -sy \n')

parser.add_argument("-o", "--output", default="pdf",
                    choices=['pdf', 'eps', 'png'],
                    help='Output file format.\n'
                    'Default is PDF if ghostscript (gs) is available and PNG otherwise\n'
                    '> Usage: MarsPlot Custom.in -o png \n'
                    '       : MarsPlot Custom.in -o png -pw 500 (set pixel width to 500, default is 2000)\n')

parser.add_argument('-vert', '--vertical', action='store_true', default=False,
                    help='Output figures in portrain instead of landscape format. \n')

parser.add_argument("-pw", "--pwidth", default=2000, type=float,
                    help=argparse.SUPPRESS)

parser.add_argument('-dir', '--directory', default=os.getcwd(),
                    help='Target directory if input files are not in current directory. \n'
                    '> Usage: MarsPlot Custom.in [other options] -dir /u/akling/FV3/verona/c192L28_dliftA/history')


parser.add_argument('--debug',  action='store_true',
                    help='Debug flag: do not bypass errors')

# ======================================================
#                  MAIN PROGRAM
# ======================================================
def main():
    global output_path, input_paths, out_format, debug
    output_path     = os.getcwd()
    out_format      = parser.parse_args().output
    debug           = parser.parse_args().debug
    input_paths     = []
    input_paths.append(parser.parse_args().directory)

    global Ncdf_num         # Hosts the simulation timestamps
    global objectList       # Contains all figure objects
    global customFileIN     # The Custom.in template name

    global levels, my_dpi, label_size, title_size, label_factor, tick_factor, title_factor
    levels          = 21    # Number of contours for 2D plots
    my_dpi          = 96.   # Pixels per inch for figure output
    label_size      = 18    # Label size for title, xlabel, and ylabel
    title_size      = 24    # Label size for title, xlabel, and ylabel
    label_factor    = 3/10  # Reduces the font size as the number of panels increases
    tick_factor     = 1/2
    title_factor    = 10/12

    global width_inch       # Pixel width for saving figure
    global height_inch      # Pixel width for saving figure
    global vertical_page

    # Portrait instead of landscape format for figure pages
    vertical_page = parser.parse_args().vertical

    # Directory containing shared templates
    global shared_dir
    shared_dir = '/u/mkahre/MCMC/analysis/working/shared_templates'

    # Set figure dimensions
    pixel_width = parser.parse_args().pwidth
    if vertical_page:
        width_inch  = pixel_width/1.4/my_dpi
        height_inch = pixel_width/my_dpi
    else:
        width_inch  = pixel_width/my_dpi
        height_inch = pixel_width/1.4/my_dpi

    objectList = [Fig_2D_lon_lat('fixed.zsurf', True),
                  Fig_2D_lat_lev('atmos_average.ucomp', True),
                  Fig_2D_time_lat('atmos_average.taudust_IR', False),
                  Fig_2D_lon_lev('atmos_average_pstd.temp', False),
                  Fig_2D_time_lev('atmos_average_pstd.temp', False),
                  Fig_2D_lon_time('atmos_average.temp', False),
                  Fig_1D('atmos_average.temp', False)]

    # =============================

    # Group together the first two figures
    objectList[0].subID = 1
    objectList[0].nPan  = 2  # 1st object in a 2 panel figure
    objectList[1].subID = 2
    objectList[1].nPan  = 2  # 2nd object in a 2 panel figure

    # Begin main loop:
    # Option 1: Inspect content of a netcdf file
    if parser.parse_args().inspect_file:
        # NAS-specific, check if the file is on tape (Lou only)
        check_file_tape(parser.parse_args().inspect_file, abort=False)

        if parser.parse_args().dump:
            # Dump variable content
            print_varContent(parser.parse_args().inspect_file,
                             parser.parse_args().dump, False)
        elif parser.parse_args().stat:
            # Print variable stats
            print_varContent(parser.parse_args().inspect_file,
                             parser.parse_args().stat, True)
        else:
            # Show information for all variables
            print_fileContent(parser.parse_args().inspect_file)

    # Option 2: Generate a template file
    elif parser.parse_args().template or parser.parse_args().temp:
        make_template()

    # Gather simulation information from template or inline argument
    else:
        # Option 2, case A: Use Custom.in for everything
        if parser.parse_args().custom_file:
            print('Reading '+parser.parse_args().custom_file.name)
            namelist_parser(parser.parse_args().custom_file.name)

        # Option 2, case B: Use Custom.in from ~/FV3/templates for everything
        if parser.parse_args().do:
            print('Reading '+path_to_template(parser.parse_args().do))
            namelist_parser(path_to_template(parser.parse_args().do))

        # Set bounds (e.g. start file, end file)
        if parser.parse_args().date: # a single date or a range of dates is provided
            # First check if the value provided is of the right type
            try:
                bound = np.asarray(parser.parse_args().date).astype(float)
            except Exception as e:
                prRed('*** Syntax Error***')
                prRed(
                    """Please use:   'MarsPlot Custom.in -d XXXX [YYYY] -o out' """)
                exit()

        else: # If no date is provided, default to last 'fixed' file created in directory
            bound = get_Ncdf_num()
            # If one or multiple 'fixed' files are found, use last created
            if bound is not None:
                bound = bound[-1]
        # -----

        # Initialization
        Ncdf_num = get_Ncdf_num() # Get all timestamps in directory

        if Ncdf_num is not None:
            # Apply bounds to the desired dates
            Ncdf_num = select_range(Ncdf_num, bound)
            nfiles = len(Ncdf_num)  # number of timestamps
        else:  # If no 'fixed' file specified, assume we will be looking at one single file
            nfiles = 1

        #print('MarsPlot is running...')
        # Make a plots/ folder in the current directory if it does not exist
        dir_plot_present = os.path.exists(output_path+'/'+'plots')
        if not dir_plot_present:
            os.makedirs(output_path+'/'+'plots')

        fig_list = list() # List of figures

        # ============ Do plots ============
        global i_list
        for i_list in range(0, len(objectList)):

            status = objectList[i_list].plot_type + \
                ' :'+objectList[i_list].varfull
            # Display the status of the figure in progress
            progress(i_list, len(objectList), status, None)

            objectList[i_list].do_plot()

            if objectList[i_list].success and out_format == 'pdf' and not debug:
                sys.stdout.write("\033[F")
                # If successful, flush the previous output
                sys.stdout.write("\033[K")

            status = objectList[i_list].plot_type+' :' + \
                objectList[i_list].varfull+objectList[i_list].fdim_txt
            progress(i_list, len(objectList), status,
                     objectList[i_list].success)
            # Add the figure to the list of figures (fig_list)
            # Only for the last panel on a page
            if objectList[i_list].subID == objectList[i_list].nPan:
                if i_list < len(objectList)-1 and not objectList[i_list+1].addLine:
                    fig_list.append(objectList[i_list].fig_name)
                # Last subplot
                if i_list == len(objectList)-1:
                    fig_list.append(objectList[i_list].fig_name)

        progress(100, 100, 'Done')  # 100% complete

        # ============ Make Multipage PDF ============
        if out_format == "pdf" and len(fig_list) > 0:
            print('Merging figures...')
            #print("Plotting figures:",fig_list)
            # Debug file (masked). Use to redirect output from ghostscript
            debug_filename = output_path+'/.debug_MCMC_plots.txt'
            fdump = open(debug_filename, 'w')

            # Construct list of figures
            all_fig = ' '
            for figID in fig_list:
                # Add outer quotes(" ") to deal with whitespace in Windows, e.g. '"/Users/my folder/Diagnostics.pdf"'
                figID = '"'+figID+'"'
                all_fig += figID+' '

            # Output name for the PDF
            try:
                if parser.parse_args().do:
                    basename = parser.parse_args().do[0]
                else:
                    input_file = output_path+'/'+parser.parse_args().custom_file.name
                    # Get the input template file name, e.g. "Custom_01"
                    basename = input_file.split('/')[-1].split('.')[0].strip()

            except:
                # Special case where no Custom.in is provided
                basename = 'Custom'

            # Default name is Custom.in -> output Diagnostics.pdf
            if basename == 'Custom':
                output_pdf = fig_name = output_path+'/'+'Diagnostics.pdf'
            # Default name is Custom_XX.in -> output Diagnostics_XX.pdf
            elif basename[0:7] == "Custom_":
                output_pdf = fig_name = output_path+'/Diagnostics_' + \
                    basename[7:9]+'.pdf'  # Match input file name
            # Input file name is different, use it
            else:
                output_pdf = fig_name = output_path+'/' + \
                    basename+'.pdf'  # Match input file name

            # Also add outer quotes to the output PDF
            output_pdf = '"'+output_pdf+'"'
            # Command to make a multipage PDF out of the the individual figures using ghostscript.
            # Remove the temporary files when done
            cmd_txt = 'gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dEPSCrop -sOutputFile=' + \
                output_pdf+' '+all_fig

            # On NAS, the ghostscript has been renamed 'gs.bin'. If the above fail, try:
            try:
                subprocess.check_call(
                    cmd_txt, shell=True, stdout=fdump, stderr=fdump)
            except subprocess.CalledProcessError:
                cmd_txt = 'gs.bin -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER -dEPSCrop -sOutputFile=' + \
                    output_pdf+' '+all_fig
            # ================

            try:
                # Test the ghostscript and remove commands, exit otherwise
                subprocess.check_call(
                    cmd_txt, shell=True, stdout=fdump, stderr=fdump)
                # Execute the commands now
                # Run ghostscript to merge the PDF
                subprocess.call(cmd_txt, shell=True,
                                stdout=fdump, stderr=fdump)
                cmd_txt = 'rm -f '+all_fig
                # Remove temporary PDF figures
                subprocess.call(cmd_txt, shell=True,
                                stdout=fdump, stderr=fdump)
                cmd_txt = 'rm -f '+'"'+debug_filename+'"'
                subprocess.call(cmd_txt, shell=True)  # Remove debug file
                # If the plot directory was not present initially, remove the one we just created
                if not dir_plot_present:
                    cmd_txt = 'rm -r '+'"'+output_path+'"'+'/plots'
                    subprocess.call(cmd_txt, shell=True)
                give_permission(output_pdf)
                print(output_pdf + ' was generated')

            except subprocess.CalledProcessError:
                print(
                    "ERROR with ghostscript when merging PDF, please try alternative formats.")
                if debug:
                    raise

# ======================================================
#                  DATA OPERATION UTILITIES
# ======================================================

# USER PREFERENCES - AXIS FORMATTING

content_txt = section_content_amescap_profile('MarsPlot.py Settings')
exec(content_txt)  # Load all variables in that section

global add_sol_time_axis, lon_coord_type, include_NaNs

# Whether to include sol in addition to Ls on time axis (default = Ls only):
add_sol_time_axis = eval('np.array(add_sol_to_time_axis)')

# Defines which longitude coordinates to use (-180-180 v 0-360; default = 0-360):
lon_coord_type = eval('np.array(lon_coordinate)')

# Defines whether means include NaNs ('True', np.mean) or ignore NaNs ('False', like np.nanmean). Default = False:
include_NaNs = eval('np.array(show_NaN_in_slice)')


def mean_func(arr, axis):
    '''This function performs a mean over the selected axis, ignoring or including NaN values as specified by show_NaN_in_slice in amescap_profile'''
    if include_NaNs:
        return np.mean(arr, axis=axis)
    else:
        return np.nanmean(arr, axis=axis)

def shift_data(lon, data):
    '''
    This function shifts the longitude and data from 0/360 to -180/+180.
    Args:
        lon:  1D array of longitude (0/360)
        data: 2D array with last dimension = longitude
    Returns:
        lon:  1D array of longitude (-180/+180)
        data: shifted data
    Note: Use np.ma.hstack instead of np.hstack to keep the masked array properties.
    '''
    if lon_coord_type == 180:
        lon_180 = lon.copy()
        nlon = len(lon_180)
        # For 1D plots: If 1D, reshape array
        if len(data.shape) <= 1:
            data = data.reshape(1, nlon)

        lon_180[lon_180 > 180] -= 360.
        data = np.hstack((data[:, lon_180 < 0], data[:, lon_180 >= 0]))
        lon_180 = np.append(lon_180[lon_180 < 0], lon_180[lon_180 >= 0])
        # If 1D plot, squeeze array
        if data.shape[0] == 1:
            data = np.squeeze(data)
    elif lon_coord_type == 360:
        lon_180, data = lon, data
    else:
        raise ValueError('Longitude coordinate type invalid. Please specify "180" or "360" after lon_coordinate in amescap_profile.')
    return lon_180, data


def MY_func(Ls_cont):
    '''
    This function returns the Mars Year.
    Args:
        Ls_cont: solar longitude ('areo'), continuous
    Returns:
        MY: the Mars Year (integer)
    '''
    return (Ls_cont)//(360.)+1


def get_lon_index(lon_query_180, lons):
    '''
    This function returns the indices that will extract data from the netcdf file from a range of *longitudes*.
    Args:
        lon_query_180: longitudes in -180/+180: value, [min, max], or None
        lons:          1D array of longitude in 0/360
    Returns:
        loni:          1D array of file indices
        txt_lon:       text descriptor for the extracted longitudes
    *** Note that the keyword 'all' is passed as -99999 by the rT() functions
    '''
    Nlon = len(lons)
    lon_query_180 = np.array(lon_query_180)

    # If None, set to default (i.e.'all' for zonal average)
    if lon_query_180.any() == None:
        lon_query_180 = np.array(-99999)

    # ============== FV3 format ==============
    # If lon = 0/360, convert to -180/+180
    # ========================================
    if lons.max() > 180:
        # If one longitude is provided
        if lon_query_180.size == 1:
            # Request zonal average
            if lon_query_180 == -99999:
                loni = np.arange(0, Nlon)
                txt_lon = ', zonal avg'
            else:
                # Get closest value
                lon_query_360 = lon180_to_360(lon_query_180)
                loni = np.argmin(np.abs(lon_query_360-lons))
                txt_lon = ', lon=%.1f' % (lon360_to_180(lons[loni]))
        # If a range of longitudes is provided
        elif lon_query_180.size == 2:
            lon_query_360 = lon180_to_360(lon_query_180)
            loni_bounds = np.array([np.argmin(
                np.abs(lon_query_360[0]-lons)), np.argmin(np.abs(lon_query_360[1]-lons))])
            # if loni_bounds[0] > loni_bounds[1]: loni_bounds = np.flipud(loni_bounds) # lon should be increasing for extraction # TODO
            # Normal case (e.g. -45W>45E)
            if loni_bounds[0] < loni_bounds[1]:
                loni = np.arange(loni_bounds[0], loni_bounds[1]+1)
            else:
                # Loop around (e.g. 160E>-40W)
                loni = np.append(np.arange(loni_bounds[0], len(
                    lons)), np.arange(0, loni_bounds[1]+1))
                prPurple(lon360_to_180(lons[loni]))
            lon_bounds_180 = lon360_to_180(
                [lons[loni_bounds[0]], lons[loni_bounds[1]]])

            # if lon_bounds_180[0] > lon_bounds_180[1]: lon_bounds_180 = np.flipud(lon_bounds_180) # lon should be also increasing for display
            txt_lon = ', lon=avg[%.1f<->%.1f]' % (
                lon_bounds_180[0], lon_bounds_180[1])

        # =========== Legacy Format ===========
        # Lon = -180/+180
        # ===================================
    else:
        # If one longitude is provided
        if lon_query_180.size == 1:
            # request zonal average
            if lon_query_180 == -99999:
                loni = np.arange(0, Nlon)
                txt_lon = ', zonal avg'
            else:
                # Get closest value
                loni = np.argmin(np.abs(lon_query_180-lons))
                txt_lon = ', lon=%.1f' % (lons[loni])
        # If a range of longitudes is provided
        elif lon_query_180.size == 2:
            loni_bounds = np.array([np.argmin(
                np.abs(lon_query_180[0]-lons)), np.argmin(np.abs(lon_query_180[1]-lons))])
            # Normal case (e.g. -45W>45E)
            if loni_bounds[0] < loni_bounds[1]:
                loni = np.arange(loni_bounds[0], loni_bounds[1]+1)
            else:
                # Loop around (e.g. 160E>-40W)
                loni = np.append(np.arange(loni_bounds[0], len(
                    lons)), np.arange(0, loni_bounds[1]+1))
            txt_lon = ', lon=avg[%.1f<->%.1f]' % (
                lons[loni_bounds[0]], lons[loni_bounds[1]])

    return loni, txt_lon


def get_lat_index(lat_query, lats):
    '''
    This function returns the indices that will extract data from the netcdf file from a range of *latitudes*.
    Args:
        lat_query: requested latitudes (-90/+90)
        lats:      1D array of latitudes
    Returns:
        lati:      1D array of file indices
        txt_lat:   text descriptor for the extracted latitudes
    *** Note that the keyword 'all' is passed as -99999 by the rT() functions
    '''
    Nlat = len(lats)
    lat_query = np.array(lat_query)
    # If None, set to default (i.e.equator)
    if lat_query.any() == None:
        lat_query = np.array(0.)
    # If one latitude is provided
    if lat_query.size == 1:
        # Request meridional average
        if lat_query == -99999:
            lati = np.arange(0, Nlat)
            txt_lat = ', merid. avg'
        else:
            # Get closest value
            lati = np.argmin(np.abs(lat_query-lats))
            txt_lat = ', lat=%g' % (lats[lati])
    # If a range of latitudes are provided
    elif lat_query.size == 2:
        lat_bounds = np.array(
            [np.argmin(np.abs(lat_query[0]-lats)), np.argmin(np.abs(lat_query[1]-lats))])
        if lat_bounds[0] > lat_bounds[1]:
            # Latitude should be increasing for extraction
            lat_bounds = np.flipud(lat_bounds)
        lati = np.arange(lat_bounds[0], lat_bounds[1]+1)
        txt_lat = ', lat=avg[%g<->%g]' % (lats[lati[0]], lats[lati[-1]])
    return lati, txt_lat


def get_tod_index(tod_query, tods):
    '''
    This function returns the indices that will extract data from the netcdf file from a range of *times of day*.
    Args:
        tod_query: requested time of day (0-24)
        tods:      1D array of times of day
    Returns:
        todi:      1D array of file indices
        txt_tod:   text descriptor for the extracted time of day
    *** Note that the keyword 'all' is passed as -99999 by the rT() functions
    '''
    Ntod = len(tods)
    tod_query = np.array(tod_query)
    # If None, set to default (3pm)
    if tod_query.any() == None:
        tod_query = np.array(15)
    # If one time of day is provided
    if tod_query.size == 1:
        # Request diurnal average
        if tod_query == -99999:
            todi = np.arange(0, Ntod)
            txt_tod = ', tod avg'
        else:
            # Get closest value
            todi = np.argmin(np.abs(tod_query-tods))
            txt_tmp = UT_LTtxt(tods[todi]/24., lon_180=0., roundmin=1)
            txt_tod = ', tod= %s' % (txt_tmp)
    # If a range of times of day are provided
    elif tod_query.size == 2:
        tod_bounds = np.array(
            [np.argmin(np.abs(tod_query[0]-tods)), np.argmin(np.abs(tod_query[1]-tods))])
        # Normal case (e.g. 4am>10am)
        if tod_bounds[0] < tod_bounds[1]:
            todi = np.arange(tod_bounds[0], tod_bounds[1]+1)
        else:
            # Loop around (e.g. 18pm>6am)
            todi = np.append(np.arange(tod_bounds[0], len(
                tods)), np.arange(0, tod_bounds[1]+1))
        txt_tmp = UT_LTtxt(tods[todi[0]]/24., lon_180=0., roundmin=1)
        txt_tmp2 = UT_LTtxt(tods[todi[-1]]/24., lon_180=0., roundmin=1)
        txt_tod = ', tod=avg[%s<->%s]' % (txt_tmp, txt_tmp2)
    return todi, txt_tod


def get_level_index(level_query, levs):
    '''
    This function returns the indices that will extract data from the netcdf file from a range of *pressures* (resp. depth for 'zgrid').
    Args:
        level_query: requested  pressure [Pa] (depth [m])
        levs:        1D array of levels in the native coordinates [Pa] ([m])
    Returns:
        levi:        1D array of file indices
        txt_lev:     text descriptor for the extracted pressure (depth)
    *** Note that the keyword 'all' is passed as -99999 by the rT() functions
    '''
    level_query = np.array(level_query)
    Nz = len(levs)
    # If None, set to default (surface)
    if level_query.any() == None:
        # If provided a big number > Psfc (even for a 10-bar Early Mars simulation)
        level_query = np.array(2*10**7)

    # If one level is provided
    if level_query.size == 1:
        # Average
        if level_query == -99999:
            levi = np.arange(0, Nz)
            txt_level = ', column avg'
        # Specific level
        else:
            levi = np.argmin(np.abs(level_query-levs))

        # Provide smart labeling
            if level_query > 10.**7:  # None (i.e.surface was requested)
                txt_level = ', at sfc'
            else:
                #txt_level=', lev=%g Pa'%(levs[levi])
                txt_level = ', lev={0:1.2e} Pa/m'.format(levs[levi])

    elif level_query.size == 2:  # Bounds are provided
        levi_bounds = np.array(
            [np.argmin(np.abs(level_query[0]-levs)), np.argmin(np.abs(level_query[1]-levs))])
        if levi_bounds[0] > levi_bounds[1]:
            # Level should be increasing for extraction
            levi_bounds = np.flipud(levi_bounds)
        levi = np.arange(levi_bounds[0], levi_bounds[1]+1)
        lev_bounds = [levs[levi[0]], levs[levi[-1]]]  # This is for display
        if lev_bounds[0] < lev_bounds[1]:
            # Level should be decreasing for display
            lev_bounds = np.flipud(lev_bounds)
        txt_level = ', lev=avg[{0:1.2e}<->{1:1.2e}] Pa/m'.format(
            lev_bounds[0], lev_bounds[1])

    return levi, txt_level


def get_time_index(Ls_query_360, LsDay):
    '''
    This function returns the indices that will extract data from the netcdf file from a range of solar longitudes [0-360].
    First try the Mars Year of the last timestep, then try the year before that. Use whichever Ls period is closest to the requested date.

    Args:
        Ls_query_360: requested solar longitudes
        Ls_c:         1D array of continuous solar longitudes
    Returns:
        ti:           1D array of file indices
        txt_time:     text descriptor for the extracted solar longitudes
    *** Note that the keyword 'all' is passed as -99999 by the rT() functions
    '''

    # Special case where the file has only one timestep, transform LsDay to array:
    if len(np.atleast_1d(LsDay)) == 1:
        LsDay = np.array([LsDay])

    Nt = len(LsDay)
    Ls_query_360 = np.array(Ls_query_360)

    # If None, set to default (i.e.last timestep)
    if Ls_query_360.any() == None:
        Ls_query_360 = np.mod(LsDay[-1], 360.)

    # If one time is provided
    if Ls_query_360.size == 1:
        # Time average average requested
        if Ls_query_360 == -99999:
            ti = np.arange(0, Nt)
            txt_time = ', time avg'
        else:
            # Get the Mars Year of the last timestep in the file
            MY_end = MY_func(LsDay[-1])
            if MY_end >= 1:
                # Check if the desired Ls is available in this Mars Year
                Ls_query = Ls_query_360+(MY_end-1) * \
                    360.  # (MY starts at 1, not zero)
            else:
                Ls_query = Ls_query_360
            # If this time is greater than the last Ls, look one year back
            if Ls_query > LsDay[-1] and MY_end > 1:
                MY_end -= 1  # One year back
                Ls_query = Ls_query_360+(MY_end-1)*360.
            ti = np.argmin(np.abs(Ls_query-LsDay))
            txt_time = ', Ls= (MY%2i) %.2f' % (MY_end, np.mod(LsDay[ti], 360.))

    # If a range of times are provided
    elif Ls_query_360.size == 2:

        # Get the Mars Year of the last timestep in the file
        MY_last = MY_func(LsDay[-1])
        if MY_last >= 1:
            # Try the Mars Year of the last timestep
            Ls_query_last = Ls_query_360[1]+(MY_last-1)*360.
        else:
            Ls_query_last = Ls_query_360[1]
        # First consider the further end of the desired range
        # If this time is greater that the last Ls, look one year back
        if Ls_query_last > LsDay[-1] and MY_last > 1:
            MY_last -= 1
            Ls_query_last = Ls_query_360[1] + \
                (MY_last-1)*360.  # (MY starts at 1, not zero)
        ti_last = np.argmin(np.abs(Ls_query_last-LsDay))
        # Then get the first value for that Mars Year
        MY_beg = MY_last.copy()
        # Try the Mars Year of the last timestep
        Ls_query_beg = Ls_query_360[0]+(MY_beg-1)*360.
        ti_beg = np.argmin(np.abs(Ls_query_beg-LsDay))

        # If the start value is higher, search in the year before for ti_beg
        if ti_beg >= ti_last:
            MY_beg -= 1
            Ls_query_beg = Ls_query_360[0]+(MY_beg-1)*360.
            ti_beg = np.argmin(np.abs(Ls_query_beg-LsDay))

        ti = np.arange(ti_beg, ti_last+1)

        Ls_bounds = [LsDay[ti[0]], LsDay[ti[-1]]]  # This is for display
        txt_time = ', Ls= avg [(MY%2i) %.2f <-> (MY%2i) %.2f]' % (MY_beg,
                                                                  np.mod(Ls_bounds[0], 360.), MY_last, np.mod(Ls_bounds[1], 360.))

    return ti, txt_time

# ======================================================
#                  TEMPLATE UTILITIES
# ======================================================

def filter_input(txt, typeIn='char'):
    '''
    Read template for the type of data expected.
    Args:
        txt:    string, typically from the right side of an equal sign in template '3', '3,4', or 'all'
        typeIn: type of data expected: 'char', 'float', 'int', 'bool'
    Returns:
        out:    float or 1D array [val1, val2] in the expected format

    '''
    # If None or empty string
    if txt == 'None' or not txt:
        return None

    # If two values are provided
    if "," in txt:
        answ = []
        for i in range(0, len(txt.split(','))):
            # For a 'char', read all text as one
            #if typeIn=='char': answ.append(txt.split(',')[i].strip())
            if typeIn == 'char':
                answ = txt
            if typeIn == 'float':
                answ.append(float(txt.split(',')[i].strip()))
            if typeIn == 'int':
                answ.append(int(txt.split(',')[i].strip()))
            if typeIn == 'bool':
                answ.append(txt.split(',')[i].strip() == 'True')
        return answ
    else:
        if typeIn == 'char':
            answ = txt
        if typeIn == 'bool':
            answ = ('True' == txt)
        # For 'float' and 'int', pass the 'all' key word as -99999
        if typeIn == 'float':
            if txt == 'all':
                answ = -99999.
            elif txt == 'AXIS':
                answ = -88888.
            else:
                answ = float(txt)
        if typeIn == 'int':
            if txt == 'all':
                answ = -99999
            else:
                answ = int(txt) # True if text matches
        return answ


def rT(typeIn='char'):
    '''
    Read template for the type of data expected.
    Args:
        typeIn: type of data expected: 'char', 'float', 'int', 'bool'
    Returns:
        out:    float or 1D array [val1, val2] in the expected format

    '''
    global customFileIN
    raw_input = customFileIN.readline()

    # Get text on the right side of the equal sign IF there is only one
    # equal sign in string (e.g. '02400.atmos_average2{lat=20}')
    if len(raw_input.split('=')) == 2:
        txt = raw_input.split('=')[1].strip()

    # Read the string manually if there is more than one equal sign
    # (e.g. '02400.atmos_average2{lat=20,tod=4}')
    elif len(raw_input.split('=')) > 2:
        current_varfull = ''
        record = False
        for i in range(0, len(raw_input)):
            if record:
                current_varfull += raw_input[i]
            if raw_input[i] == '=':
                record = True
        txt = current_varfull.strip()

    return filter_input(txt, typeIn)


def read_axis_options(axis_options_txt):
    '''
    Return axis customization options.
    Args:
        axis_options_txt: One line string = 'Axis Options  : lon = [5,8] | lat = [None,None] | cmap = jet | scale= lin | proj = cart'
    Returns:
        Xaxis:          X-axis bounds as a numpy array or None if undedefined
        Yaxis:          Y-axis bounds as a numpy array or None if undedefined
        custom_line1:   string, colormap (e.g. 'jet', 'nipy_spectral') or line options (e.g. '--r' for dashed red)
        custom_line2:   linear (lin) or logarithmic (log) color scale
        custom_line3:   string, projection (e.g. 'ortho -125,45')
    '''
    list_txt = axis_options_txt.split(':')[1].split('|')
    # Xaxis: get bounds
    txt = list_txt[0].split('=')[1].replace('[', '').replace(']', '')
    Xaxis = []
    for i in range(0, len(txt.split(','))):
        if txt.split(',')[i].strip() == 'None':
            Xaxis = None
            break
        else:
            Xaxis.append(float(txt.split(',')[i].strip()))
    # Yaxis: get bounds
    txt = list_txt[1].split('=')[1].replace('[', '').replace(']', '')
    Yaxis = []
    for i in range(0, len(txt.split(','))):
        if txt.split(',')[i].strip() == 'None':
            Yaxis = None
            break
        else:
            Yaxis.append(float(txt.split(',')[i].strip()))
    # Line or colormap
    custom_line1 = list_txt[2].split('=')[1].strip()
    custom_line2 = None
    custom_line3 = None
    # Scale: lin or log (2D plots only)

    if len(list_txt) == 4:
        custom_line2 = list_txt[3].split('=')[1].strip()
        if custom_line2.strip() == 'None':
            custom_line2 = None
    if len(list_txt) == 5:
        custom_line2 = list_txt[3].split('=')[1].strip()
        custom_line3 = list_txt[4].split('=')[1].strip()
        if custom_line2.strip() == 'None':
            custom_line2 = None
        if custom_line3.strip() == 'None':
            custom_line3 = None
    return Xaxis, Yaxis, custom_line1, custom_line2, custom_line3


def split_varfull(varfull):
    '''
    Split the 'varfull' object into its component parts.
    Args:
        varfull:    a 'varfull' object (e.g. 'atmos_average@2.zsurf', '02400.atmos_average@2.zsurf')
    Returns:
        sol_array: a sol number (e.g. 2400) or None (if none is provided)
        filetype:  file type (i.e. 'atmos_average')
        var:       variable of interest (i.e. 'zsurf')
        simuID:    integer, simulation ID (Python indices start at zero so ID = 2 -> 1)
    '''

    # Default case: no sol number provided (e.g. 'atmos_average2.zsurf')
    # Extract variables and file from varfull

    if varfull.count('.') == 1:
        sol_array = np.array([None])
        filetypeID = varfull.split('.')[0].strip()  # File and ID
        var = varfull.split('.')[1].strip()         # Variable name

    # Case 2: sol number is provided (e.g. '02400.atmos_average2.zsurf'
    elif varfull.count('.') == 2:
        sol_array = np.array(
            [int(varfull.split('.')[0].strip())])   # Sol number
        filetypeID = varfull.split('.')[1].strip()  # File and ID
        var = varfull.split('.')[2].strip()         # Variable name
    # Split filename and simulation ID

    if '@' in filetypeID:
        filetype = filetypeID.split('@')[0].strip()
        # Simulation ID starts at zero in the code
        simuID = int(filetypeID.split('@')[1].strip())-1
    else:
        # No digit (i.e. reference simulation)
        simuID = 0
        filetype = filetypeID
    return sol_array, filetype, var, simuID


def remove_whitespace(raw_input):
    '''
    Remove whitespace inside an expression. This is different from the '.strip()' method,
    which only removes whitespaces at the edges of a string.
    Args:
        raw_input:          a string, e.g. '[atmos_average.temp] +  2'
    Returns:
        processed_input:    the string without whitespaces, e.g. [atmos_average.temp] + 2'
    '''
    processed_input = ''
    for i in range(0, len(raw_input)):
        if raw_input[i] != ' ':
            processed_input += raw_input[i]
    return processed_input


def clean_comma_whitespace(raw_input):
    '''
    Remove the commas and whitespaces inside an expression.
    Args:
        raw_input:          a string (e.g. 'lat=3. , lon=2,lev=10.')
    Returns:
        processed_input:    the string without whitespaces or commas (e.g. 'lat=3.lon=2lev=10.')
    '''
    processed_input = ''
    for i in range(0, len(raw_input)):
        if raw_input[i] != ',':
            processed_input += raw_input[i]
    return remove_whitespace(processed_input)


def get_list_varfull(raw_input):
    '''
    Given an expression object with '[]' return the variable needed.
    Args:
        raw_input:  a complex 'varfull' object (e.g. '2*[atmos_average.temp]+[atmos_average2.ucomp]*1000')
    Returns:
        var_list:   a list of variables to load (e.g. ['atmos_average.temp', 'atmos_average2.ucomp'])
    '''
    var_list = []
    record = False
    current_name = ''
    for i in range(0, len(raw_input)):
        if raw_input[i] == ']':
            record = False
            var_list.append(current_name.strip())
            current_name = ''
        if record:
            current_name += raw_input[i]
        if raw_input[i] == '[':
            record = True
    return var_list


def get_overwrite_dim_2D(varfull_bracket, plot_type, fdim1, fdim2, ftod):
    '''
    Given a single 'varfull' object with '{}', return the new dimensions that will overwrite the default dimensions.
    Args:
        varfull_bracket:    a 'varfull' object with any of the following:
                            atmos_average.temp{lev=10;ls=350;lon=155;lat=25}
                            (brackets and semi-colons separated)
        plot_type:          the type of plot

    Returns:
        varfull:            the 'varfull' without brackets (e.g. 'atmos_average.temp')
        fdim_out1,
        fdim_out1,
        ftod_out:           the dimensions to update

    2D_lon_lat:  fdim1 = ls
                 fdim2 = lev

    2D_lat_lev:  fdim1 = ls
                 fdim2 = lon

    2D_time_lat: fdim1 = lon
                 fdim2 = lev

    2D_lon_lev:  fdim1 = ls
                 fdim2 = lat

    2D_time_lev: fdim1 = lat
                 fdim2 = lon

    2D_lon_time: fdim1 = lat
                 fdim2 = lev
    '''

    # Initialization: use the dimension provided in the template
    fdim_out1 = fdim1
    fdim_out2 = fdim2
    # Left of the '{' character:
    varfull_no_bracket = varfull_bracket.split(
        '{')[0].strip()
    # Right of the'{' character, with the last '}' removed:
    overwrite_txt = remove_whitespace(varfull_bracket.split('{')[1][:-1])
    # Count the number of equal signs in the string
    ndim_update = overwrite_txt.count('=')
    # Split to different blocks (e.g. 'lat = 3.' and 'lon = 20')
    split_dim = overwrite_txt.split(';')
    if overwrite_txt.count(';') < overwrite_txt.count('=')-1:
        prYellow("""*** Error: use semicolon ';' to separate dimensions '{}'""")
    for i in range(0, ndim_update):
        # Check if the requested dimension exists:
        if split_dim[i].split('=')[0] not in ['ls', 'lev', 'lon', 'lat', 'tod']:
            prYellow("""*** Warning*** Ignoring dimension: '"""+split_dim[i].split('=')[
                     0]+"""' because it is not recognized. Valid dimensions = 'ls','lev','lon', 'lat' or 'tod'""")

        if plot_type == '2D_lon_lat':
            if split_dim[i].split('=')[0] == 'ls':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lev':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')
        if plot_type == '2D_lat_lev':
            if split_dim[i].split('=')[0] == 'ls':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lon':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')
        if plot_type == '2D_time_lat':
            if split_dim[i].split('=')[0] == 'lon':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lev':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')
        if plot_type == '2D_lon_lev':
            if split_dim[i].split('=')[0] == 'ls':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lat':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')
        if plot_type == '2D_time_lev':
            if split_dim[i].split('=')[0] == 'lat':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lon':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')
        if plot_type == '2D_lon_time':
            if split_dim[i].split('=')[0] == 'lat':
                fdim_out1 = filter_input(split_dim[i].split('=')[1], 'float')
            if split_dim[i].split('=')[0] == 'lev':
                fdim_out2 = filter_input(split_dim[i].split('=')[1], 'float')

        # Always get time of day
        ftod_out = None
        if split_dim[i].split('=')[0] == 'tod':
            ftod_out = filter_input(split_dim[i].split('=')[1], 'float')
    # NOTE: filter_input() converts the text (e.g. '3' or '4,5') to a real variable
    # (e.g. numpy.array([3.]) or numpy.array([4.,5.]))
    return varfull_no_bracket, fdim_out1, fdim_out2, ftod_out


def get_overwrite_dim_1D(varfull_bracket, t_in, lat_in, lon_in, lev_in, ftod_in):
    '''
    Given a single 'varfull' object with '{}', return the new dimensions that will overwrite the default dimensions
    Args:
        varfull_bracket:    a 'varfull' object with any of the following:
                            atmos_average.temp{lev=10;ls=350;lon=155;lat=25;tod=15}
        t_in, lat_in,
        lon_in, lev_in,
        ftod_in:            the variables as defined by self.t, self.lat, self.lon, self.lev, self.ftod

    Returns:
        'varfull' the 'varfull' without brackets: e.g. 'atmos_average.temp'
        t_out,lat_out,lon_out,lev_out,ftod_out: the dimensions to update
    '''
    # Initialization: Use the dimension provided in the template
    t_out = t_in
    lat_out = lat_in
    lon_out = lon_in
    lev_out = lev_in
    # Left of the '{' character:
    varfull_no_bracket = varfull_bracket.split(
        '{')[0].strip()
    # Right of the'{' character, with the last '}' removed:
    overwrite_txt = remove_whitespace(varfull_bracket.split('{')[1][:-1])
    # Count the number of equal signs in the string
    ndim_update = overwrite_txt.count('=')
    # Split to different blocks (e.g. 'lat = 3.' and 'lon = 20')
    split_dim = overwrite_txt.split(';')
    for i in range(0, ndim_update):
        # Check if the requested dimension exists:
        if split_dim[i].split('=')[0] not in ['time', 'lev', 'lon', 'lat', 'tod']:
            prYellow("""*** Warning*** ignoring dimension: '"""+split_dim[i].split('=')[
                     0]+"""' because it is not recognized. Valid dimensions = 'time','lev','lon', 'lat' or 'tod'""")

        if split_dim[i].split('=')[0] == 'ls':
            t_out = filter_input(split_dim[i].split('=')[1], 'float')
        if split_dim[i].split('=')[0] == 'lat':
            lat_out = filter_input(split_dim[i].split('=')[1], 'float')
        if split_dim[i].split('=')[0] == 'lon':
            lon_out = filter_input(split_dim[i].split('=')[1], 'float')
        if split_dim[i].split('=')[0] == 'lev':
            lev_out = filter_input(split_dim[i].split('=')[1], 'float')

        # Always get time of day
        ftod_out = None
        if split_dim[i].split('=')[0] == 'tod':
            ftod_out = filter_input(split_dim[i].split('=')[1], 'float')
    # NOTE: filter_input() converts the text (e.g. '3' or '4,5') to a real variable
    # (e.g. numpy.array([3.]) or numpy.array([4.,5.]))

    return varfull_no_bracket, t_out, lat_out, lon_out, lev_out, ftod_out


def create_exec(raw_input, varfull_list):
    expression_exec = raw_input
    for i in range(0, len(varfull_list)):
        swap_txt = '['+varfull_list[i]+']'
        expression_exec = expression_exec.replace(swap_txt, 'VAR[%i]' % (i))
    return expression_exec


def fig_layout(subID, nPan, vertical_page=False):
    '''
    Returns figure layout.
    Args:
        subID:          integer, current subplot number
        nPan:           integer, number of panels desired on page (max = 64, 8x8)
        vertical_page:  if True, reverse the tuple for portrait format
    Returns:
        out:            tuple, layout: plt.subplot(nrows = out[0], ncols = out[1], plot_number = out[2])
    '''
    out = list((0, 0, 0))  # Initialization

    if nPan == 1:
        layout = (1, 1)  # nrow, ncol
    if nPan == 2:
        layout = (1, 2)
    if nPan == 3 or nPan == 4:
        layout = (2, 2)
    if nPan == 5 or nPan == 6:
        layout = (2, 3)
    if nPan == 7 or nPan == 8:
        layout = (2, 4)
    if nPan == 9:
        layout = (3, 3)
    if 10 <= nPan <= 12:
        layout = (3, 4)
    if 13 <= nPan <= 16:
        layout = (4, 4)
    if 17 <= nPan <= 20:
        layout = (4, 5)
    if 21 <= nPan <= 25:
        layout = (5, 5)
    if 26 <= nPan <= 30:
        layout = (5, 6)
    if 30 <= nPan <= 36:
        layout = (6, 6)
    if 36 <= nPan <= 42:
        layout = (7, 6)
    if 42 <= nPan <= 49:
        layout = (7, 7)
    if 49 <= nPan <= 56:
        layout = (8, 7)
    if 56 <= nPan <= 64:
        layout = (8, 8)
    if vertical_page:
        layout = layout[::-1]

    # Finally the current plot
    out[0:2] = layout
    out[2] = subID

    return out


def make_template():
    global customFileIN  # Will be modified
    global current_version
    newname = output_path+'/Custom.in'
    newname = create_name(newname)

    customFileIN = open(newname, 'w')

    lh = """# """  # Add a line header. Primary use is to change the text color in vim

    # Create header with instructions. Add the version number to the title.
    customFileIN.write(
        "===================== |MarsPlot V%s| ===================\n" % (current_version))
    if parser.parse_args().template:  # Additional instructions if requested
        customFileIN.write(
            lh+"""================================================= INSTRUCTIONS =================================================\n""")
        customFileIN.write(lh+"""- Copy/paste template for the desired plot type. - Do not modify text left of an equal '=' sign. \n""")
        customFileIN.write(lh+"""- Add comments using '#'                         - Skip plots by setting <<<< Plot = False >>>> \n""")
        customFileIN.write(lh+"""- Capitalize 'True', 'False', and 'None'.        - Do not use quotes ('') anywhere in this file. \n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""Group figures onto pages using'HOLD ON' and 'HOLD OFF'. \n""")
        customFileIN.write(lh+"""Optionally, use 'row,col' to specify the layout: HOLD ON 2,3'. \n""")
        customFileIN.write(lh+"""Use 'ADD LINE' between 1D plots to overplot on the same figure. \n""")
        customFileIN.write(lh+"""Figures templates must appear after 'START' and before 'STOP'. \n""")
        customFileIN.write(lh+"""Set the colorbar range with 'Cmin, Cmax'. Scientific notation (e.g. 1e-6, 2e3) is supported. \n""")
        customFileIN.write(lh+"""Set the colorbar intervals directly by providing a list (e.g. 1e-6, 1e-4, 1e-2, 1e-0). \n""")
        customFileIN.write(lh+"""Set the contour intervals for '2nd Variable' in a list (e.g. 150, 200, 250, 300, 350). \n""")
        customFileIN.write(lh+"""The vertical grid of the *.nc file used in the plot determines what 'Level' refers to.\n""")
        customFileIN.write(lh+"""   'Level' can be: 'level', 'pfull', 'pstd', 'plevs' [Pa] or 'zstd', 'zagl', or 'zgrid' [m].\n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""============================================ ALGEBRA ============================================\n""")
        customFileIN.write(lh+"""Use square brackets '[]' for element-wise operations: \n""")
        customFileIN.write(lh+"""   '[fixed.zsurf]/(10.**3)'            Convert between units ([m] to [km], in this case).\n""")
        customFileIN.write(lh+"""   '[file.var1]/[file.var2]*610'       Multiply variables together.\n""")
        customFileIN.write(lh+"""   '[file.var]-[file@2.var]'           Difference plot of 'var' from 2 simulations.\n""")
        customFileIN.write(lh+"""   '[file.var]-[file.var{lev=10}]'     Difference plot of 'var' at two levels.\n""")
        customFileIN.write(lh+"""Square brackets support the following expressions: sqrt, log, exp, abs, min, max, & mean.\n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""========================================= FREE DIMENSIONS =========================================\n""")
        customFileIN.write(lh+"""Dimensions can be 'time', 'lev', 'lat', 'lon', or 'tod'.\n""")
        customFileIN.write(lh+"""Dimensions default to None when a value or range is not specified. None corresponds to: \n""")
        customFileIN.write(lh+"""   time  =  -1      The last (most recent) timestep (Nt).\n""")
        customFileIN.write(lh+"""   lev   =  sfc     Nz for *.nc files, 0 for *_pstd.nc files.\n""")
        customFileIN.write(lh+"""   lat   =  0       Equator\n""")
        customFileIN.write(lh+"""   lon   =  'all'   Zonal average over all longitudes\n""")
        customFileIN.write(lh+"""   tod   =  '15'    3 PM UT \n""")
        customFileIN.write(lh+"""Setting a dimension equal to a number finds the value closest to that number. \n""")
        customFileIN.write(lh+"""Setting a dimension equal to 'all' averages the dimension over all values. \n""")
        customFileIN.write(lh+"""Setting a dimension equal to a range averages the dimension over the values in the range. \n""")
        customFileIN.write(lh+"""You can also overwrite a dimension in the Main Variable input using curvy brackets '{}' and the\n""")
        customFileIN.write(lh+"""   dimension name. Separate the arguments with semi-colons ';' \n""")
        customFileIN.write(lh+"""       e.g. Main Variable  = atmos_average.temp{ls = 90; lev= 5.,10; lon= all; lat=45} \n""")
        customFileIN.write(lh+"""   Values must correspond to the units of the variable in the file: \n""")
        customFileIN.write(lh+"""       time [Ls], lev [Pa/m], lon [+/-180 deg], and lat [deg]. \n""")
        customFileIN.write(lh+"""* You can only select a time of day (tod) in diurn files using this syntax: \n""")
        customFileIN.write(lh+"""       e.g. Main Variable  = atmos_diurn.ps{tod = 20} \n""")
        customFileIN.write(lh+"""You can also specify the fontsize in Title using curvy brackets and 'size':\n""")
        customFileIN.write(lh+"""       e.g. Title = Temperature [K] {size = 20}.\n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""==================================== TIME SERIES AND 1D PLOTS ====================================\n""")
        customFileIN.write(lh+"""Set the X axis variable by indicating AXIS after the appropriate dimension: \n""")
        customFileIN.write(lh+"""       e.g. Ls = AXIS \n""")
        customFileIN.write(lh+"""The other dimensions remain FREE DIMENSIONS and accept values as described above. \n""")
        customFileIN.write(lh+"""The 'Diurnal [hr]' dimension only accepts 'AXIS' or 'None'. Indicate time of day only using the'\n""")
        customFileIN.write(lh+"""   'tod' syntax as described in FREE DIMENSIONS. \n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""================================== AXIS OPTIONS AND PROJECTIONS ==================================\n""")
        customFileIN.write(lh+"""Set the X and Y axis limits, map projection, colormap, and linestyle under Axis Options. \n""")
        customFileIN.write(lh+"""All Matplolib styles are supported. \n""")
        customFileIN.write(lh+"""   'cmap'  colormap    'jet' (winds), 'nipy_spectral' (temperature), 'bwr' (diff plot), etc. \n""")
        customFileIN.write(lh+"""   'scale' gradient    'lin' (linear), 'log' (logarithmic; Cmin, Cmax is typically expected. \n""")
        customFileIN.write(lh+"""   'line'  linestyle   '-r' (solid red), '--g' (dashed green), '-ob' (solid blue + markers). \n""")
        customFileIN.write(lh+"""   'proj'  projection  Cylindrical: 'cart' (Cartesian), 'robin' (Robinson), 'moll' (Mollweide), \n""")
        customFileIN.write(lh+"""                       Azithumal: 'Npole lat' (North Pole), 'Spole lat' (South Pole),\n""")
        customFileIN.write(lh+"""                       'ortho lon,lat' (Orthographic). \n""")
        customFileIN.write(lh+"""\n""")
        customFileIN.write(lh+"""===================== FILES FROM MULTIPLE SIMULATIONS =====================\n""")
        customFileIN.write(lh+"""Under <<< Simulations >>>, there are numbered lines ('N>') for you to use to indicate the \n""")
        customFileIN.write(lh+"""   path to the *.nc file you want to reference. Empty fields are ignored. \n""")
        customFileIN.write(lh+"""Provide the FULL PATH on the line, e.g. '2> /u/User/FV3/path/to/history'. \n""")
        customFileIN.write(lh+"""Specify the *.nc file from which to plot using the '@' symbol + the simulation number:\n""")
        customFileIN.write(lh+"""   in the call to Main Variable, e.g. Main Variable = atmos_average@2.temp \n""")
        customFileIN.write(lh+"""\n""")
    customFileIN.write(
        "<<<<<<<<<<<<<<<<<<<<<< Simulations >>>>>>>>>>>>>>>>>>>>>\n")
    customFileIN.write("ref> None\n")
    customFileIN.write("2> \n")
    customFileIN.write("3>\n")
    customFileIN.write(
        "=======================================================\n")
    customFileIN.write("START\n")
    customFileIN.write("\n")  # new line
    # ===============================================================

    # For the default list of figures in main(), create a template.
    for i in range(0, len(objectList)):
        if objectList[i].subID == 1 and objectList[i].nPan > 1:
            customFileIN.write('HOLD ON\n')
        objectList[i].make_template()
        customFileIN.write('\n')
        if objectList[i].nPan > 1 and objectList[i].subID == objectList[i].nPan:
            customFileIN.write('HOLD OFF\n')

        # Separate the empty templates
        if i == 1:
            customFileIN.write("""#=========================================================================\n""")
            customFileIN.write("""#================== Empty Templates (set to False)========================\n""")
            customFileIN.write("""#========================================================================= \n""")
            customFileIN.write(""" \n""")
    customFileIN.close()

    # NAS system only: set group permissions to the file and print a completion message
    give_permission(newname)
    print(newname + ' was created ')

def give_permission(filename):
    # NAS system only: set group permissions to the file
    try:
        subprocess.check_call(['setfacl -v'], shell=True, stdout=open(os.devnull, "w"),
                              stderr=open(os.devnull, "w"))  # Catch error and standard output
        cmd_txt = 'setfacl -R -m g:s0846:r '+filename
        subprocess.call(cmd_txt, shell=True)
    except subprocess.CalledProcessError:
        pass


def namelist_parser(Custom_file):
    '''
    Parse a template.
    Args:
        Custom_file: full path to Custom.in file
    Actions:
        Update global variable, FigLayout, objectList
    '''
    global objectList
    global customFileIN
    global input_paths
    # A Custom.in file is provided, flush the default figures in main()

    objectList  = []  # All individual plots
    panelList   = []  # List of panels
    subplotList = []  # Layout of figures
    addLineList = []  # Add several lines to plot on the same graph
    layoutList  = []
    # Number for the object (e.g. 1,[2,3],4... would have 2 & 3 plotted in a two-panel plot)
    nobj        = 0
    npanel      = 1  # Number of panels plotted along this object (e.g: '1' for object #1 and '2' for the objects #2 and #3)
    subplotID   = 1  # Subplot ID for each object (e.g. '1' for object #1, '1' for object #2, and '2' for object #3)
    holding     = False
    addLine     = False
    addedLines  = 0  # Line plots
    npage       = 0  # Plot number at the start of a new page (e.g. 'HOLD ON')
    # Used if layout is provided with HOLD ON (e.g. HOLD ON 2,3')
    layout      = None

    customFileIN = open(Custom_file, 'r')

    # Get version number in the header
    version = float(customFileIN.readline().split('|')
                    [1].strip().split('V')[1].strip())
    # Check if the main versions are compatible (e.g. Versions 1.1 and 1.2 are OK but not 1.0 and 2.0)
    if int(version) != int(current_version):
        prYellow('*** Warning ***')
        prYellow('Using MarsPlot V%s but Custom.in template is deprecated (V%s)' % (
            current_version, version))
        prYellow('***************')

    # Skip the header
    while (customFileIN.readline()[0] != '<'):
        pass
    # Read paths under <<<<<<<<<< Simulations >>>>>>>>>>>
    while True:
        line = customFileIN.readline()
        if line[0] == '#':  # Skip comments
            pass
        else:
            if line[0] == '=':
                break  # Finished reading
            # Special case: use a reference simulation
            if line.split('>')[0] == 'ref':
                # If it is different from default, overwrite it
                if line.split('>')[1].strip() != 'None':
                    input_paths[0] = line.split('>')[1].strip()
            else:
                if '>' in line:  # Line contains '>' symbol
                    if line.split('>')[1].strip():  # Line exists and is not blank
                        input_paths.append(line.split('>')[1].strip())

    # Skip lines until the keyword 'START' is found
    nsafe = 0  # Initialize counter for safety
    while True and nsafe < 2000:
        line = customFileIN.readline()
        if line.strip() == 'START':
            break
        nsafe += 1
    if nsafe == 2000:
        prRed(
            """ Custom.in is missing a 'START' keyword after the '=====' simulation block """)

    # Start reading the figure templates
    while True:
        line = customFileIN.readline()

        if not line or line.strip() == 'STOP':
            break  # Reached end of file

        if line.strip()[0:7] == 'HOLD ON':
            holding = True
            subplotID = 1

            # Get layout info
            if ',' in line:  # Layout is provided (e.g. 'HOLD ON 2,3')
                # This returns '2,3' from above as a string
                tmp = line.split('ON')[-1].strip()
                layout = [int(tmp.split(',')[0]), int(
                    tmp.split(',')[1])]  # This returns [2,3]
            else:
                layout = None
        # Adding a 1D plot to an existing line plot
        if line.strip() == 'ADD LINE':
            addLine = True

        if line[0] == '<':  # If new figure
            figtype, boolPlot = get_figure_header(line)
            if boolPlot:  # Only if we want to plot the field
                # Add object to the list
                if figtype == 'Plot 2D lon X lat':
                    objectList.append(Fig_2D_lon_lat())
                if figtype == 'Plot 2D time X lat':
                    objectList.append(Fig_2D_time_lat())
                if figtype == 'Plot 2D lat X lev':
                    objectList.append(Fig_2D_lat_lev())
                if figtype == 'Plot 2D lon X lev':
                    objectList.append(Fig_2D_lon_lev())
                if figtype == 'Plot 2D time X lev':
                    objectList.append(Fig_2D_time_lev())
                if figtype == 'Plot 2D lon X time':
                    objectList.append(Fig_2D_lon_time())
                if figtype == 'Plot 1D':
                    objectList.append(Fig_1D())
                objectList[nobj].read_template()
                nobj += 1

                # Debug only
                #print('------nobj=',nobj,' npage=',npage,'-------------------')
                # ===================

                if holding and not addLine:
                    subplotList.append(subplotID)
                    panelList.append(subplotID)
                    subplotID += 1
                    # Add +1 panel to all plots on current page
                    for iobj in range(npage, nobj-1):
                        panelList[iobj] += 1

                elif holding and addLine:
                    # Do not update subplot ID if adding lines
                    subplotList.append(subplotID-1)
                    panelList.append(subplotID-1)

                else:
                    # Do not hold: one plot per page. Reset the page counter.
                    panelList.append(1)
                    subplotList.append(1)
                    npage = nobj
                    layout = None

                if layout:
                    layoutList.append(layout)
                else:
                    layoutList.append(None)
                # ====================

                if addLine:
                    addedLines += 1
                    addLineList.append(addedLines)
                else:
                    addLineList.append(0)  # No added lines
                    addedLines = 0  # Reset line counter

                # Debug only
                # for ii in range(0,len(   subplotList)):
                #    prCyan('[X,%i,%i,%i]'%(subplotList[ii],panelList[ii],addLineList[ii]))
                # =================

                # Deprecated - an old way to attribute the plot numbers without using npage
                # if holding:
                #     subplotList.append(subplotID-addedLines)
                #     panelList.append(subplotID-addedLines)
                #     if not addLine:
                #         # add +1 to the number of panels for the previous plots
                #         n=1
                #         while n<=subplotID-1:
                #             panelList[nobj-n-1]+=1 #print('editing %i panels, now %i'%(subplotID-1,nobj-n-1))
                #             n+=1
                #     subplotID+=1
                # else :
                #     panelList.append(1)
                #     subplotList.append(1)
                # ========================================================

            addLine = False  # Reset after reading each block
        if line.strip() == 'HOLD OFF':
            holding = False
            subplotID = 1
            npage = nobj

    # Make sure we are not still holding figures
    if holding:
        prRed('*** Error ***')
        prRed("""Missing 'HOLD OFF' statement in """+Custom_file)
        exit()
    # Make sure we are not still holding figures
    if addLine:
        prRed('*** Error ***')
        prRed("""Cannot have 'ADD LINE' after the last figure in """+Custom_file)
        exit()
    # Finished reading the file, attribute the right number of figure and panels for each plot
    # print('======= Summary =========')
    for i in range(0, nobj):
        objectList[i].subID = subplotList[i]
        objectList[i].nPan = panelList[i]
        objectList[i].addLine = addLineList[i]
        objectList[i].layout = layoutList[i]

        # Debug only
        # prPurple('%i:[%i,%i,%i]'%(i,objectList[i].subID,objectList[i].nPan,objectList[i].addLine))
    customFileIN.close()


def get_figure_header(line_txt):
    '''
    This function returns the type of figure, indicates that plotting is set to True.
    Args:
        line_txt: string, figure header from Custom.in (i.e.'<<<<<<<<<| Plot 2D lon X lat = True |>>>>>>>>')
    Returns:
        figtype:  string, figure type (i.e  Plot 2D lon X lat)
        boolPlot: bool, False if plot skipped
    '''
    line_cmd = line_txt.split('|')[1].strip()  # Plot 2D lon X lat = True
    figtype  = line_cmd.split('=')[0].strip()  # Plot 2D lon X lat
    boolPlot = line_cmd.split('=')[1].strip() == 'True'  # Return True
    return figtype, boolPlot


def format_lon_lat(lon_lat, type):
    '''
    Format latitude and longitude as labels (e.g. 30S, 30N, 45W, 45E)
    Args:
        lon_lat (float): latitude or longitude (+180/-180)
        type (string):   'lat' or 'lon'
    Returns:
        lon_lat_label:   string, formatted label
    '''
    # Initialize
    letter = ""
    if type == 'lon':
        if lon_lat < 0:
            letter = "W"
        if lon_lat > 0:
            letter = "E"
    elif type == 'lat':
        if lon_lat < 0:
            letter = "S"
        if lon_lat > 0:
            letter = "N"
    # Remove minus sign, if any
    lon_lat = abs(lon_lat)
    return "%i%s" % (lon_lat, letter)


# ======================================================
#                  FILE SYSTEM UTILITIES
# ======================================================

def get_Ncdf_num():
    '''
    Get the sol numbers of all the netcdf files in the directory.
    This test is based on the presence of a least one 'fixed' file in the current directory.
    Args:
        None
    Returns:
        Ncdf_num: a sorted array of sols
    '''
    list_dir = os.listdir(input_paths[0]) # e.g. '00350.fixed.nc' or '00000.fixed.nc'
    avail_fixed = [k for k in list_dir if '.fixed.nc' in k]
    # Remove .fixed.nc (returning '00350' or '00000')
    list_num = [item[0:5] for item in avail_fixed]
    # Transform to array (returning [0, 350])
    Ncdf_num = np.sort(np.asarray(list_num).astype(float))
    if Ncdf_num.size == 0:
        Ncdf_num = None
    #    print("No 'fixed' detected in "+input_paths[0])
    #    raise SystemExit #Exit cleanly
    return Ncdf_num


def select_range(Ncdf_num, bound):
    '''
    Args:
        Ncdf_num:   a sorted array of sols
        bound:      integer, represents a date (e.g. 0350) or an array containing the sol bounds (e.g. [min max])
    Returns:
        Ncdf_num:   a sorted array of sols within the bounds
    '''
    bound = np.array(bound)
    if bound.size == 1:
        Ncdf_num = Ncdf_num[Ncdf_num == bound]
        if Ncdf_num.size == 0:
            prRed('*** Error ***')
            prRed("File %05d.fixed.nc not found" % (bound))
            exit()
    elif bound.size == 2:
        Ncdf_num = Ncdf_num[Ncdf_num >= bound[0]]
        Ncdf_num = Ncdf_num[Ncdf_num <= bound[1]]
        if Ncdf_num.size == 0:
            prRed('*** Error ***')
            prRed(
                "No 'fixed' file with date between [%05d-%05d] detected. Please double check date range." % (bound[0], bound[1]))
            exit()
    return Ncdf_num


def create_name(root_name):
    '''
    Modify desired file name if a file with that name already exists.
    Args:
        root_name:  desired name for the file (e.g."/path/custom.in" or "/path/figure.png")
    Returns:
        new_name:   new name if the file already exists (e.g. "/path/custom_01.in" or "/path/figure_01.png")
    '''
    n = 1
    # Get extension length (e.g. 2 for *.nc, 3 for *.png)
    len_ext = len(root_name.split('.')[-1])
    ext = root_name[-len_ext:]
    # Initialization
    new_name = root_name
    # If example.png already exists, create example_01.png
    if os.path.isfile(new_name):
        new_name = root_name[0:-(len_ext+1)]+'_%02d' % (n)+'.'+ext
    # If example_01.png already exists, create example_02.png etc.
    while os.path.isfile(root_name[0:-(len_ext+1)]+'_%02d' % (n)+'.'+ext):
        n = n+1
        new_name = root_name[0:-(len_ext+1)]+'_%02d' % (n)+'.'+ext
    return new_name


def path_to_template(custom_name):
    '''
    Modify desired file name if a file with that name already exists.
    Args:
        custom_name:    Custom.in file name. Accepted formats are my_custom or my_custom.in
    Returns:
        full_path:      Full path to the template (e.g. /u/$USER/FV3/templates/my_custom.in)
                        If the file is not found, try the shared directory (/u/mkahre/MCMC...)
    '''
    local_dir = sys.prefix+'/mars_templates'

    # Convert the 1-element list to a string
    custom_name = custom_name[0]
    if custom_name[-3:] != '.in':
        custom_name = custom_name+'.in'  # Add extension if not provided
    # First look in  ~/FV3/templates
    if not os.path.isfile(local_dir+'/'+custom_name):
        # Then look in  /lou/s2n/mkahre/MCMC/analysis/working/templates
        if not os.path.isfile(shared_dir+'/'+custom_name):
            prRed('*** Error ***')
            prRed('File '+custom_name+' not found in '+local_dir +
                  ' ... nor in \n                          '+shared_dir)
            # If a local ~/FV3/templates path does not exist, suggest it be created
            if not os.path.exists(local_dir):
                prYellow('Note: directory: ~/FV3/templates' +
                         ' does not exist, create it with:')
                prCyan('mkdir '+local_dir)
            exit()
        else:
            return shared_dir+'/'+custom_name
    else:
        return local_dir+'/'+custom_name


def progress(k, Nmax, txt='', success=True):
    """
    Display a progress bar to monitor heavy calculations.
    Args:
        k:      current iteration of the outer loop
        Nmax:   max iteration of the outer loop
    Returns:
        Running... [#---------] 10.64 %
    """
    import sys
    progress = float(k)/Nmax
    # Sets the length of the progress bar
    barLength = 10
    block = int(round(barLength*progress))
    bar = "[{0}]".format("#"*block + "-"*(barLength-block))
    # bar = "Running... [\033[96m{0}\033[00m]".format( "#"*block + "-"*(barLength-block))  # add color
    if success == True:
        # status="%i %% (%s)"%(100*progress,txt) # No color
        status = "%3i %% \033[92m(%s)\033[00m" % (100*progress, txt)  # Green
    elif success == False:
        status = "%3i %% \033[91m(%s)\033[00m" % (100*progress, txt)  # Red
    elif success == None:
        status = "%3i %% (%s)" % (100*progress, txt)  # Red
    text = '\r'+bar+status+'\n'
    sys.stdout.write(text)
    if not debug:
        sys.stdout.flush()


def prep_file(var_name, file_type, simuID, sol_array):
    '''
    Open the file as a Dataset or MFDataset object depending on its status on tape (Lou)
    Note that the input arguments are typically extracted from a 'varfull' object (e.g. '03340.atmos_average.ucomp')
    and not from a file whose existence on the disk is known beforehand.
    Args:
        var_name:   variable to extract (e.g. 'ucomp')
        file_type:  MGCM output file type (e.g. 'average' for atmos_average_pstd)
        simuID:     Simulation ID number (e.g. 2 for 2nd simulation)
        sol_array:  Date in file name (e.g. [3340,4008])

    Returns:
        f: Dataset or MFDataset object
        var_info: longname and units
        dim_info: dimensions e.g. ('time', 'lat','lon')
        dims:    shape of the array e.g. [133,48,96]
    '''
    global input_paths
    # global variable that holds the different sol numbers (e.g. [1500,2400])
    global Ncdf_num
    # A specific sol is requested (e.g. [2400])
    Sol_num_current = [0]  # Set dummy value
    # First check if the file exist on tape without a sol number (e.g. 'Luca_dust_MY24_dust.nc' exists on the disk)
    if os.path.isfile(input_paths[simuID]+'/'+file_type+'.nc'):
        file_has_sol_number = False
    # If the file does NOT exist, append the sol number provided by MarsPlot (e.g. Custom.in -d XXXX)
    # or the sol number of the last file in the directory
    else:
        file_has_sol_number = True
        # Two options here: First a file number is explicitly provided in varfull, (e.g. 00668.atmos_average.nc)
        if sol_array != [None]:
            Sol_num_current = sol_array
        elif Ncdf_num != None:
            Sol_num_current = Ncdf_num
    # Create a list of files (even if only one file is provided)
    nfiles = len(Sol_num_current)
    file_list = [None]*nfiles  # Initialize the list

    # Loop over the requested timesteps
    for i in range(0, nfiles):
        if file_has_sol_number:  # Include sol number
            file_list[i] = input_paths[simuID] + \
                '/%05d.' % (Sol_num_current[i])+file_type+'.nc'
        else:  # No sol number
            file_list[i] = input_paths[simuID]+'/'+file_type+'.nc'
        check_file_tape(file_list[i], abort=False)
    # We know the files exist on tape, now open it with MFDataset if an aggregation dimension is detected
    try:
        f = MFDataset(file_list, 'r')
    except IOError:
        # This IOError should be: 'master dataset ***.nc does not have a aggregation dimension'
        # Use Dataset otherwise
        f = Dataset(file_list[0], 'r')

    var_info = getattr(f.variables[var_name], 'long_name', '') + \
        ' [' + getattr(f.variables[var_name], 'units', '')+']'
    dim_info = f.variables[var_name].dimensions
    dims = f.variables[var_name].shape
    return f, var_info, dim_info, dims


class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x < 0:
            return LogFormatterSciNotation.__call__(self, x, pos=None)
        else:
            return "{x:g}".format(x=x)


# ======================================================
#                  FIGURE DEFINITIONS
# ======================================================
class Fig_2D(object):
    # Parent class for 2D figures
    def __init__(self, varfull='fileYYY.XXX', doPlot=False, varfull2=None):

        self.title    = None
        self.varfull  = varfull
        self.range    = None
        self.fdim1    = None
        self.fdim2    = None
        self.ftod     = None  # Time of day
        self.varfull2 = varfull2
        self.contour2 = None
        # Logic
        self.doPlot    = doPlot
        self.plot_type = self.__class__.__name__[4:]

        # Extract filetype, variable, and simulation ID (initialized only for the default plots)
        # Note that the 'varfull' objects for the default plots are simple (e.g. atmos_average.ucomp)
        self.sol_array, self.filetype, self.var, self.simuID = split_varfull(
            self.varfull)
        # prCyan(self.sol_array);prYellow(self.filetype);prGreen(self.var);prPurple(self.simuID)
        if self.varfull2:
            self.sol_array2, self.filetype2, self.var2, self.simuID2 = split_varfull(
                self.varfull2)

        # Multipanel
        self.nPan   = 1
        self.subID  = 1
        self.layout = None  # e.g. [2,3], used only if 'HOLD ON 2,3' is used

        # Annotation for free dimensions
        self.fdim_txt  = ''
        self.success   = False
        self.addLine   = False
        self.vert_unit = ''  # m or Pa

        # Axis options
        self.Xlim = None
        self.Ylim = None
        self.axis_opt1 = 'jet'
        self.axis_opt2 = 'lin' # Linear or logscale
        self.axis_opt3 = None  # place holder for projections

    def make_template(self, plot_txt, fdim1_txt, fdim2_txt, Xaxis_txt, Yaxis_txt):
        customFileIN.write(
            "<<<<<<<<<<<<<<| {0:<15} = {1} |>>>>>>>>>>>>>\n".format(plot_txt, self.doPlot))
        customFileIN.write("Title          = %s\n" % (self.title))          # 1
        customFileIN.write("Main Variable  = %s\n" % (self.varfull))        # 2
        customFileIN.write("Cmin, Cmax     = %s\n" % (self.range))          # 3
        customFileIN.write("{0:<15}= {1}\n".format(fdim1_txt, self.fdim1))  # 4
        customFileIN.write("{0:<15}= {1}\n".format(fdim2_txt, self.fdim2))  # 4
        customFileIN.write("2nd Variable   = %s\n" % (self.varfull2))       # 6
        customFileIN.write("Contours Var 2 = %s\n" % (self.contour2))       # 7

        # Write colormap AND projection if plot is 2D_lon_lat
        if self.plot_type == '2D_lon_lat':
            customFileIN.write("Axis Options  : {0} = [None,None] | {1} = [None,None] | cmap = jet | scale = lin | proj = cart \n".format(
                Xaxis_txt, Yaxis_txt)) # 8
        else:
            customFileIN.write("Axis Options  : {0} = [None,None] | {1} = [None,None] | cmap = jet |scale = lin \n".format(
                Xaxis_txt, Yaxis_txt)) # 8

    def read_template(self):
        self.title    = rT('char')  # 1
        self.varfull  = rT('char')  # 2
        self.range    = rT('float') # 3
        self.fdim1    = rT('float') # 4
        self.fdim2    = rT('float') # 5
        self.varfull2 = rT('char')  # 6
        self.contour2 = rT('float') # 7
        self.Xlim, self.Ylim, self.axis_opt1, self.axis_opt2, self.axis_opt3 = read_axis_options(
            customFileIN.readline()) # 8

        # Various sanity checks
        if self.range and len(np.atleast_1d(self.range)) == 1:
            prYellow(
                '*** Warning *** In plot %s, Cmin, Cmax must be two values. Resetting to default' % (self.varfull))
            self.range = None

        # Do not update the variable after reading template
        # self.sol_array,self.filetype,self.var,self.simuID=split_varfull(self.varfull)
        #if self.varfull2: self.sol_array2,self.filetype2,self.var2,self.simuID2=split_varfull(self.varfull2)

    def data_loader_2D(self, varfull, plot_type):
        # Simply plot one of the variables in the file
        if not '[' in varfull:
            # If overwriting a dimension, get the new dimension and trim 'varfull' from the '{lev=5.}' part
            if '{' in varfull:
                varfull, fdim1_extract, fdim2_extract, ftod_extract = get_overwrite_dim_2D(
                    varfull, plot_type, self.fdim1, self.fdim2, self.ftod)
                # fdim1_extract,fdim2_extract constains the dimensions to overwrite is '{}' are provided of the default self.fdim1, self.fdim2  otherwise
            else:  # no '{ }' used to overwrite the dimensions, copy the plot defaults
                fdim1_extract, fdim2_extract, ftod_extract = self.fdim1, self.fdim2, self.ftod

            sol_array, filetype, var, simuID = split_varfull(varfull)
            xdata, ydata, var, var_info = self.read_NCDF_2D(
                var, filetype, simuID, sol_array, plot_type, fdim1_extract, fdim2_extract, ftod_extract)
        # Recognize an operation on the variables
        else:
            VAR = []
            # Extract individual variables and prepare for execution
            varfull         = remove_whitespace(varfull)
            varfull_list    = get_list_varfull(varfull)
            # Initialize list of requested dimensions
            fdim1_list      = [None]*len(varfull_list)
            fdim2_list      = [None]*len(varfull_list)
            ftod_list       = [None]*len(varfull_list)
            expression_exec = create_exec(varfull, varfull_list)

            for i in range(0, len(varfull_list)):
                # If overwriting a dimension, get the new dimension and trim 'varfull' from the '{lev=5.}' part
                if '{' in varfull_list[i]:
                    varfull_list[i], fdim1_list[i], fdim2_list[i], ftod_list[i] = get_overwrite_dim_2D(
                        varfull_list[i], plot_type, self.fdim1, self.fdim2, self.ftod)
                else:  # No '{ }' used to overwrite the dimensions, copy the plot defaults
                    fdim1_list[i], fdim2_list[i], ftod_list[i] = self.fdim1, self.fdim2, self.ftod

                sol_array, filetype, var, simuID = split_varfull(
                    varfull_list[i])
                xdata, ydata, temp, var_info = self.read_NCDF_2D(
                    var, filetype, simuID, sol_array, plot_type, fdim1_list[i], fdim2_list[i], ftod_list[i])
                VAR.append(temp)
            var_info = varfull
            var = eval(expression_exec)

        return xdata, ydata, var, var_info

    def read_NCDF_2D(self, var_name, file_type, simuID, sol_array, plot_type, fdim1, fdim2, ftod):
        f, var_info, dim_info, dims = prep_file(
            var_name, file_type, simuID, sol_array)

        # Get the file type ('fixed', 'diurn', 'average', 'daily') and interpolation type (pfull, zstd, etc.)
        f_type, interp_type = FV3_file_type(f)

        # Initialize dimensions (these are in all the .nc files)
        lat = f.variables['lat'][:]
        lati = np.arange(0, len(lat))
        lon = f.variables['lon'][:]
        loni = np.arange(0, len(lon))

        # If self.fdim is empty, add the variable name (do only once)
        add_fdim = False
        if not self.fdim_txt.strip():
            add_fdim = True
        var_thin = False

        # ------------------------ Time of Day ----------------------------
        # For diurn files, select data on the time of day axis and update dimensions
        # so that the resulting variable is the same as in 'average' and 'daily' files.
        # Time of day is always the 2nd dimension (dim_info[1])

        if f_type == 'diurn' and dim_info[1][:11] == 'time_of_day':
            tod = f.variables[dim_info[1]][:]
            todi, temp_txt = get_tod_index(ftod, tod)
            # Update dim_info from ('time', 'time_of_day_XX, 'lat', 'lon') to  ('time', 'lat', 'lon')
            # OR ('time', 'time_of_day_XX, 'pfull', 'lat', 'lon') to  ('time', 'pfull', 'lat', 'lon')
            dim_info = (dim_info[0],)+dim_info[2:]

            if add_fdim:
                self.fdim_txt += temp_txt
        # -----------------------------------------------------------------------

        # Load variable depending on the requested free dimensions
        # ====== static ======= ignore 'level' and 'time' dimension
        if dim_info == ('lat', 'lon'):
            var = f.variables[var_name][lati, loni]
            f.close()
            return lon, lat, var, var_info

        # ====== time,lat,lon =======
        if dim_info == ('time', 'lat', 'lon'):
            # Initialize dimension
            t = f.variables['time'][:]
            LsDay = np.squeeze(f.variables['areo'][:])
            ti = np.arange(0, len(t))
            # For 'diurn' file, change time_of_day(time, 24, 1) to time_of_day(time) at midnight UT
            if f_type == 'diurn' and len(LsDay.shape) > 1:
                LsDay = np.squeeze(LsDay[:, 0])
            # Stack the 'time' and 'areo' array as one variable
            t_stack = np.vstack((t, LsDay))

            if plot_type == '2D_lon_lat':
                ti, temp_txt = get_time_index(fdim1, LsDay)
            if plot_type == '2D_time_lat':
                loni, temp_txt = get_lon_index(fdim1, lon)
            if plot_type == '2D_lon_time':
                lati, temp_txt = get_lat_index(fdim1, lat)

            if add_fdim:
                self.fdim_txt += temp_txt

            # Extract data and close file
            # If 'diurn', do the time of day average first.
            if f_type == 'diurn':
                var = f.variables[var_name][ti, todi, lati, loni].reshape(len(np.atleast_1d(ti)), len(np.atleast_1d(todi)),
                                                                          len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
                var = mean_func(var, axis=1)
            else:
                var = f.variables[var_name][ti, lati, loni].reshape(
                    len(np.atleast_1d(ti)), len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
            f.close()
            w = area_weights_deg(var.shape, lat[lati])

            # Return data
            if plot_type == '2D_lon_lat':
                # Time average
                return lon, lat, mean_func(var, axis=0), var_info
            if plot_type == '2D_time_lat':
                # Transpose, X dimension must be in last column of variable
                return t_stack, lat, mean_func(var, axis=2).T, var_info
            if plot_type == '2D_lon_time':
                return lon, t_stack, np.average(var, weights=w, axis=1), var_info

        # ====== time, level, lat, lon =======
        if (dim_info   == ('time', 'pfull', 'lat', 'lon')
           or dim_info == ('time', 'level', 'lat', 'lon')
           or dim_info == ('time', 'pstd',  'lat', 'lon')
           or dim_info == ('time', 'zstd',  'lat', 'lon')
           or dim_info == ('time', 'zagl',  'lat', 'lon')
           or dim_info == ('time', 'zgrid', 'lat', 'lon')
           or dim_info == ('zgrid', 'lat',  'lon')):

            if dim_info[1] in ['pfull', 'level', 'pstd']:
                self.vert_unit = 'Pa'
            if dim_info[1] in ['zagl', 'zstd', 'zgrid']:
                self.vert_unit = 'm'
            if dim_info[0] in ['zgrid']:  # Thermal inertia is a special case
                self.vert_unit = 'm'
                var_thin = True

            # Initialize dimensions
            if var_thin == True:
                levs = f.variables[dim_info[0]][:]  # dim_info[0] is 'zgrid'
                zi   = np.arange(0, len(levs))
            elif var_thin == False:
                # dim_info[1] is either 'pfull', 'level', 'pstd', 'zstd', 'zagl', or 'zgrid'
                levs = f.variables[dim_info[1]][:]
                zi   = np.arange(0, len(levs))
                t    = f.variables['time'][:]
                LsDay   = np.squeeze(f.variables['areo'][:])
                ti   = np.arange(0, len(t))
                # For 'diurn' file, change time_of_day(time, 24, 1) to time_of_day(time) at midnight UT
                if f_type == 'diurn' and len(LsDay.shape) > 1:
                    LsDay = np.squeeze(LsDay[:, 0])
                # Stack the 'time' and 'areo' arrays as one variable
                t_stack = np.vstack((t, LsDay))

            if plot_type == '2D_lon_lat':
                if var_thin == True:
                    zi, temp_txt = get_level_index(fdim2, levs)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                elif var_thin == False:
                    ti, temp_txt = get_time_index(fdim1, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    zi, temp_txt = get_level_index(fdim2, levs)
                    if add_fdim:
                        self.fdim_txt += temp_txt

            if plot_type == '2D_time_lat':
                loni, temp_txt  = get_lon_index(fdim1, lon)
                if add_fdim:
                    self.fdim_txt += temp_txt
                zi, temp_txt    = get_level_index(fdim2, levs)
                if add_fdim:
                    self.fdim_txt += temp_txt

            if plot_type == '2D_lat_lev':
                if var_thin == True:
                    loni, temp_txt  = get_lon_index(fdim2, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                elif var_thin == False:
                    ti, temp_txt    = get_time_index(fdim1, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    loni, temp_txt  = get_lon_index(fdim2, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt

            if plot_type == '2D_lon_lev':
                if var_thin == True:
                    lati, temp_txt  = get_lat_index(fdim2, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                elif var_thin == False:
                    ti, temp_txt    = get_time_index(fdim1, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    lati, temp_txt  = get_lat_index(fdim2, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt

            if plot_type == '2D_time_lev':
                lati, temp_txt  = get_lat_index(fdim1, lat)
                if add_fdim:
                    self.fdim_txt += temp_txt
                loni, temp_txt  = get_lon_index(fdim2, lon)
                if add_fdim:
                    self.fdim_txt += temp_txt

            if plot_type == '2D_lon_time':
                lati, temp_txt  = get_lat_index(fdim1, lat)
                if add_fdim:
                    self.fdim_txt += temp_txt
                zi, temp_txt    = get_level_index(fdim2, levs)
                if add_fdim:
                    self.fdim_txt += temp_txt

            # If 'diurn' do the time of day average first.
            if f_type == 'diurn':
                var = f.variables[var_name][ti, todi, zi, lati, loni].reshape(len(np.atleast_1d(ti)), len(np.atleast_1d(todi)),
                                                                              len(np.atleast_1d(zi)), len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
                var = mean_func(var, axis=1)
            elif var_thin == True:
                var = f.variables[var_name][zi, lati, loni].reshape(len(np.atleast_1d(zi)),
                                                                    len(np.atleast_1d(
                                                                        lati)),
                                                                    len(np.atleast_1d(loni)))
            else:
                var = f.variables[var_name][ti, zi, lati, loni].reshape(len(np.atleast_1d(ti)),
                                                                        len(np.atleast_1d(
                                                                            zi)),
                                                                        len(np.atleast_1d(
                                                                            lati)),
                                                                        len(np.atleast_1d(loni)))
            f.close()
            w = area_weights_deg(var.shape, lat[lati])

            #(u'time', u'pfull', u'lat', u'lon')
            if var_thin == True:
                if plot_type == '2D_lon_lat':
                    return lon,   lat,  mean_func(var, axis=0), var_info
                if plot_type == '2D_lat_lev':
                    return lat, levs,    mean_func(var, axis=2), var_info
                if plot_type == '2D_lon_lev':
                    return lon, levs,    mean_func(var, weights=w, axis=1), var_info
            else:
                if plot_type == '2D_lon_lat':
                    return lon,   lat,  mean_func(mean_func(var, axis=1), axis=0), var_info
                if plot_type == '2D_time_lat':
                    # transpose
                    return t_stack, lat,  mean_func(mean_func(var, axis=1), axis=2).T, var_info
                if plot_type == '2D_lat_lev':
                    return lat, levs,    mean_func(mean_func(var, axis=3), axis=0), var_info
                if plot_type == '2D_lon_lev':
                    return lon, levs,    mean_func(np.average(var, weights=w, axis=2), axis=0), var_info
                if plot_type == '2D_time_lev':
                    # transpose
                    return t_stack, levs, mean_func(np.average(var, weights=w, axis=2), axis=2).T, var_info
                if plot_type == '2D_lon_time':
                    return lon, t_stack, mean_func(np.average(var, weights=w, axis=2), axis=1), var_info

    def plot_dimensions(self):
        prYellow(f'{self.ax.get_position()}')

    def make_title(self, var_info, xlabel, ylabel):
        if self.title:
            # If Title is provided
            if '{fontsize=' in self.title:
                # If fontsize is specified
                fs = int(remove_whitespace(
                    (self.title).split("{fontsize=")[1].split("}")[0]))
                title_text = ((self.title).split("{fontsize=")[0])
                plt.title(title_text, fontsize=fs -
                          self.nPan*title_factor, wrap=False)
            else:
                # If fontsize is not specified
                plt.title(self.title, fontsize=title_size -
                          self.nPan*title_factor)
        else:
            # If title is not provided
            plt.title(
                var_info+'\n'+self.fdim_txt[1:], fontsize=title_size-self.nPan*title_factor, wrap=False)

        plt.xlabel(xlabel, fontsize=label_size-self.nPan*label_factor)
        plt.ylabel(ylabel, fontsize=label_size-self.nPan*label_factor)

    def make_colorbar(self, levs):
        if self.axis_opt2 == 'log':
            formatter = LogFormatter(10, labelOnlyBase=False)
            if self.range:
                cbar = plt.colorbar(
                    ticks=levs, orientation='horizontal', aspect=30, format=formatter)
            else:
                cbar = plt.colorbar(orientation='horizontal',
                                    aspect=30, format=formatter)

        else:
            cbar = plt.colorbar(orientation='horizontal', aspect=30)

        # Shrink the colorbar label as the number of subplots increases
        cbar.ax.tick_params(labelsize=label_size-self.nPan*label_factor)

    def return_norm_levs(self):
        norm = None
        levs = None
        if self.axis_opt2 == 'log':
            # Logarithmic colormap
            norm = LogNorm()
        else:
            # Linear colormap (default)
            self.axis_opt2 = 'lin'
            norm = None
        if self.range:
            if self.axis_opt2 == 'lin':
                # If two numbers are provided (e.g. Cmin,Cmax)
                if len(self.range) == 2:
                    levs = np.linspace(self.range[0], self.range[1], levels)
                # If a list is provided setting the intervals explicitly
                else:
                    levs = self.range

            if self.axis_opt2 == 'log':
                if self.range[0] <= 0 or self.range[1] <= 0:
                    prRed(
                        '*** Error using log scale, bounds cannot be zero or negative')
                levs = np.logspace(
                    np.log10(self.range[0]), np.log10(self.range[1]), levels)
        return norm, levs

    def exception_handler(self, e, ax):
        if debug:
            raise
        sys.stdout.write("\033[F")
        # Cursor up one line, then clear the line's previous output
        sys.stdout.write("\033[K")
        prYellow('*** Warning *** %s' % (e))
        ax.text(0.5, 0.5, 'ERROR:'+str(e), horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round", ec=(
                    1., 0.5, 0.5), fc=(1., 0.8, 0.8),),
                transform=ax.transAxes, wrap=True, fontsize=16)

    def fig_init(self):
        # Create figure
        if self.layout is None:
            # If no layout is specified
            out = fig_layout(self.subID, self.nPan, vertical_page)
        else:
            # If layout is specified
            out = np.append(self.layout, self.subID)
        if self.subID == 1:
            # Create figure if 1st panel
            # 1.4 is ratio (16:9 screen would be 1.77)
            fig = plt.figure(facecolor='white',
                             figsize=(width_inch, height_inch))

        ax = plt.subplot(out[0], out[1], out[2])  # nrow, ncol, subID
        ax.patch.set_color('.1')  # Nans are grey
        return ax

    def fig_save(self):
        # Save the figure
        if self.subID == self.nPan:  # Last subplot
            if self.subID == 1:  # 1 plot
                if not '[' in self.varfull:
                    # Add split '{' in case 'varfull' contains layer. Does not do anything else.
                    sensitive_name = self.varfull.split('{')[0].strip()
                    # If 'varfull' is a complex expression
                else:
                    sensitive_name = 'expression_' + \
                        get_list_varfull(self.varfull)[0].split('{')[0].strip()
            else:  # Multipanel
                sensitive_name = 'multi_panel'
            plt.tight_layout()
            self.fig_name = output_path+'/plots/'+sensitive_name+'.'+out_format
            self.fig_name = create_name(self.fig_name)
            plt.savefig(self.fig_name, dpi=my_dpi)
            if out_format != "pdf":
                print("Saved:" + self.fig_name)

    def filled_contour(self, xdata, ydata, var):
        cmap = self.axis_opt1
        # Personalized colormaps
        if cmap == 'wbr':
            cmap = wbr_cmap()
        if cmap == 'rjw':
            cmap = rjw_cmap()
        if cmap == 'dkass_temp':
            cmap = dkass_temp_cmap()
        if cmap == 'dkass_dust':
            cmap = dkass_dust_cmap()

        norm, levs = self.return_norm_levs()

        if self.range:
            plt.contourf(xdata, ydata, var, levs,
                         extend='both', cmap=cmap, norm=norm)
        else:
            plt.contourf(xdata, ydata, var, levels, cmap=cmap, norm=norm)

        self.make_colorbar(levs)

    def solid_contour(self, xdata, ydata, var, contours):
        # Prevent error message when drawing contours
        np.seterr(divide='ignore', invalid='ignore')
        if contours is None:
            CS = plt.contour(xdata, ydata, var, 11, colors='k', linewidths=2)
        else:
            # If one contour is provided (as float), convert it to an array
            if type(contours) == float:
                contours = [contours]
            CS = plt.contour(xdata, ydata, var, contours,
                             colors='k', linewidths=2)
        plt.clabel(CS, inline=1, fontsize=14, fmt='%g')


# ===============================

class Fig_2D_lon_lat(Fig_2D):

    # Make_template calls method from the parent class
    def make_template(self):
        super(Fig_2D_lon_lat, self).make_template(
            'Plot 2D lon X lat', 'Ls 0-360', 'Level Pa/m', 'lon', 'lat')

    def get_topo_2D(self, varfull, plot_type):
        '''
        This function returns the longitude, latitude, and topography to overlay as contours in a 2D_lon_lat plot.
        Because the main variable requested may be complex (e.g. [00668.atmos_average_psdt2.temp]/1000.), we will ensure to
        load the matching topography (here 00668.fixed.nc from the 2nd simulation). This function essentially does a simple
        task in a complicated way. Note that a great deal of the code is borrowed from the data_loader_2D() function.

        Returns:
            zsurf: topography or 'None' if no matching 'fixed' file is found
        '''

        if not '[' in varfull:
            # If overwriting a dimension, get the new dimension and trim 'varfull' from the '{lev=5.}' part
            if '{' in varfull:
                varfull, _, _, _ = get_overwrite_dim_2D(
                    varfull, plot_type, self.fdim1, self.fdim2, self.ftod)
            sol_array, filetype, var, simuID = split_varfull(varfull)
        # Recognize an operation on the variables
        else:
            # Extract individual variables and prepare for execution
            varfull = remove_whitespace(varfull)
            varfull_list = get_list_varfull(varfull)
            f = get_list_varfull(varfull)
            sol_array, filetype, var, simuID = split_varfull(varfull_list[0])

        # If requesting a lat-lon plot for 00668.atmos_average.nc, try to find matching 00668.fixed.nc
        try:
            f, var_info, dim_info, dims = prep_file(
                'zsurf', 'fixed', simuID, sol_array)
            # Get the file type ('fixed', 'diurn', 'average', 'daily') and interpolation type (pfull, zstd, etc.)
            zsurf = f.variables['zsurf'][:, :]
            f.close()
        except:
            # If input file does not have a corresponding fixed file, return None
            zsurf = None
        return zsurf

    def do_plot(self):

        # Create figure
        ax = super(Fig_2D_lon_lat, self).fig_init()
        try:  # Try to create the figure, return error otherwise
            lon, lat, var, var_info = super(Fig_2D_lon_lat, self).data_loader_2D(
                self.varfull, self.plot_type)
            lon_shift, var = shift_data(lon, var)
            # Try to get topography if a matching 'fixed' file exists
            try:
                surf = self.get_topo_2D(self.varfull, self.plot_type)
                _, zsurf = shift_data(lon, zsurf)
                add_topo = True
            except:
                add_topo = False

            projfull = self.axis_opt3

            # ------------------------------------------------------------------------
            # If proj = cart, use the generic contours utility from the Fig_2D() class
            # ------------------------------------------------------------------------
            if projfull == 'cart':

                super(Fig_2D_lon_lat, self).filled_contour(lon_shift, lat, var)
                # Add topography contour
                if add_topo:
                    plt.contour(lon_shift, lat, zsurf, 11, colors='k',
                                linewidths=0.5, linestyles='solid')

                if self.varfull2:
                    _, _, var2, var_info2 = super(Fig_2D_lon_lat, self).data_loader_2D(
                        self.varfull2, self.plot_type)
                    lon_shift, var2 = shift_data(lon, var2)
                    super(Fig_2D_lon_lat, self).solid_contour(
                        lon_shift, lat, var2, self.contour2)
                    var_info += " (& "+var_info2+")"

                if self.Xlim:
                    plt.xlim(self.Xlim[0], self.Xlim[1])
                if self.Ylim:
                    plt.ylim(self.Ylim[0], self.Ylim[1])

                super(Fig_2D_lon_lat, self).make_title(
                    var_info, 'Longitude', 'Latitude')
             # --- Annotation---
                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.yaxis.set_major_locator(MultipleLocator(15))
                ax.yaxis.set_minor_locator(MultipleLocator(5))
                plt.xticks(fontsize=label_size-self.nPan *
                           tick_factor, rotation=0)
                plt.yticks(fontsize=label_size-self.nPan *
                           tick_factor, rotation=0)

            # -------------------------------------------------------------------
            #                      Special Projections
            # --------------------------------------------------------------------
            else:
                # Personalized colormaps
                cmap = self.axis_opt1
                if cmap == 'wbr':
                    cmap = wbr_cmap()
                if cmap == 'rjw':
                    cmap = rjw_cmap()
                norm, levs = super(Fig_2D_lon_lat, self).return_norm_levs()

                ax.axis('off')
                # Nans are reversed to white for projections
                ax.patch.set_color('1')
                if projfull[0:5] in ['Npole', 'Spole', 'ortho']:
                    ax.set_aspect('equal')
                # ---------------------------------------------------------------

                if projfull == 'robin':
                    LON, LAT = np.meshgrid(lon_shift, lat)
                    X, Y = robin2cart(LAT, LON)

                    # Add meridans and parallelss
                    for mer in np.arange(-180, 180, 30):
                        xg, yg = robin2cart(lat, lat*0+mer)
                        plt.plot(xg, yg, ':k', lw=0.5)
                    # Label every other meridian
                    for mer in np.arange(-180, 181, 90):
                        xl, yl = robin2cart(lat.min(), mer)
                        lab_txt = format_lon_lat(mer, 'lon')
                        plt.text(xl, yl, lab_txt, fontsize=label_size-self.nPan*label_factor,
                                 verticalalignment='top', horizontalalignment='center')
                    for par in np.arange(-60, 90, 30):
                        xg, yg = robin2cart(lon_shift*0+par, lon_shift)
                        plt.plot(xg, yg, ':k', lw=0.5)
                        xl, yl = robin2cart(par, 180)
                        lab_txt = format_lon_lat(par, 'lat')
                        plt.text(xl, yl, lab_txt, fontsize=label_size -
                                 self.nPan*label_factor)
                # ---------------------------------------------------------------

                if projfull == 'moll':
                    LON, LAT = np.meshgrid(lon_shift, lat)
                    X, Y = mollweide2cart(LAT, LON)
                    # Add meridans and parallelss
                    for mer in np.arange(-180, 180, 30):
                        xg, yg = mollweide2cart(lat, lat*0+mer)
                        plt.plot(xg, yg, ':k', lw=0.5)
                    # Label every other meridian
                    for mer in [-180, 0, 180]:
                        xl, yl = mollweide2cart(lat.min(), mer)
                        lab_txt = format_lon_lat(mer, 'lon')
                        plt.text(xl, yl, lab_txt, fontsize=label_size-self.nPan*label_factor,
                                 verticalalignment='top', horizontalalignment='center')

                    for par in np.arange(-60, 90, 30):
                        xg, yg = mollweide2cart(lon_shift*0+par, lon_shift)
                        xl, yl = mollweide2cart(par, 180)
                        lab_txt = format_lon_lat(par, 'lat')
                        plt.plot(xg, yg, ':k', lw=0.5)
                        plt.text(xl, yl, lab_txt, fontsize=label_size -
                                 self.nPan*label_factor)

                if projfull[0:5] in ['Npole', 'Spole', 'ortho']:
                    # Common to all azimuthal projections
                    lon180_original = lon_shift.copy()
                    var, lon_shift = add_cyclic(var, lon_shift)
                    if add_topo:
                        zsurf, _ = add_cyclic(zsurf, lon180_original)
                    lon_lat_custom = None  # Initialization
                    lat_b = None

                    # Get custom lat-lon, if any
                    if len(projfull) > 5:
                        lon_lat_custom = filter_input(projfull[5:], 'float')

                if projfull[0:5] == 'Npole':
                    # Reduce data
                    lat_b = 60
                    if not(lon_lat_custom is None):
                        lat_b = lon_lat_custom  # Bounding lat
                    lat_bi, _ = get_lat_index(lat_b, lat)
                    lat = lat[lat_bi:]
                    var = var[lat_bi:, :]
                    if add_topo:
                        zsurf = zsurf[lat_bi:, :]
                    LON, LAT = np.meshgrid(lon_shift, lat)
                    X, Y = azimuth2cart(LAT, LON, 90, 0)

                    # Add meridans and parallels
                    for mer in np.arange(-180, 180, 30):
                        xg, yg = azimuth2cart(lat, lat*0+mer, 90)
                        plt.plot(xg, yg, ':k', lw=0.5)
                    # Skip 190W to leave room for the Title
                    for mer in np.arange(-150, 180, 30):
                        # Place label 3 degrees south of the bounding latitude
                        xl, yl = azimuth2cart(lat.min()-3, mer, 90)
                        lab_txt = format_lon_lat(mer, 'lon')
                        plt.text(xl, yl, lab_txt, fontsize=label_size-self.nPan*label_factor,
                                 verticalalignment='top', horizontalalignment='center')
                    # Parallels start from 80N, every 10 degrees
                    for par in np.arange(80, lat.min(), -10):
                        xg, yg = azimuth2cart(lon_shift*0+par, lon_shift, 90)
                        plt.plot(xg, yg, ':k', lw=0.5)
                        xl, yl = azimuth2cart(par, 180, 90)
                        lab_txt = format_lon_lat(par, 'lat')
                        plt.text(xl, yl, lab_txt, fontsize=5)
                if projfull[0:5] == 'Spole':
                    lat_b = -60
                    if not(lon_lat_custom is None):
                        lat_b = lon_lat_custom  # Bounding lat
                    lat_bi, _ = get_lat_index(lat_b, lat)
                    lat = lat[:lat_bi]
                    var = var[:lat_bi, :]
                    if add_topo:
                        zsurf = zsurf[:lat_bi, :]
                    LON, LAT = np.meshgrid(lon_shift, lat)
                    X, Y = azimuth2cart(LAT, LON, -90, 0)
                    # Add meridans and parallels
                    for mer in np.arange(-180, 180, 30):
                        xg, yg = azimuth2cart(lat, lat*0+mer, -90)
                        plt.plot(xg, yg, ':k', lw=0.5)
                    # Skip zero to leave room for the Title
                    for mer in np.append(np.arange(-180, 0, 30), np.arange(30, 180, 30)):
                        # Place label 3 degrees north of the bounding latitude
                        xl, yl = azimuth2cart(lat.max()+3, mer, -90)
                        lab_txt = format_lon_lat(mer, 'lon')
                        plt.text(xl, yl, lab_txt, fontsize=label_size-self.nPan*label_factor,
                                 verticalalignment='top', horizontalalignment='center')
                    # Parallels start from 80S, every 10 degrees
                    for par in np.arange(-80, lat.max(), 10):
                        xg, yg = azimuth2cart(lon_shift*0+par, lon_shift, -90)
                        plt.plot(xg, yg, ':k', lw=0.5)
                        xl, yl = azimuth2cart(par, 180, -90)
                        lab_txt = format_lon_lat(par, 'lat')
                        plt.text(xl, yl, lab_txt, fontsize=5)

                if projfull[0:5] == 'ortho':
                    # Initialization
                    lon_p, lat_p = -120, 20
                    if not(lon_lat_custom is None):
                        lon_p = lon_lat_custom[0]
                        lat_p = lon_lat_custom[1]  # Bounding lat
                    LON, LAT = np.meshgrid(lon_shift, lat)
                    X, Y, MASK = ortho2cart(LAT, LON, lat_p, lon_p)
                    # Mask opposite side of the planet
                    var = var*MASK
                    if add_topo:
                        zsurf = zsurf*MASK
                    # Add meridans and parallels
                    for mer in np.arange(-180, 180, 30):
                        xg, yg, maskg = ortho2cart(
                            lat, lat*0+mer, lat_p, lon_p)
                        plt.plot(xg*maskg, yg, ':k', lw=0.5)
                    for par in np.arange(-60, 90, 30):
                        xg, yg, maskg = ortho2cart(
                            lon_shift*0+par, lon_shift, lat_p, lon_p)
                        plt.plot(xg*maskg, yg, ':k', lw=0.5)

                if self.range:
                    plt.contourf(X, Y, var, levs, extend='both',
                                 cmap=cmap, norm=norm)
                else:
                    plt.contourf(X, Y, var, levels, cmap=cmap, norm=norm)

                super(Fig_2D_lon_lat, self).make_colorbar(levs)

                # Add topography contours
                if add_topo:
                    plt.contour(X, Y, zsurf, 11, colors='k',
                                linewidths=0.5, linestyles='solid')  # topo

                # =================================================================================
                # ======================== Solid Contour 2nd Variable =============================
                # =================================================================================
                if self.varfull2:
                    lon, lat, var2, var_info2 = super(
                        Fig_2D_lon_lat, self).data_loader_2D(self.varfull2, self.plot_type)
                    lon_shift, var2 = shift_data(lon, var2)

                    if projfull == 'robin':
                        LON, LAT = np.meshgrid(lon_shift, lat)
                        X, Y = robin2cart(LAT, LON)

                    if projfull == 'moll':
                        LON, LAT = np.meshgrid(lon_shift, lat)
                        X, Y = mollweide2cart(LAT, LON)

                    if projfull[0:5] in ['Npole', 'Spole', 'ortho']:
                        # Common to all azithumal projections
                        var2, lon_shift = add_cyclic(var2, lon_shift)
                        lon_lat_custom = None  # Initialization
                        lat_b = None

                        # Get custom lat-lon, if any
                        if len(projfull) > 5:
                            lon_lat_custom = filter_input(
                                projfull[5:], 'float')

                    if projfull[0:5] == 'Npole':
                        # Reduce data
                        lat_b = 60
                        if not(lon_lat_custom is None):
                            lat_b = lon_lat_custom  # Bounding lat
                        lat_bi, _ = get_lat_index(lat_b, lat)
                        lat = lat[lat_bi:]
                        var2 = var2[lat_bi:, :]
                        LON, LAT = np.meshgrid(lon_shift, lat)
                        X, Y = azimuth2cart(LAT, LON, 90, 0)
                    if projfull[0:5] == 'Spole':
                        lat_b = -60
                        if not(lon_lat_custom is None):
                            lat_b = lon_lat_custom  # Bounding lat
                        lat_bi, _ = get_lat_index(lat_b, lat)
                        lat = lat[:lat_bi]
                        var2 = var2[:lat_bi, :]
                        LON, LAT = np.meshgrid(lon_shift, lat)
                        X, Y = azimuth2cart(LAT, LON, -90, 0)

                    if projfull[0:5] == 'ortho':
                        # Initialization
                        lon_p, lat_p = -120, 20
                        if not(lon_lat_custom is None):
                            lon_p = lon_lat_custom[0]
                            lat_p = lon_lat_custom[1]  # Bounding lat
                        LON, LAT = np.meshgrid(lon_shift, lat)
                        X, Y, MASK = ortho2cart(LAT, LON, lat_p, lon_p)
                        # Mask opposite side of the planet
                        var2 = var2*MASK

                    # Prevent error message for "contours not found"
                    np.seterr(divide='ignore', invalid='ignore')
                    if self.contour2 is None:
                        CS = plt.contour(
                            X, Y, var2, 11, colors='k', linewidths=2)
                    else:
                        # If one contour is provided (as a float), convert it to an array
                        if type(self.contour2) == float:
                            self.contour2 = [self.contour2]
                        CS = plt.contour(
                            X, Y, var2, self.contour2, colors='k', linewidths=2)
                    plt.clabel(CS, inline=1, fontsize=14, fmt='%g')

                    var_info += " (& "+var_info2+")"

                if self.title:
                    plt.title((self.title), fontsize=title_size -
                              self.nPan*title_factor)
                else:
                    plt.title(
                        var_info+'\n'+self.fdim_txt[1:], fontsize=title_size-self.nPan*title_factor, wrap=False)

            self.success = True

        except Exception as e:  # Return the error
            super(Fig_2D_lon_lat, self).exception_handler(e, ax)
        super(Fig_2D_lon_lat, self).fig_save()


class Fig_2D_time_lat(Fig_2D):

    def make_template(self):
        # make_template calls method from the parent class
        super(Fig_2D_time_lat, self).make_template(
            'Plot 2D time X lat', 'Lon +/-180', 'Level [Pa/m]', 'Ls', 'lat')
        #self.fdim1,  self.fdim2, self.Xlim, self.Ylim

    def do_plot(self):
        # Create figure
        ax = super(Fig_2D_time_lat, self).fig_init()
        try:  # Try to create the figure, return error otherwise

            t_stack, lat, var, var_info = super(
                Fig_2D_time_lat, self).data_loader_2D(self.varfull, self.plot_type)
            SolDay = t_stack[0, :]
            LsDay = t_stack[1, :]

            super(Fig_2D_time_lat, self).filled_contour(LsDay, lat, var)

            if self.varfull2:
                _, _, var2, var_info2 = super(Fig_2D_time_lat, self).data_loader_2D(
                    self.varfull2, self.plot_type)
                super(Fig_2D_time_lat, self).solid_contour(
                    LsDay, lat, var2, self.contour2)
                var_info += " (& "+var_info2+")"

            # Axis formatting
            if self.Xlim:
                idmin = np.argmin(np.abs(SolDay-self.Xlim[0]))
                idmax = np.argmin(np.abs(SolDay-self.Xlim[1]))
                plt.xlim([LsDay[idmin], LsDay[idmax]])

            if self.Ylim:
                plt.ylim(self.Ylim[0], self.Ylim[1])

            Ls_ticks = [item for item in ax.get_xticks()]
            labels = [item for item in ax.get_xticklabels()]

            for i in range(0, len(Ls_ticks)):
                # Find timestep closest to this tick
                id = np.argmin(np.abs(LsDay-Ls_ticks[i]))
                if add_sol_time_axis:
                    labels[i] = '%g%s\nsol %i' % (np.mod(Ls_ticks[i], 360.), degr, SolDay[id])
                else:
                    labels[i] = '%g%s' % (np.mod(Ls_ticks[i], 360.), degr)
            ax.set_xticklabels(labels, fontsize=label_size -
                               self.nPan*tick_factor, rotation=0)

            super(Fig_2D_time_lat, self).make_title(
                var_info, 'L$_s$', 'Latitude')

            ax.yaxis.set_major_locator(MultipleLocator(15))
            ax.yaxis.set_minor_locator(MultipleLocator(5))
            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)

            self.success = True

        except Exception as e:  # Return the error
            super(Fig_2D_time_lat, self).exception_handler(e, ax)
        super(Fig_2D_time_lat, self).fig_save()


class Fig_2D_lat_lev(Fig_2D):

    def make_template(self):
        # make_template calls method from the parent class
        super(Fig_2D_lat_lev, self).make_template('Plot 2D lat X lev',
                                                  'Ls 0-360 ', 'Lon +/-180', 'Lat', 'level[Pa/m]')
        #self.fdim1,  self.fdim2, self.Xlim,self.Ylim

    def do_plot(self):
        # Create figure
        ax = super(Fig_2D_lat_lev, self).fig_init()
        try:  # Try to create the figure, return error otherwise

            lat, pfull, var, var_info = super(
                Fig_2D_lat_lev, self).data_loader_2D(self.varfull, self.plot_type)
            super(Fig_2D_lat_lev, self).filled_contour(lat, pfull, var)

            if self.varfull2:
                _, _, var2, var_info2 = super(Fig_2D_lat_lev, self).data_loader_2D(
                    self.varfull2, self.plot_type)
                super(Fig_2D_lat_lev, self).solid_contour(
                    lat, pfull, var2, self.contour2)
                var_info += " (& "+var_info2+")"

            if self.vert_unit == 'Pa':
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.yaxis.set_major_formatter(CustomTicker())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ylabel_txt = 'Pressure [Pa]'
            else:
                ylabel_txt = 'Altitude [m]'

            if self.Xlim:
                plt.xlim(self.Xlim)
            if self.Ylim:
                plt.ylim(self.Ylim)

            super(Fig_2D_lat_lev, self).make_title(
                var_info, 'Latitude', ylabel_txt)

            ax.xaxis.set_major_locator(MultipleLocator(15))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)

            self.success = True
        except Exception as e:  # Return the error
            super(Fig_2D_lat_lev, self).exception_handler(e, ax)
        super(Fig_2D_lat_lev, self).fig_save()


class Fig_2D_lon_lev(Fig_2D):

    def make_template(self):
        # make_template calls method from the parent class
        super(Fig_2D_lon_lev, self).make_template('Plot 2D lon X lev',
                                                  'Ls 0-360 ', 'Latitude', 'Lon +/-180', 'level[Pa/m]')

    def do_plot(self):
        # Create figure
        ax = super(Fig_2D_lon_lev, self).fig_init()
        try:  # Try to create the figure, return error otherwise

            lon, pfull, var, var_info = super(
                Fig_2D_lon_lev, self).data_loader_2D(self.varfull, self.plot_type)
            lon_shift, var = shift_data(lon, var)

            super(Fig_2D_lon_lev, self).filled_contour(lon_shift, pfull, var)

            if self.varfull2:
                _, _, var2, var_info2 = super(Fig_2D_lon_lev, self).data_loader_2D(
                    self.varfull2, self.plot_type)
                _, var2 = shift_data(lon, var2)
                super(Fig_2D_lon_lev, self).solid_contour(
                    lon_shift, pfull, var2, self.contour2)
                var_info += " (& "+var_info2+")"

            if self.vert_unit == 'Pa':
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.yaxis.set_major_formatter(CustomTicker())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ylabel_txt = 'Pressure [Pa]'
            else:
                ylabel_txt = 'Altitude [m]'

            if self.Xlim:
                plt.xlim(self.Xlim)
            if self.Ylim:
                plt.ylim(self.Ylim)

            super(Fig_2D_lon_lev, self).make_title(
                var_info, 'Longitude', ylabel_txt)

            ax.xaxis.set_major_locator(MultipleLocator(30))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)

            self.success = True
        except Exception as e:  # Return the error
            super(Fig_2D_lon_lev, self).exception_handler(e, ax)
        super(Fig_2D_lon_lev, self).fig_save()


class Fig_2D_time_lev(Fig_2D):

    def make_template(self):
        # make_template calls method from the parent class
        super(Fig_2D_time_lev, self).make_template(
            'Plot 2D time X lev', 'Latitude', 'Lon +/-180', 'Ls', 'level[Pa/m]')

    def do_plot(self):
        # Create figure
        ax = super(Fig_2D_time_lev, self).fig_init()
        try:  # Try to create the figure, return error otherwise

            t_stack, pfull, var, var_info = super(
                Fig_2D_time_lev, self).data_loader_2D(self.varfull, self.plot_type)
            SolDay = t_stack[0, :]
            LsDay = t_stack[1, :]
            super(Fig_2D_time_lev, self).filled_contour(LsDay, pfull, var)

            if self.varfull2:
                _, _, var2, var_info2 = super(Fig_2D_time_lev, self).data_loader_2D(
                    self.varfull2, self.plot_type)
                super(Fig_2D_time_lev, self).solid_contour(
                    LsDay, pfull, var2, self.contour2)
                var_info += " (& "+var_info2+")"

            # Axis formatting
            if self.Xlim:
                idmin = np.argmin(np.abs(SolDay-self.Xlim[0]))
                idmax = np.argmin(np.abs(SolDay-self.Xlim[1]))
                plt.xlim([LsDay[idmin], LsDay[idmax]])
            if self.Ylim:
                plt.ylim(self.Ylim)

            Ls_ticks = [item for item in ax.get_xticks()]
            labels = [item for item in ax.get_xticklabels()]

            for i in range(0, len(Ls_ticks)):
                # Find timestep closest to this tick
                id = np.argmin(np.abs(LsDay-Ls_ticks[i]))
                if add_sol_time_axis:
                    labels[i] = '%g%s\nsol %i' % (
                        np.mod(Ls_ticks[i], 360.), degr, SolDay[id])
                else:
                    labels[i] = '%g%s' % (np.mod(Ls_ticks[i], 360.), degr)
            ax.set_xticklabels(labels, fontsize=label_size -
                               self.nPan*tick_factor, rotation=0)

            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)

            if self.vert_unit == 'Pa':
                ax.set_yscale("log")
                ax.invert_yaxis()
                ax.yaxis.set_major_formatter(CustomTicker())
                ax.yaxis.set_minor_formatter(NullFormatter())
                ylabel_txt = 'Pressure [Pa]'
            else:
                ylabel_txt = 'Altitude [m]'

            super(Fig_2D_time_lev, self).make_title(
                var_info, 'L$_s$', ylabel_txt)

            self.success = True
        except Exception as e:  # Return the error
            super(Fig_2D_time_lev, self).exception_handler(e, ax)
        super(Fig_2D_time_lev, self).fig_save()


class Fig_2D_lon_time(Fig_2D):

    def make_template(self):
        # make_template calls method from the parent class
        super(Fig_2D_lon_time, self).make_template(
            'Plot 2D lon X time', 'Latitude', 'Level [Pa/m]', 'Lon +/-180', 'Ls')

    def do_plot(self):
        # Create figure
        ax = super(Fig_2D_lon_time, self).fig_init()
        try:  # Try to create the figure, return error otherwise

            lon, t_stack, var, var_info = super(
                Fig_2D_lon_time, self).data_loader_2D(self.varfull, self.plot_type)
            lon_shift, var = shift_data(lon, var)

            SolDay = t_stack[0, :]
            LsDay = t_stack[1, :]
            super(Fig_2D_lon_time, self).filled_contour(lon_shift, LsDay, var)

            if self.varfull2:
                _, _, var2, var_info2 = super(Fig_2D_lon_time, self).data_loader_2D(
                    self.varfull2, self.plot_type)
                _, var2 = shift_data(lon, var2)
                super(Fig_2D_lon_time, self).solid_contour(
                    lon_shift, LsDay, var2, self.contour2)
                var_info += " (& "+var_info2+")"

            # Axis formatting
            if self.Xlim:
                plt.xlim(self.Xlim)

            # Axis formatting
            if self.Ylim:
                idmin = np.argmin(np.abs(SolDay-self.Ylim[0]))
                idmax = np.argmin(np.abs(SolDay-self.Ylim[1]))
                plt.ylim([LsDay[idmin], LsDay[idmax]])

            Ls_ticks = [item for item in ax.get_yticks()]
            labels = [item for item in ax.get_yticklabels()]

            for i in range(0, len(Ls_ticks)):
                # Find timestep closest to this tick
                id = np.argmin(np.abs(LsDay-Ls_ticks[i]))
                if add_sol_time_axis:
                    labels[i] = '%g%s\nsol %i' % (np.mod(Ls_ticks[i], 360.), degr, SolDay[id])
                else:
                    labels[i] = '%g%s' % (np.mod(Ls_ticks[i], 360.), degr)
            ax.set_yticklabels(labels, fontsize=label_size -
                               self.nPan*tick_factor, rotation=0)

            ax.xaxis.set_major_locator(MultipleLocator(30))
            ax.xaxis.set_minor_locator(MultipleLocator(10))

            super(Fig_2D_lon_time, self).make_title(
                var_info, 'Longitude', 'L$_s$')
            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)

            self.success = True
        except Exception as e:  # Return the error
            super(Fig_2D_lon_time, self).exception_handler(e, ax)
        super(Fig_2D_lon_time, self).fig_save()


class Fig_1D(object):
    # Parent class for 1D figure
    def __init__(self, varfull='atmos_average.ts', doPlot=True):

        self.title = None
        self.legend = None
        self.varfull = varfull
        self.t = 'AXIS'  # Default value for AXIS
        self.lat = None
        self.lon = None
        self.lev = None
        self.ftod = None  # Time of day, requested input
        self.hour = None  # Hour of day, bool, for 'diurn' plots only
        # Logic
        self.doPlot = doPlot
        self.plot_type = '1D_time'

        # Extract filetype, variable, and simulation ID (initialization only)
        self.sol_array, self.filetype, self.var, self.simuID = split_varfull(
            self.varfull)

        # Multipanel
        self.nPan = 1
        self.subID = 1
        self.addLine = False
        self.layout = None  # Page layout, e.g. [2,3], used only if 'HOLD ON 2,3' is used
        # Annotation for free dimensions
        self.fdim_txt = ''
        self.success = False
        self.vert_unit = ''  # m or Pa
        # Axis options

        self.Dlim = None  # Dimension limit
        self.Vlim = None  # Variable limit
        self.axis_opt1 = '-'

    def make_template(self):
        customFileIN.write(
            "<<<<<<<<<<<<<<| Plot 1D = {0} |>>>>>>>>>>>>>\n".format(self.doPlot))
        customFileIN.write("Title          = %s\n" % (self.title))      # 1
        customFileIN.write("Legend         = %s\n" % (self.legend))     # 2
        customFileIN.write("Main Variable  = %s\n" % (self.varfull))    # 3
        customFileIN.write("Ls 0-360       = {0}\n".format(self.t))     # 4
        customFileIN.write("Latitude       = {0}\n".format(self.lat))   # 5
        customFileIN.write("Lon +/-180     = {0}\n".format(self.lon))   # 6
        customFileIN.write("Level [Pa/m]   = {0}\n".format(self.lev))   # 7
        customFileIN.write("Diurnal  [hr]  = {0}\n".format(self.hour))  # 8
        customFileIN.write(
            "Axis Options  : lat,lon+/-180,[Pa/m],Ls = [None,None] | var = [None,None] | linestyle = - | axlabel = None \n")  # 7

    def read_template(self):
        self.title = rT('char')     # 1
        self.legend = rT('char')    # 2
        self.varfull = rT('char')   # 3
        self.t = rT('float')        # 4
        self.lat = rT('float')      # 5
        self.lon = rT('float')      # 6
        self.lev = rT('float')      # 7
        self.hour = rT('float')     # 8
        self.Dlim, self.Vlim, self.axis_opt1, self.axis_opt2, _ = read_axis_options(
            customFileIN.readline()) # 7

        self.plot_type = self.get_plot_type()

    def get_plot_type(self):
        '''
        Note that the "self.t == 'AXIS' test" and the "self.t = -88888" assignment are only used when MarsPlot
        is not passed a template.
        '''
        ncheck = 0
        graph_type = 'Error'
        if self.t == -88888 or self.t == 'AXIS':
            self.t = -88888
            graph_type = '1D_time'
            ncheck += 1
        if self.lat == -88888 or self.lat == 'AXIS':
            self.lat = -88888
            graph_type = '1D_lat'
            ncheck += 1
        if self.lon == -88888 or self.lon == 'AXIS':
            self.lon = -88888
            graph_type = '1D_lon'
            ncheck += 1
        if self.lev == -88888 or self.lev == 'AXIS':
            self.lev = -88888
            graph_type = '1D_lev'
            ncheck += 1
        if self.hour == -88888 or self.hour == 'AXIS':
            self.hour = -88888
            graph_type = '1D_diurn'
            ncheck += 1
        if ncheck == 0:
            prYellow(
                '''*** Warning *** In 1D plot, %s: use 'AXIS' to set the varying dimension ''' % (self.varfull))
        if ncheck > 1:
            prYellow(
                '''*** Warning *** In 1D plot, %s: 'AXIS' keyword can only be used once ''' % (self.varfull))
        return graph_type

    def data_loader_1D(self, varfull, plot_type):

        if not '[' in varfull:
            if '{' in varfull:
                varfull, t_req, lat_req, lon_req, lev_req, ftod_req = get_overwrite_dim_1D(
                    varfull, self.t, self.lat, self.lon, self.lev, self.ftod)
                # t_req, lat_req, lon_req, lev_req contain the dimensions to overwrite if '{}' are provided
                # otherwise, default to self.t, self.lat, self.lon, self.lev
            else:
                # No '{ }' are used to overwrite the dimensions, copy the plot defaults
                t_req, lat_req, lon_req, lev_req, ftod_req = self.t, self.lat, self.lon, self.lev, self.ftod
            sol_array, filetype, var, simuID = split_varfull(varfull)
            xdata, var, var_info = self.read_NCDF_1D(
                var, filetype, simuID, sol_array, plot_type, t_req, lat_req, lon_req, lev_req, ftod_req)

            leg_text = '%s' % (var_info)
            varlabel = '%s' % (var_info)

        else:
            VAR = []
            # Extract individual variables and prepare for execution
            varfull = remove_whitespace(varfull)
            varfull_list = get_list_varfull(varfull)
            expression_exec = create_exec(varfull, varfull_list)

            # Initialize list of requested dimensions
            t_list = [None]*len(varfull_list)
            lat_list = [None]*len(varfull_list)
            lon_list = [None]*len(varfull_list)
            lev_list = [None]*len(varfull_list)
            ftod_list = [None]*len(varfull_list)
            expression_exec = create_exec(varfull, varfull_list)

            for i in range(0, len(varfull_list)):
                # If overwriting a dimension, get the new dimension and trim 'varfull' from the '{lev=5.}' part
                if '{' in varfull_list[i]:
                    varfull_list[i], t_list[i], lat_list[i], lon_list[i], lev_list[i], ftod_list[i] = get_overwrite_dim_1D(
                        varfull_list[i], self.t, self.lat, self.lon, self.lev, self.ftod)
                else:  # No '{ }' used to overwrite the dimensions, copy the plot defaults
                    t_list[i], lat_list[i], lon_list[i], lev_list[i], ftod_list[i] = self.t, self.lat, self.lon, self.lev, self.ftod
                sol_array, filetype, var, simuID = split_varfull(
                    varfull_list[i])
                xdata, temp, var_info = self.read_NCDF_1D(
                    var, filetype, simuID, sol_array, plot_type, t_list[i], lat_list[i], lon_list[i], lev_list[i], ftod_list[i])
                VAR.append(temp)
            leg_text = '%s %s%s' % (var, var_info.split(
                " ")[-1], expression_exec.split("]")[-1])
            varlabel = '%s' % (var)
            var_info = varfull
            var = eval(expression_exec)

        return xdata, var, var_info, leg_text, varlabel

    def read_NCDF_1D(self, var_name, file_type, simuID, sol_array, plot_type, t_req, lat_req, lon_req, lev_req, ftod_req):
        '''
        Given an expression object with '[]', return the appropriate variable.
        Args:
            var_name:   variable name (e.g. 'temp')
            file_type:  MGCM output file type. Must be 'fixed' or 'average'
            sol_array:  sol if different from default (e.g. '02400')
            plot_type:  '1D-time','1D_lon', '1D_lat', '1D_lev' and '1D_time'
            t_req, lat_req, lon_req,     lev_req,          ftod_req:
             (Ls),  (lat),   (lon),  (level [Pa/m]) and (time of day) requested
        Returns:
            dim_array: the axis (e.g. an array of longitudes)
            var_array: the variable extracted
        '''

        f, var_info, dim_info, dims = prep_file(
            var_name, file_type, simuID, sol_array)

        # Get the file type ('fixed', 'diurn', 'average', 'daily') and interpolation type ('pfull', 'zstd', etc.)
        f_type, interp_type = FV3_file_type(f)

        # If self.fdim is empty, add the variable (do only once)
        add_fdim = False
        if not self.fdim_txt.strip():
            add_fdim = True

        # Initialize dimensions (These are in all the .nc files)
        lat = f.variables['lat'][:]
        lati = np.arange(0, len(lat))
        lon = f.variables['lon'][:]
        loni = np.arange(0, len(lon))

        # ------------------------Time of Day ----------------------------
        # For diurn files, we will select data on the 'time of day' axis and update the dimensions so
        # that the resulting variable is in the format of the 'average' and 'daily' files. This
        # simplifies the logic a bit so that all 'daily', 'average', and 'diurn' files are treated the
        # same when the request is 1D-time, 1D_lon, 1D_lat, and 1D_lev. Naturally, the plot type
        # '1D_diurn' will be an exeception so the following lines should be skipped if that is the case.

        # Time of day is always the 2nd dimension (i.e. dim_info[1])

        # Note: This step is performed only if the file is a 'diurn' file and the requested plot
        # is 1D_lat, 1D_lev, or 1D_time
        if (f_type == 'diurn' and dim_info[1][:11] == 'time_of_day') and not plot_type == '1D_diurn':
            tod = f.variables[dim_info[1]][:]
            todi, temp_txt = get_tod_index(ftod_req, tod)
            # Update dim_info from ('time', 'time_of_day_XX, 'lat', 'lon') to  ('time', 'lat', 'lon')
            # OR ('time', 'time_of_day_XX, 'pfull', 'lat', 'lon') to  ('time', 'pfull', 'lat', 'lon')
            dim_info = (dim_info[0],)+dim_info[2:]
            if add_fdim:
                self.fdim_txt += temp_txt

        # ====== static ======= Ignore 'level' and 'time' dimensions
        if dim_info == (u'lat', u'lon'):
            if plot_type == '1D_lat':
                loni, temp_txt = get_lon_index(lon_req, lon)
            elif plot_type == '1D_lon':
                lati, temp_txt = get_lat_index(lat_req, lat)

            if add_fdim:
                self.fdim_txt += temp_txt
            var = f.variables[var_name][lati, loni].reshape(
                len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
            f.close()
            w = area_weights_deg(var.shape, lat[lati])

            if plot_type == '1D_lat':
                return lat, mean_func(var, axis=1), var_info
            if plot_type == '1D_lon':
                return lon, np.average(var, weights=w, axis=0), var_info

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~ This Section is for 1D_time, 1D_lat, 1D_lon, and 1D_lev only ~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if not plot_type == '1D_diurn':
            # ====== time, lat, lon =======
            if dim_info == (u'time', u'lat', u'lon'):

                # Initialize dimension
                t = f.variables['time'][:]
                LsDay = np.squeeze(f.variables['areo'][:])
                ti = np.arange(0, len(t))
                # For 'diurn' file, change 'time_of_day(time, 24, 1)' to 'time_of_day(time)' at midnight UT
                if f_type == 'diurn' and len(LsDay.shape) > 1:
                    LsDay = np.squeeze(LsDay[:, 0])
                # Stack the 'time' and 'areo' arrays as one variable
                t_stack = np.vstack((t, LsDay))

                if plot_type == '1D_lat':
                    ti, temp_txt = get_time_index(t_req, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    loni, temp_txt = get_lon_index(lon_req, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                if plot_type == '1D_lon':
                    lati, temp_txt = get_lat_index(lat_req, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    ti, temp_txt = get_time_index(t_req, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                if plot_type == '1D_time':
                    loni, temp_txt = get_lon_index(lon_req, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    lati, temp_txt = get_lat_index(lat_req, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt

                if f_type == 'diurn':
                    var = f.variables[var_name][ti, todi, lati, loni].reshape(len(np.atleast_1d(ti)), len(np.atleast_1d(todi)),
                                                                              len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
                    var = mean_func(var, axis=1)
                else:
                    var = f.variables[var_name][ti, lati, loni].reshape(
                        len(np.atleast_1d(ti)), len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))

                f.close()

                w = area_weights_deg(var.shape, lat[lati])

                # Return data
                if plot_type == '1D_lat':
                    return lat,    mean_func(mean_func(var, axis=2), axis=0), var_info
                if plot_type == '1D_lon':
                    return lon,    mean_func(np.average(var, weights=w, axis=1), axis=0), var_info
                if plot_type == '1D_time':
                    return t_stack, mean_func(np.average(var, weights=w, axis=1), axis=1), var_info

            # ====== time, level, lat, lon =======
            if (dim_info == (u'time', u'pfull', u'lat', u'lon')
                or dim_info == (u'time', u'level', u'lat', u'lon')
                or dim_info == (u'time', u'pstd', u'lat', u'lon')
                or dim_info == (u'time', u'zstd', u'lat', u'lon')
                or dim_info == (u'time', u'zagl', u'lat', u'lon')
                    or dim_info == (u'time', u'zgrid', u'lat', u'lon')):

                if dim_info[1] in ['pfull', 'level', 'pstd']:
                    self.vert_unit = 'Pa'
                if dim_info[1] in ['zagl', 'zstd', 'zgrid']:
                    self.vert_unit = 'm'

                # Initialize dimensions
                levs = f.variables[dim_info[1]][:]
                zi = np.arange(0, len(levs))
                t = f.variables['time'][:]
                LsDay = np.squeeze(f.variables['areo'][:])
                ti = np.arange(0, len(t))
                # For 'diurn' file, change 'time_of_day(time, 24, 1)' to 'time_of_day(time)' at midnight UT
                if f_type == 'diurn' and len(LsDay.shape) > 1:
                    LsDay = np.squeeze(LsDay[:, 0])
                # Stack the 'time' and 'areo' arrays as one variable
                t_stack = np.vstack((t, LsDay))

                if plot_type == '1D_lat':
                    ti, temp_txt = get_time_index(t_req, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    loni, temp_txt = get_lon_index(lon_req, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    zi, temp_txt = get_level_index(lev_req, levs)
                    if add_fdim:
                        self.fdim_txt += temp_txt

                if plot_type == '1D_lon':
                    lati, temp_txt = get_lat_index(lat_req, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    ti, temp_txt = get_time_index(t_req, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    zi, temp_txt = get_level_index(lev_req, levs)
                    if add_fdim:
                        self.fdim_txt += temp_txt

                if plot_type == '1D_time':
                    loni, temp_txt = get_lon_index(lon_req, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    lati, temp_txt = get_lat_index(lat_req, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    zi, temp_txt = get_level_index(lev_req, levs)
                    if add_fdim:
                        self.fdim_txt += temp_txt

                if plot_type == '1D_lev':
                    ti, temp_txt = get_time_index(t_req, LsDay)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    lati, temp_txt = get_lat_index(lat_req, lat)
                    if add_fdim:
                        self.fdim_txt += temp_txt
                    loni, temp_txt = get_lon_index(lon_req, lon)
                    if add_fdim:
                        self.fdim_txt += temp_txt

                # Fix for new netcdf4 version: Get array elements instead of manipulating the variable
                # It used to be that 'var = f.variables[var_name]'

                # If 'diurn', do the 'time of day' average first
                if f_type == 'diurn':
                    var = f.variables[var_name][ti, todi, zi, lati, loni].reshape(len(np.atleast_1d(ti)), len(np.atleast_1d(todi)),
                                                                                  len(np.atleast_1d(zi)), len(np.atleast_1d(lati)), len(np.atleast_1d(loni)))
                    var = mean_func(var, axis=1)
                else:
                    reshape_shape = [len(np.atleast_1d(ti)),
                                     len(np.atleast_1d(zi)),
                                     len(np.atleast_1d(lati)),
                                     len(np.atleast_1d(loni))]
                    var = f.variables[var_name][ti, zi,
                                                lati, loni].reshape(reshape_shape)
                f.close()

                w = area_weights_deg(var.shape, lat[lati])

                #(u'time', u'pfull', u'lat', u'lon')
                if plot_type == '1D_lat':
                    return lat,    mean_func(mean_func(mean_func(var, axis=3), axis=1), axis=0), var_info
                if plot_type == '1D_lon':
                    return lon,    mean_func(mean_func(np.average(var, weights=w, axis=2), axis=1), axis=0), var_info
                if plot_type == '1D_time':
                    return t_stack, mean_func(mean_func(np.average(var, weights=w, axis=2), axis=2), axis=1), var_info
                if plot_type == '1D_lev':
                    return levs,   mean_func(mean_func(np.average(var, weights=w, axis=2), axis=2), axis=0), var_info

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~ This Section is for 1D_diurn only ~~~~~~~~~~~~~~~~~~~
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        else:
            # Find name of 'time of day' variable (i.e. 'time_of_day_16' or 'time_of_day_24')
            tod_dim_name = find_tod_in_diurn(f)
            tod = f.variables[tod_dim_name][:]
            todi = np.arange(0, len(tod))

            # ====== time, lat, lon =======
            if f.variables[var_name].dimensions == ('time', tod_dim_name, 'lat', 'lon'):

                # Initialize dimension
                t = f.variables['time'][:]
                LsDay = np.squeeze(f.variables['areo'][:])
                ti = np.arange(0, len(t))
                # For 'diurn' file, change 'time_of_day(time, 24, 1)' to 'time_of_day(time)' at midnight UT
                if f_type == 'diurn' and len(LsDay.shape) > 1:
                    LsDay = np.squeeze(LsDay[:, 0])
                # Stack the 'time' and 'areo' arrays as one variable
                t_stack = np.vstack((t, LsDay))

                loni, temp_txt = get_lon_index(lon_req, lon)
                if add_fdim:
                    self.fdim_txt += temp_txt
                lati, temp_txt = get_lat_index(lat_req, lat)
                if add_fdim:
                    self.fdim_txt += temp_txt
                ti, temp_txt = get_time_index(t_req, LsDay)
                if add_fdim:
                    self.fdim_txt += temp_txt

                reshape_shape = [len(np.atleast_1d(ti)), len(np.atleast_1d(tod)),
                                 len(np.atleast_1d(lati)), len(np.atleast_1d(loni))]

                # Broadcast dimensions before extraction. This is a 'new' requirement for numpy
                var = f.variables[var_name][ti, :,
                                            lati, loni].reshape(reshape_shape)
                f.close()

                w = area_weights_deg(var.shape, lat[lati])
                # Return data
                #('time','time_of_day','lat', u'lon')
                return tod, mean_func(mean_func(np.average(var, weights=w, axis=2), axis=2), axis=0), var_info

            # ====== time, level, lat, lon =======
            if (dim_info == ('time', tod_dim_name, 'pfull', 'lat', 'lon')
                or dim_info == ('time', tod_dim_name, 'level', 'lat', 'lon')
                or dim_info == ('time', tod_dim_name, 'pstd', 'lat', 'lon')
                or dim_info == ('time', tod_dim_name, 'zstd', 'lat', 'lon')
                or dim_info == ('time', tod_dim_name, 'zagl', 'lat', 'lon')
                    or dim_info == ('time', tod_dim_name, 'zgrid', 'lat', 'lon')):

                if dim_info[1] in ['pfull', 'level', 'pstd']:
                    self.vert_unit = 'Pa'
                if dim_info[1] in ['zagl', 'zstd', 'zgrid']:
                    self.vert_unit = 'm'

                # Initialize dimensions
                levs = f.variables[dim_info[2]][:]

                t = f.variables['time'][:]
                LsDay = np.squeeze(f.variables['areo'][:])
                ti = np.arange(0, len(t))
                # For 'diurn' file, change 'time_of_day(time, 24, 1)' to 'time_of_day(time)' at midnight UT
                if f_type == 'diurn' and len(LsDay.shape) > 1:
                    LsDay = np.squeeze(LsDay[:, 0])
                # Stack the 'time' and 'areo' arrays as one variable
                t_stack = np.vstack((t, LsDay))

                ti, temp_txt = get_time_index(t_req, LsDay)
                if add_fdim:
                    self.fdim_txt += temp_txt
                lati, temp_txt = get_lat_index(lat_req, lat)
                if add_fdim:
                    self.fdim_txt += temp_txt
                loni, temp_txt = get_lon_index(lon_req, lon)
                if add_fdim:
                    self.fdim_txt += temp_txt
                zi, temp_txt = get_level_index(lev_req, levs)
                if add_fdim:
                    self.fdim_txt += temp_txt

                reshape_shape = [len(np.atleast_1d(ti)), len(np.atleast_1d(tod)), len(np.atleast_1d(zi)),
                                 len(np.atleast_1d(lati)), len(np.atleast_1d(loni))]

                var = f.variables[var_name][ti, :, zi,
                                            lati, loni].reshape(reshape_shape)
                f.close()

                w = area_weights_deg(var.shape, lat[lati])

                #('time','time_of_day', 'pfull', 'lat', 'lon')

                return tod,   mean_func(mean_func(mean_func(np.average(var, weights=w, axis=3), axis=3), axis=2), axis=0), var_info

    def exception_handler(self, e, ax):
        if debug:
            raise
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")
        prYellow('*** Warning *** Attempting %s profile for %s: %s' %
                 (self.plot_type, self.varfull, str(e)))
        ax.text(0.5, 0.5, 'ERROR:'+str(e), horizontalalignment='center', verticalalignment='center',
                bbox=dict(boxstyle="round", ec=(
                    1., 0.5, 0.5), fc=(1., 0.8, 0.8),),
                transform=ax.transAxes, wrap=True, fontsize=16)

    def fig_init(self):
        # Create figure
        if self.layout is None:  # No layout specified
            out = fig_layout(self.subID, self.nPan, vertical_page)
        else:
            out = np.append(self.layout, self.subID)

        if self.subID == 1 and not self.addLine:
            fig = plt.figure(facecolor='white', figsize=(
                width_inch, height_inch))  # Create figure if first panel
        if not self.addLine:
            ax = plt.subplot(out[0], out[1], out[2])  # nrow, ncol, subID
        else:

            ax = plt.gca()

        return ax

    def fig_save(self):

        # Save the figure
        if self.subID == self.nPan:  # Last subplot
            if self.subID == 1:      # If 1 plot
                if not '[' in self.varfull:
                    # Add split '{' if 'varfull' contains layer. Does not do anything otherwise
                    sensitive_name = self.varfull.split('{')[0].strip()
                else:
                    sensitive_name = 'expression_' + \
                        get_list_varfull(self.varfull)[0].split('{')[0].strip()
            else:  # Multipanel
                sensitive_name = 'multi_panel'

            self.fig_name = output_path+'/plots/'+sensitive_name+'.'+out_format
            self.fig_name = create_name(self.fig_name)

            if i_list < len(objectList)-1 and not objectList[i_list+1].addLine:
                plt.savefig(self.fig_name, dpi=my_dpi)
                if out_format != "pdf":
                    print("Saved:" + self.fig_name)
            # Last subplot
            if i_list == len(objectList)-1:
                plt.savefig(self.fig_name, dpi=my_dpi)
                if out_format != "pdf":
                    print("Saved:" + self.fig_name)

    def do_plot(self):
        # Create figure
        ax = self.fig_init()

        try:
            # Try to create the figure, return error otherwise
            xdata, var, var_info, leg_text, varlabel = self.data_loader_1D(
                self.varfull, self.plot_type)

            if self.legend:
                txt_label = self.legend
            else:
                # txt_label=var_info+'\n'+self.fdim_txt[1:] # Remove the first comma in fdim_txt to print to the new line
                # ============ CB vvvv
                if self.nPan > 1:
                    txt_label = leg_text
                else:
                    # txt_label=None
                    # Remove the first comma in fdim_txt to print to the new line
                    txt_label = var_info+'\n'+self.fdim_txt[1:]

            if self.title:
                if '{' in self.title:
                    fs = int(remove_whitespace(
                        (self.title).split("=")[1].split("}")[0]))
                    title_text = ((self.title).split("{")[0])
                    plt.title(title_text, fontsize=fs -
                              self.nPan*title_factor, wrap=False)
                else:
                    plt.title((self.title), fontsize=title_size -
                              self.nPan*title_factor)
            else:
                plt.title(
                    var_info+'\n'+self.fdim_txt[1:], fontsize=title_size-self.nPan*title_factor, wrap=False)
                # ============ CB ^^^^

            if self.plot_type == '1D_lat':

                plt.plot(var, xdata, self.axis_opt1,
                         lw=3, ms=7, label=txt_label)
                plt.ylabel('Latitude', fontsize=label_size -
                           self.nPan*label_factor)
                # Label is provided
                if self.axis_opt2:
                    plt.xlabel(self.axis_opt2, fontsize=label_size -
                               self.nPan*label_factor)
                else:
                    plt.xlabel(varlabel, fontsize=label_size -
                               self.nPan*label_factor)

                ax.yaxis.set_major_locator(MultipleLocator(15))
                ax.yaxis.set_minor_locator(MultipleLocator(5))
                if self.Dlim:
                    plt.ylim(self.Dlim)
                if self.Vlim:
                    plt.xlim(self.Vlim)

            if self.plot_type == '1D_lon':
                lon_shift, var = shift_data(xdata, var)

                plt.plot(lon_shift, var, self.axis_opt1,
                         lw=3, ms=7, label=txt_label)
                plt.xlabel('Longitude', fontsize=label_size -
                           self.nPan*label_factor)
                # Label is provided
                if self.axis_opt2:
                    plt.ylabel(self.axis_opt2, fontsize=label_size -
                               self.nPan*label_factor)
                else:
                    plt.ylabel(varlabel, fontsize=label_size -
                               self.nPan*label_factor)

                ax.xaxis.set_major_locator(MultipleLocator(30))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                if self.Dlim:
                    plt.xlim(self.Dlim)
                if self.Vlim:
                    plt.ylim(self.Vlim)

            if self.plot_type == '1D_time':
                SolDay = xdata[0, :]
                LsDay = xdata[1, :]
                # If simulations span different years, they can be stacked (overplotted)
                if parser.parse_args().stack_year:
                    LsDay = np.mod(LsDay, 360)

                plt.plot(LsDay, var, self.axis_opt1, lw=3, ms=7, label=txt_label)

                # Label is provided
                if self.axis_opt2:
                    plt.ylabel(self.axis_opt2, fontsize=label_size -
                               self.nPan*label_factor)
                else:
                    plt.ylabel(varlabel, fontsize=label_size -
                               self.nPan*label_factor)

                # Axis formatting
                if self.Vlim:
                    plt.ylim(self.Vlim)

                if self.Dlim:
                    plt.xlim(self.Dlim)  # TODO

                Ls_ticks = [item for item in ax.get_xticks()]
                labels = [item for item in ax.get_xticklabels()]

                for i in range(0, len(Ls_ticks)):
                    # Find timestep closest to this tick
                    id = np.argmin(np.abs(LsDay-Ls_ticks[i]))
                    if add_sol_time_axis:
                        labels[i] = '%g%s\nsol %i' % (
                            np.mod(Ls_ticks[i], 360.), degr, SolDay[id])
                    else:
                        labels[i] = '%g%s' % (np.mod(Ls_ticks[i], 360.), degr)
                ax.set_xticklabels(labels, fontsize=label_size -
                                   self.nPan*tick_factor, rotation=0)

            if self.plot_type == '1D_lev':

                plt.plot(var, xdata, self.axis_opt1,
                         lw=3, ms=7, label=txt_label)

                # Label is provided
                if self.axis_opt2:
                    plt.xlabel(self.axis_opt2, fontsize=label_size -
                               self.nPan*label_factor)
                else:
                    plt.xlabel(varlabel, fontsize=label_size -
                               self.nPan*label_factor)

                if self.vert_unit == 'Pa':
                    ax.set_yscale("log")
                    ax.invert_yaxis()
                    ax.yaxis.set_major_formatter(CustomTicker())
                    ax.yaxis.set_minor_formatter(NullFormatter())
                    ylabel_txt = 'Pressure [Pa]'
                else:
                    ylabel_txt = 'Altitude [m]'

                plt.ylabel(ylabel_txt, fontsize=label_size -
                           self.nPan*label_factor)

                if self.Dlim:
                    plt.ylim(self.Dlim)
                if self.Vlim:
                    plt.xlim(self.Vlim)

            if self.plot_type == '1D_diurn':
                plt.plot(xdata, var, self.axis_opt1,
                         lw=3, ms=7, label=txt_label)
                plt.xlabel('Time [hr]', fontsize=label_size -
                           self.nPan*label_factor)

                # Label is provided
                if self.axis_opt2:
                    plt.ylabel(self.axis_opt2, fontsize=label_size -
                               self.nPan*label_factor)
                else:
                    plt.ylabel(varlabel, fontsize=label_size -
                               self.nPan*label_factor)

                ax.xaxis.set_major_locator(MultipleLocator(4))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                # Default: set X dim to 0-24. Can be overwritten
                plt.xlim([0, 24])

                # Axis formatting
                if self.Dlim:
                    plt.xlim(self.Dlim)
                if self.Vlim:
                    plt.ylim(self.Vlim)

            # ==== Common labeling ====
            plt.xticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.yticks(fontsize=label_size-self.nPan*tick_factor, rotation=0)
            plt.legend(fontsize=title_size-self.nPan*title_factor)
            plt.grid(True)

            self.success = True
        except Exception as e:  # Return the error
            self.exception_handler(e, ax)
        self.fig_save()


# ======================================================
#                  END OF PROGRAM
# ======================================================

if __name__ == '__main__':
    main()
