import os
import sys
import subprocess
from netCDF4 import Dataset, MFDataset
import numpy as np
import re
#=========================================================================
#=========================Scripts utilities===============================
#=========================================================================


# The functions below allow to print in different color
def prRed(skk): print("\033[91m{}\033[00m".format(skk))
def prGreen(skk): print("\033[92m{}\033[00m".format(skk))
def prCyan(skk): print("\033[96m{}\033[00m".format(skk))
def prYellow(skk): print("\033[93m{}\033[00m".format(skk))
def prPurple(skk): print("\033[95m{}\033[00m".format(skk))
def prLightPurple(skk): print("\033[94m{}\033[00m".format(skk))

def MY_func(Ls_cont):
    '''
    This function return the Mars Year
    Args:
        Ls_cont: solar longitude, contineuous
    Returns:
        MY : int the Mars year
    '''
    return (Ls_cont)//(360.)+1

def find_tod_in_diurn(fNcdf):
    '''
    Return the variable for the local time axis in diurn files.
    Original implementation by Victoria H.
    Args:
        fNcdf: an (open) Netcdf file object
    Return:
        tod (string): 'time_of_day_16'or 'time_of_day_24'
    '''
    regex=re.compile('time_of_day.')
    varset=fNcdf.variables.keys()
    return [string for string in varset if re.match(regex, string)][0] #Extract the 1st element of the list



def print_fileContent(fileNcdf):
    '''
    Print the content of a Netcdf file in a compact format. Variables are sorted by dimensions.
    Args:
        fileNcdf: full path to netcdf file
    Returns:
        None (print in the terminal)
    '''
    #Define Colors for printing
    def Green(skk): return"\033[92m{}\033[00m".format(skk)
    def Cyan(skk): return "\033[96m{}\033[00m".format(skk)
    def Yellow(skk):return"\033[93m{}\033[00m".format(skk)
    def Purple(skk):return"\033[95m{}\033[00m".format(skk)
    if not os.path.isfile(fileNcdf):
        print(fileNcdf+' not found')
    else:
        f=Dataset(fileNcdf, 'r')
        print("===================DIMENSIONS==========================")
        print(list(f.dimensions.keys()))
        print(str(f.dimensions))
        print("====================CONTENT==========================")
        all_var=f.variables.keys() #get all variables
        all_dims=list() #initialize empty list
        for ivar in all_var:
            all_dims.append(f.variables[ivar].dimensions) #get all the variables dimensions
        all_dims=set(all_dims) #filter duplicates (an object of type set() is an unordered collections of distinct objects
        all_dims=sorted(all_dims,key=len) #sort dimensions by lenght, e.g ('lat') will come before ('lat','lon')
        var_done=list()
        for idim in all_dims:
            for ivar in all_var:
                if f.variables[ivar].dimensions==idim :
                    txt_dim=getattr(f.variables[ivar],'dimensions','')
                    txt_shape=getattr(f.variables[ivar],'shape','')
                    txt_long_name=getattr(f.variables[ivar],'long_name','')
                    txt_units=getattr(f.variables[ivar],'units','')
                    print(Green(ivar.ljust(15))+': '+Purple(txt_dim)+'= '+Cyan(txt_shape)+', '+Yellow(txt_long_name)+\
                    '  ['+txt_units+']')

        try: #This part will be skipped if  the netcdf file does not contains a 'time' variable
            t_ini=f.variables['time'][0];t_end=f.variables['time'][-1]
            Ls_ini=np.squeeze(f.variables['areo'])[0];Ls_end=np.squeeze(f.variables['areo'])[-1]
            MY_ini=MY_func(Ls_ini);MY_end=MY_func(Ls_end)
            print('')
            print('Ls ranging from %6.2f to %6.2f: %.2f days'%(np.mod(Ls_ini,360.),np.mod(Ls_end,360.),t_end-t_ini))
            print('               (MY %02i)   (MY %02i)'%(MY_ini,MY_end))
        except:
            pass
        f.close()
        print("=====================================================")

def print_varContent(fileNcdf,list_varfull,print_stat=False):
    '''
    Print the content of a variable inside a Netcdf file
    This test is based on the existence of a least one  00XXX.fixed.nc in the current directory.
    Args:
        fileNcdf:      full path to netcdf file
        list_varfull:  list of variable names and optional slices, e.g ['lon' ,'ps[:,10,20]']
        print_stat:  if true, print min, mean and max instead of values
    Returns:
        None (print in the terminal)
    '''
    #Define Colors for printing
    def Cyan(skk): return "\033[96m{}\033[00m".format(skk)
    def Red(skk):  return "\033[91m{}\033[00m".format(skk)
    if not os.path.isfile(fileNcdf):
        print(fileNcdf+' not found')
    else:

        if print_stat:
            print(Cyan('__________________________________________________________________________'))
            print(Cyan('           VAR            |      MIN      |      MEAN     |      MAX      |'))
            print(Cyan('__________________________|_______________|_______________|_______________|'))
        for varfull in list_varfull:
            try:
                slice='[:]'
                if '[' in varfull:
                    varname,slice=varfull.strip().split('[');slice='['+slice
                else:
                    varname=varfull.strip()
                cmd_txt="""f.variables['"""+varname+"""']"""+slice
                f=Dataset(fileNcdf, 'r')
                var=eval(cmd_txt)

                if print_stat:
                    Min=np.nanmin(var)
                    Mean=np.nanmean(var)
                    Max=np.nanmax(var)
                    print(Cyan('%26s|%15g|%15g|%15g|'%(varfull,Min,Mean,Max)))
                    if varname=='areo':
                        # If variable is areo, also print the modulo
                        print(Cyan('%17s(mod 360)|(%13g)|(%13g)|(%13g)|'%(varfull,np.nanmin(var%360),np.nanmean(var%360),np.nanmax(var%360))))
                else:
                    if varname!='areo':
                        print(Cyan(varfull+'= '))
                        print(Cyan(var))
                    else:
                        #Special case for areo, also print modulo
                        print(Cyan('areo (areo mod 360)='))
                        for ii in var: print(ii,ii%360)

                    print(Cyan('______________________________________________________________________'))
            except:
                if print_stat:
                    print(Red('%26s|%15s|%15s|%15s|'%(varfull,'','','')))
                else:
                    print(Red(varfull))
        #Last line for the table
        if print_stat:
            print(Cyan('__________________________|_______________|_______________|_______________|'))
        f.close()





def give_permission(filename):
    '''
    # NAS system only: set group permission to the file
    '''
    import subprocess
    import os

    try:
        subprocess.check_call(['setfacl -v'],shell=True,stdout=open(os.devnull, "w"),stderr=open(os.devnull, "w")) #catch error and standard output
        cmd_txt='setfacl -R -m g:s0846:r '+filename
        subprocess.call(cmd_txt,shell=True)
    except subprocess.CalledProcessError:
        pass

def check_file_tape(fileNcdf,abort=False):
    '''
    Relevant for use on the NASA Advanced Supercomputing (NAS) environnment only
    Check if a file is present on the disk by running the NAS dmls -l data migration command.
    This avoid the program to stall if the files need to be migrated from the disk to the tape
    Args:
        fileNcdf: full path to netcdf file
        exit: boolean. If True, exit the program (avoid stalling the program if file is not on disk)
    Returns:
        None (print status and abort program)
    '''
    # If the filename provided is not a netcdf file, exit program right away
    if fileNcdf[-3:]!='.nc':
        prRed('*** Error ***')
        prRed(fileNcdf + ' is not a netcdf file \n' )
        exit()
    #== Then check if the file actually exists on the system,  exit otherwise.


    try:
        #== NAS system only: file exists, check if it is active on disk or needs to be migrated from Lou
        subprocess.check_call(["dmls"],shell=True,stdout=open(os.devnull, "w"),stderr=open(os.devnull, "w")) #check if dmls command is available (NAS systems only)
        cmd_txt='dmls -l '+fileNcdf+"""| awk '{print $8,$9}'""" #get the last columns of the ls command with filename and status
        dmls_out=subprocess.check_output(cmd_txt,shell=True).decode('utf-8')  # get 3 letter identifier from dmls -l command, convert byte to string for Python 3
        if dmls_out[1:4] not in ['DUL','REG','MIG']: #file is OFFLINE, UNMIGRATING etc...
            if abort :
                prRed('*** Error ***')
                print(dmls_out)
                prRed(dmls_out[6:-1]+ ' is not available on disk, status is: '+dmls_out[0:5])
                prRed('CHECK file status with  dmls -l *.nc and run  dmget *.nc to migrate the files')
                prRed('Exiting now... \n')
                exit()
            else:
                prYellow('*** Warning ***')
                prYellow(dmls_out[6:-1]+ ' is not available on disk, status is: '+dmls_out[0:5])
                prYellow('Consider checking file status with  dmls -l *.nc and run  dmget *.nc to migrate the files')
                prYellow('Waiting for file to be migrated to disk, this may take a while...')
    except subprocess.CalledProcessError: #subprocess.check_call return an eror message
        if abort :
             exit()
        else:
            pass


def get_Ncdf_path(fNcdf):
    '''
    Return the full path of a Netcdf object.
    Note that 'Dataset' and multi-files dataset (i.e. 'MFDataset') have different
    attributes for the path, hence the need for this function.
    Args:
        fNcdf : Dataset or  MFDataset object
    Returns :
        Path: string (list) for Dataset (MFDataset)
    '''
    fname_out=getattr(fNcdf,'_files',False) #Only MFDataset has the_files attribute
    if not fname_out: fname_out=getattr(fNcdf,'filepath')() #Regular Dataset
    return fname_out

def extract_path_basename(filename):
    '''
    Return the path and basename of a file. If only the filename is provided, assume it is the current directory
    Args:
        filename: e.g. 'XXXXX.fixed.nc', './history/XXXXX.fixed.nc' or '/home/user/history/XXXXX.fixed.nc'
    Returns:
        filepath : '/home/user/history/XXXXX.fixed.nc' in all the cases above
        basename:   XXXXX.fixed.nc in all the cases above

    ***NOTE***
    This routine does not check for file existence and only operates on the provided input string.
    '''
    #Get the filename without the path
    if '/' in filename or '\\' in filename :
        filepath,basename=os.path.split(filename)
    else:
        filepath=os.getcwd()
        basename=    filename

    # Is the home ('~') symbol is included, expend the user path
    if '~' in filepath:filepath=  os.path.expanduser(filepath)
    return filepath,basename

def FV3_file_type(fNcdf):
    '''
    Return the type of output files:
    Args:
        fNcdf: an (open) Netcdf file object
    Return:
       f_type (string): 'fixed', 'contineous', or 'diurn'
       interp_type (string): 'pfull','pstd','zstd','zagl','zgrid'
    '''
    #Get the full path from the file
    fullpath=get_Ncdf_path(fNcdf)
    #If MFDataset, get the 1st file in the list
    if type(fullpath)==list:fullpath=fullpath[0]

    #Get the filename without the path
    _,filename=os.path.split(fullpath)

    #Initialize, assume the file is contineuous
    f_type='contineous'
    interp_type='unknown'
    tod_name='n/a'

    model=read_variable_dict_amescap_profile(fNcdf)

    #If 'time' is not a dimension, assume it is a 'fixed' file
    if model.time not in fNcdf.dimensions.keys():f_type='fixed'

    #If 'tod_name_XX' is present as a dimension, it is a diurn file (this is robust)
    try:
        tod_name=find_tod_in_diurn(fNcdf)
        if tod_name in fNcdf.dimensions.keys():f_type='diurn'
    except:
        pass

    dims=fNcdf.dimensions.keys()
    if model.pfull in dims: interp_type=model.pfull
    if model.pstd  in dims: interp_type=model.pstd
    if model.zstd  in dims: interp_type=model.zstd
    if model.zagl  in dims: interp_type=model.zagl
    if model.zgrid in dims: interp_type=model.zgrid
    return f_type,interp_type

def alt_FV3path(fullpaths,alt,test_exist=True):
    '''
    Internal function. given an interpolated daily, diurn or average file
    return the raw or fixed file. Accept string or list as input
    Args:
        fullpaths : e.g '/u/path/00010.atmos_average_pstd.nc' or LIST
            alt: alternative type 'raw' or 'fixed'
            test_exist=True test file existence on disk
    Returns :
        Alternative path to raw or fixed file, e.g.
                    '/u/path/00010.atmos_average.nc'
                    '/u/path/00010.fixed.nc'
    '''
    out_list=[]
    one_element=False
    #Convert as list for generality
    if type(fullpaths)==str:
        one_element=True
        fullpaths=[fullpaths]

    for fullpath in fullpaths:
        path, filename = os.path.split(fullpath)
        DDDDD=filename.split('.')[0] #Get the date
        ext=filename[-8:] #Get the extension
        #This is an interpolated file
        if alt=='raw':
            if ext in ['_pstd.nc','_zstd.nc','_zagl.nc','plevs.nc']:
                if ext =='plevs.nc':
                    file_raw=filename[0:-9]+'.nc'
                else:
                    file_raw=filename[0:-8]+'.nc'
            else:
                raise ValueError('In alt_FV3path(), FV3 file %s not recognized'%(filename))
            new_full_path=path+'/'+file_raw
        if alt=='fixed':
            new_full_path=path+'/'+DDDDD+'.fixed.nc'
        if test_exist and not (os.path.exists(new_full_path)):
            raise ValueError('In alt_FV3path() %s does not exist '%(new_full_path))

        out_list.append(new_full_path)
    if one_element:out_list=out_list[0]
    return  out_list

def smart_reader(fNcdf,var_list,suppress_warning=False):
    """
    Smarter alternative to using var=fNcdf.variables['var'][:] when handling PROCESSED files that also check
    matching XXXXX.atmos_average.nc (or daily...) and XXXXX.fixed.nc files

    Args:
        fNcdf: Netcdf file object (i.e. already opened with Dataset or MFDataset)
        var_list: variable or list of variables, e.g 'areo' or ['pk','bk','areo']
        suppress_warning: Suppress debug statement, useful if variable is not expected to be found in the file anyway
    Returns:
        out_list: variables content as singleton or values to unpack

    -------
    Example:

    from netCDF4 import Dataset

    fNcdf=Dataset('/u/akling/FV3/00668.atmos_average_pstd.nc','r')

    ucomp= fNcdf.variables['ucomp'][:]   # << this is the regular way
    vcomp= smart_reader(fNcdf,'vcomp')   # << this is exacly equivalent
    pk,bk,areo= smart_reader(fNcdf,['pk','bk','areo'])  # this will get 'areo' from 00668.atmos_average.nc is not available in the original _pstd.nc file
                                                        # if pk and bk are absent from 0668.atmos_average.nc, it will also check 00668.fixed.n
    *** NOTE ***
        -Only the variables' content is returned, not the attributes
    """

    #This out_list is for the variable
    out_list=[]
    one_element=False
    file_is_MF=False

    Ncdf_path= get_Ncdf_path(fNcdf) #Return string (Dataset) or list (MFDataset)
    if type(Ncdf_path)==list:file_is_MF=True

    #For generality convert to list if only one variable is provided, e.g 'areo'>['areo']
    if type(var_list)==str:
        one_element=True
        var_list=[var_list]

    for ivar in var_list:
    #First try to read in the original file
        if ivar in fNcdf.variables.keys():
            out_list.append(fNcdf.variables[ivar][:])
        else:
            full_path_try=alt_FV3path(Ncdf_path,alt='raw',test_exist=True)
            if file_is_MF:
                f_tmp=MFDataset(full_path_try,'r')
            else:
                f_tmp=Dataset(full_path_try,'r')

            if ivar in f_tmp.variables.keys():
                out_list.append(f_tmp.variables[ivar][:])
                if not suppress_warning: print('**Warning*** Using variable %s in %s instead of original file(s)'%(ivar,full_path_try))
                f_tmp.close()
            else:
                f_tmp.close()
                full_path_try=alt_FV3path(Ncdf_path,alt='fixed',test_exist=True)
                if file_is_MF:full_path_try=full_path_try[0]

                f_tmp=Dataset(full_path_try,'r')
                if ivar in f_tmp.variables.keys():
                    out_list.append(f_tmp.variables[ivar][:])
                    f_tmp.close()
                    if not suppress_warning: print('**Warning*** Using variable %s in %s instead of original file(s)'%(ivar,full_path_try))
                else:
                    print('***ERROR*** Variable %s not found in %s, NOR in raw output or fixed file'%(ivar,full_path_try))
                    print('            >>> Assigning  %s  to NaN'%(ivar))
                    f_tmp.close()
                    out_list.append(np.NaN)
    if one_element:out_list=out_list[0]
    return out_list


def regrid_Ncfile(VAR_Ncdf,file_Nc_in,file_Nc_target):
    '''
    Regrid a Ncdf variable from one file's structure to match another file  [Alex Kling , May 2021]
    Args:
        VAR_Ncdf: A netCDF4 variable OBJECT, e.g. 'f_in.variables['temp']' from the source file
        file_Nc_in: The opened netcdf file object  for that input variable, e.g f_in=Dataset('fname','r')
        file_Nc_target: An opened netcdf file object  for the target grid t e.g f_out=Dataset('fname','r')
    Returns:
        VAR_OUT: the VALUES of VAR_Ncdf[:], interpolated on the grid for the target file.

    *** Note***
    While the KDTree interpolation can handle a 3D dataset (lon/lat/lev instead of just 2D lon/lat) , the grid points in the vertical are just a few 10's -100's meter in the PBL vs few 10'-100's km in the horizontal. This would results in excessive weighting in the vertical, which is why the vertical dimension is handled separately.
    '''
    from amescap.FV3_utils import interp_KDTree, axis_interp
    ftype_in,zaxis_in=FV3_file_type(file_Nc_in)
    ftype_t,zaxis_t=FV3_file_type(file_Nc_target)

    #Sanity check

    if ftype_in !=ftype_t:
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must have the same type"""%(ftype_in,ftype_t))

    if zaxis_in!=zaxis_t:
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must have the same vertical grid"""%(zaxis_in,zaxis_t))

    if zaxis_in=='pfull' or zaxis_t=='pfull':
        print("""*** Warning*** in regrid_Ncfile, input file  '%s' and target file '%s' must be vertically interpolated"""%(zaxis_in,zaxis_t))


    #===Get target dimensions===
    lon_t=file_Nc_target.variables['lon'][:]
    lat_t=file_Nc_target.variables['lat'][:]
    if 'time' in VAR_Ncdf.dimensions:
        areo_t=file_Nc_target.variables['areo'][:]
        time_t=file_Nc_target.variables['time'][:]

    #===Get input dimensions===
    lon_in=file_Nc_in.variables['lon'][:]
    lat_in=file_Nc_in.variables['lat'][:]
    if 'time' in VAR_Ncdf.dimensions:
        areo_in=file_Nc_in.variables['areo'][:]
        time_in=file_Nc_in.variables['time'][:]

    #Get array elements
    var_OUT=VAR_Ncdf[:]

    #STEP 1: Lat/lon interpolation are always performed unless target lon and lat are identical
    if not (np.all(lat_in==lat_t) and np.all(lon_in==lon_t)) :
        #Special case if input longitudes is 1 element (slice or zonal average). We only interpolate on the latitude axis
        if len(np.atleast_1d(lon_in))==1:
            var_OUT=axis_interp(var_OUT, lat_in,lat_t,axis=-2, reverse_input=False, type_int='lin')
        #Special case if input latitude is 1 element (slice or medidional average) We only interpolate on the longitude axis
        elif len(np.atleast_1d(lat_in))==1:
            var_OUT=axis_interp(var_OUT, lon_in,lon_t,axis=-1, reverse_input=False, type_int='lin')
        else:#Bi-directional interpolation
            var_OUT=interp_KDTree(var_OUT,lat_in,lon_in,lat_t,lon_t) #lon/lat

    #STEP 2: Linear or log interpolation if there is a vertical axis
    if zaxis_in in VAR_Ncdf.dimensions:
        pos_axis=VAR_Ncdf.dimensions.index(zaxis_in) #Get position: 'pstd' position is 1 in ('time', 'pstd', 'lat', 'lon')
        lev_in=file_Nc_in.variables[zaxis_in][:]
        lev_t=file_Nc_target.variables[zaxis_t][:]
        #Check if the input need to be reverse, note thatwe are reusing find_n() function  which was designed for pressure interpolation
        #so the values are reverse if increasing upward (yes, this is counter intuituve)
        if lev_in[0]<lev_in[-1]:
            reverse_input=True
        else:
            reverse_input=False
        if zaxis_in in ['zagl','zstd'] :
            intType='lin'
        elif zaxis_in=='pstd':
            intType='log'
        var_OUT=axis_interp(var_OUT, lev_in,lev_t, pos_axis, reverse_input=reverse_input, type_int=intType)

    #STEP 3: Linear interpolation in Ls
    if 'time' in VAR_Ncdf.dimensions:
        pos_axis=0
        var_OUT=axis_interp(var_OUT, np.squeeze(areo_in)%360,np.squeeze(areo_t)%360, pos_axis, reverse_input=False, type_int='lin')

    #STEP 4: Linear interpolation in time of day
    #TODO the interpolation scheme is not cyclic.
    #> If Available diurn times  are 04 10 16 22 and requested time is 23, value is left to zero  and not interpololated from 22 and 04 times as it should
    # if requesting
    if ftype_in =='diurn':
        pos_axis=1

        tod_name_in=find_tod_in_diurn(file_Nc_in)
        tod_name_t=find_tod_in_diurn(file_Nc_target)
        tod_in=file_Nc_in.variables[tod_name_in][:]
        tod_t=file_Nc_target.variables[tod_name_t][:]
        var_OUT=axis_interp(var_OUT, tod_in,tod_t, pos_axis, reverse_input=False, type_int='lin')

    return var_OUT


def progress(k,Nmax):
    """
    Display a progress bar to monitor heavy calculations.
    Args:
        k: current iteration of the outer loop
        Nmax: max iteration of the outer loop
    Returns:
        Running... [#---------] 10.64 %
    """
    import sys
    from math import ceil #round yo the 2nd digit
    progress=float(k)/Nmax
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rRunning... [{0}] {1} {2}%".format( "#"*block + "-"*(barLength-block), ceil(progress*100*100)/100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def section_content_amescap_profile(section_ID):
    '''
    Execude code section in /home/user/.amescap_profile
    Args:
        section_ID: string defining the section to load, e.g 'Pressure definitions for pstd'
    Returns
        return line in that section as python code
    '''
    import os
    import numpy as np
    input_file=os.environ['HOME']+'/.amescap_profile'
    try:
        f=open(input_file, "r")
        contents=''
        rec=False
        while True:
            line = f.readline()
            if not line :break #End of File
            if line[0]== '<':
                rec=False
                if line.split('|')[1].strip() ==section_ID:
                    rec=True
                    line = f.readline()
            if rec : contents+=line
        f.close()
        if contents=='':
            prRed("No content found for <<< %s >>> block"%(section_ID))
        return contents

    except FileNotFoundError:
        prRed("Error: %s config file not found "%(input_file))
        prYellow("To use this feature, create a hidden config file from the template in your home directory with:")
        prCyan("    cp AmesCAP/mars_templates/amescap_profile  ~/.amescap_profile")
        exit()
    except Exception as exception: #Return the error
        prRed('Error')
        print(exception)


def filter_vars(fNcdf,include_list=None,giveExclude=False):
    '''
    Filter variable names in netcdf file for processing.
    Will return all dimensions (e.g. 'lon', 'lat'...), the 'areo' variable, and any variable included in include_list
    Args:
        fNcdf: an open netcdf4 object pointing to a diurn, daily or average file
        include_list: a list of variables to include, e.g. ['ucomp','vcomp']
        giveExclude: if True, instead return the variables that must be excluded from the file, i.e.
                     exclude_var = [all the variables] - [axis & dimensions] - [include_list]
    Return:
        var_list
    '''
    var_list=fNcdf.variables.keys()
    #If no list is provided, return all variables:
    if include_list is None: return var_list

    #Make sure the requested variables are present in file
    input_list_filtered=[]
    for ivar in include_list:
        if ivar in var_list:
            input_list_filtered.append(ivar)
        else:
            prYellow('***Warning***  In Script_utils/filter_vars(), variables %s not found in file'%(ivar))
    #Compute baseline variables, i.e. all dimensions, axis etc...
    baseline_var=[]
    for ivar in  var_list:
        if ivar =='areo' or (len(fNcdf.variables[ivar].dimensions))<=2 :
            baseline_var.append(ivar)
    #Return the two lists
    out_list=baseline_var+input_list_filtered
    if giveExclude:
        exclude_list= list(var_list)
        for ivar in out_list:
            exclude_list.remove(ivar)
        out_list=  exclude_list
    return  out_list

def find_fixedfile(filename):
    '''
    Batterson, Updated by Alex Nov 29 2022
    Args:
        filename   =  name of FV3 data file in use, i.e.
                   'atmos_average.tile6.nc'
    Returns:
        name_fixed: fullpath to correspnding fixed file

            DDDDD.atmos_average.nc  -> DDDDD.fixed.nc
            atmos_average.tileX.nc  -> fixed.tileX.nc

            *variations of these work too*

            DDDDD.atmos_average_plevs.nc            -> DDDDD.fixed.nc
            DDDDD.atmos_average_plevs_custom.nc     -> DDDDD.fixed.nc
            atmos_average.tileX_plevs.nc            -> fixed.tileX.nc
            atmos_average.tileX_plevs_custom.nc     -> fixed.tileX.nc
            atmos_average_custom.tileX_plevs.nc     -> fixed.tileX.nc

    '''
    filepath,fname=extract_path_basename(filename)
    #Try the 'tile' or 'standard' version of the fixed files
    if 'tile' in fname:
        name_fixed= filepath + '/fixed.tile'+fname.split('tile')[1][0] + '.nc'
    else:
        name_fixed=filepath + '/'+ fname.split('.')[0] + '.fixed.nc'
    #If neither is found set-up a default name
    if not  os.path.exists(name_fixed): name_fixed='FixedFileNotFound'
    return name_fixed


def get_longname_units(fNcdf,varname):
    '''
    Return the 'long_name' and 'units'  attributes of a netcdf variable.
    If the attributes are not present, this function will return blank strings instead of raising an error
    Args:
        fNcdf: an opened netcdf file
        varname:  A variable to extract the attribute from (e.g. 'ucomp')
    Return:
        longname_txt : long_name attribute, e.g. 'zonal winds'
        units_txt    : units attribute, e.g. [m/s]

    *** NOTE***
    Some functions in MarsVars edit the units, e.g. turn [kg]  to [kg/m], therefore the empty string is made 4
    characters in length ('    ' instead of '') to allow for editing by editing units_txt[:-2] for example
    '''
    return getattr(fNcdf.variables[varname],'long_name','    '), getattr(fNcdf.variables[varname],'units','    ')

def wbr_cmap():
    '''
    Returns a color map that goes from white>blue>green>yellow>red or 'wbr'
    '''
    from matplotlib.colors import ListedColormap
    tmp_cmap = np.zeros((254,4))
    tmp_cmap [:,3]=1. #set alpha

    tmp_cmap[:,0:3]=np.array([[255,255,255],
    [252,254,255], [250,253,255], [247,252,254], [244,251,254], [242,250,254],
    [239,249,254], [236,248,253], [234,247,253], [231,246,253], [229,245,253],
    [226,244,253], [223,243,252], [221,242,252], [218,241,252], [215,240,252],
    [213,239,252], [210,238,251], [207,237,251], [205,236,251], [202,235,251],
    [199,234,250], [197,233,250], [194,232,250], [191,231,250], [189,230,250],
    [186,229,249], [183,228,249], [181,227,249], [178,226,249], [176,225,249],
    [173,224,248], [170,223,248], [168,222,248], [165,221,248], [162,220,247],
    [157,218,247], [155,216,246], [152,214,245], [150,212,243], [148,210,242],
    [146,208,241], [143,206,240], [141,204,238], [139,202,237], [136,200,236],
    [134,197,235], [132,195,234], [129,193,232], [127,191,231], [125,189,230],
    [123,187,229], [120,185,228], [118,183,226], [116,181,225], [113,179,224],
    [111,177,223], [109,175,221], [106,173,220], [104,171,219], [102,169,218],
    [100,167,217], [ 97,165,215], [ 95,163,214], [ 93,160,213], [ 90,158,212],
    [ 88,156,211], [ 86,154,209], [ 83,152,208], [ 81,150,207], [ 79,148,206],
    [ 77,146,204], [ 72,142,202], [ 72,143,198], [ 72,144,195], [ 72,145,191],
    [ 72,146,188], [ 72,147,184], [ 72,148,181], [ 72,149,177], [ 72,150,173],
    [ 72,151,170], [ 72,153,166], [ 72,154,163], [ 72,155,159], [ 72,156,156],
    [ 72,157,152], [ 72,158,148], [ 72,159,145], [ 72,160,141], [ 72,161,138],
    [ 73,162,134], [ 73,163,131], [ 73,164,127], [ 73,165,124], [ 73,166,120],
    [ 73,167,116], [ 73,168,113], [ 73,169,109], [ 73,170,106], [ 73,172,102],
    [ 73,173, 99], [ 73,174, 95], [ 73,175, 91], [ 73,176, 88], [ 73,177, 84],
    [ 73,178, 81], [ 73,179, 77], [ 73,181, 70], [ 78,182, 71], [ 83,184, 71],
    [ 87,185, 72], [ 92,187, 72], [ 97,188, 73], [102,189, 74], [106,191, 74],
    [111,192, 75], [116,193, 75], [121,195, 76], [126,196, 77], [130,198, 77],
    [135,199, 78], [140,200, 78], [145,202, 79], [150,203, 80], [154,204, 80],
    [159,206, 81], [164,207, 81], [169,209, 82], [173,210, 82], [178,211, 83],
    [183,213, 84], [188,214, 84], [193,215, 85], [197,217, 85], [202,218, 86],
    [207,220, 87], [212,221, 87], [217,222, 88], [221,224, 88], [226,225, 89],
    [231,226, 90], [236,228, 90], [240,229, 91], [245,231, 91], [250,232, 92],
    [250,229, 91], [250,225, 89], [250,222, 88], [249,218, 86], [249,215, 85],
    [249,212, 84], [249,208, 82], [249,205, 81], [249,201, 80], [249,198, 78],
    [249,195, 77], [248,191, 75], [248,188, 74], [248,184, 73], [248,181, 71],
    [248,178, 70], [248,174, 69], [248,171, 67], [247,167, 66], [247,164, 64],
    [247,160, 63], [247,157, 62], [247,154, 60], [247,150, 59], [247,147, 58],
    [246,143, 56], [246,140, 55], [246,137, 53], [246,133, 52], [246,130, 51],
    [246,126, 49], [246,123, 48], [246,120, 47], [245,116, 45], [245,113, 44],
    [245,106, 41], [244,104, 41], [243,102, 41], [242,100, 41], [241, 98, 41],
    [240, 96, 41], [239, 94, 41], [239, 92, 41], [238, 90, 41], [237, 88, 41],
    [236, 86, 41], [235, 84, 41], [234, 82, 41], [233, 80, 41], [232, 78, 41],
    [231, 76, 41], [230, 74, 41], [229, 72, 41], [228, 70, 41], [228, 67, 40],
    [227, 65, 40], [226, 63, 40], [225, 61, 40], [224, 59, 40], [223, 57, 40],
    [222, 55, 40], [221, 53, 40], [220, 51, 40], [219, 49, 40], [218, 47, 40],
    [217, 45, 40], [217, 43, 40], [216, 41, 40], [215, 39, 40], [214, 37, 40],
    [213, 35, 40], [211, 31, 40], [209, 31, 40], [207, 30, 39], [206, 30, 39],
    [204, 30, 38], [202, 30, 38], [200, 29, 38], [199, 29, 37], [197, 29, 37],
    [195, 29, 36], [193, 28, 36], [192, 28, 36], [190, 28, 35], [188, 27, 35],
    [186, 27, 34], [185, 27, 34], [183, 27, 34], [181, 26, 33], [179, 26, 33],
    [178, 26, 32], [176, 26, 32], [174, 25, 31], [172, 25, 31], [171, 25, 31],
    [169, 25, 30], [167, 24, 30], [165, 24, 29], [164, 24, 29], [162, 23, 29],
    [160, 23, 28], [158, 23, 28], [157, 23, 27], [155, 22, 27], [153, 22, 27],
    [151, 22, 26], [150, 22, 26], [146, 21, 25]])/255.

    return ListedColormap(tmp_cmap)

def rjw_cmap():
    '''
    Returns a color map that goes from red<jade<wisteria or 'rjw'
    '''
    from matplotlib.colors import ListedColormap
    tmp_cmap = np.zeros((55,4))
    tmp_cmap [:,3]=1. #set alpha

    tmp_cmap[:,0:3]=np.array([[255,   0, 244],
       [248,  40, 244],[241,  79, 244],[234, 119, 244],[228, 158, 244],
       [221, 197, 245],[214, 190, 245],[208, 182, 245],[201, 175, 245],
       [194, 167, 245],[188, 160, 245],[181, 152, 246],[175, 145, 246],
       [140, 140, 247],[105, 134, 249],[ 70, 129, 251],[ 35, 124, 253],
       [  0, 119, 255],[  0, 146, 250],[  0, 173, 245],[  0, 200, 241],
       [  0, 227, 236],[  0, 255, 231],[  0, 255, 185],[  0, 255, 139],
       [  0, 255,  92],[  0, 255,  46],[  0, 255,   0],[ 63, 247,  43],
       [127, 240,  87],[191, 232, 130],[255, 225, 174],[255, 231, 139],
       [255, 237, 104],[255, 243,  69],[255, 249,  34],[255, 255,   0],
       [255, 241,  11],[255, 227,  23],[255, 213,  35],[255, 199,  47],
       [255, 186,  59],[255, 172,  71],[255, 160,  69],[255, 148,  67],
       [255, 136,  64],[255, 124,  62],[255, 112,  60],[255, 100,  58],
       [255,  80,  46],[255,  60,  34],[255,  40,  23],[255,  20,  11],
       [255,   0,   0],[237,  17,   0]])/255.
    return ListedColormap(tmp_cmap)

def dkass_dust_cmap():
    '''
    Returns a color map that goes from yellow>orange>red>purple
    Provided by Courtney B.
    '''
    from matplotlib.colors import ListedColormap,hex2color
    tmp_cmap = np.zeros((256,4))
    tmp_cmap [:,3]=1. #set alpha

    dkass_cmap = ['#ffffa3','#fffea1','#fffc9f','#fffa9d','#fff99b',\
    '#fff799','#fff597','#fef395','#fef293','#fef091','#feee8f',\
    '#feec8d','#fdea8b','#fde989','#fde787','#fde584','#fce382',\
    '#fce180','#fce07e','#fcde7c','#fcdc7a','#fbda78','#fbd976',\
    '#fbd774','#fbd572','#fbd370','#fad16e','#fad06c','#face6a',\
    '#facc68','#faca66','#f9c964','#f9c762','#f9c560','#f9c35e',\
    '#f9c15c','#f8c05a','#f8be58','#f8bc56','#f8ba54','#f8b952',\
    '#f7b750','#f7b54e','#f7b34b','#f7b149','#f7b047','#f6ae45',\
    '#f6ac43','#f6aa41','#f6a83f','#f5a73d','#f5a53b','#f5a339',\
    '#f5a137','#f5a035','#f49e33','#f49c31','#f49a2f','#f4982d',\
    '#f4972b','#f39529','#f39327','#f39125','#f39023','#f38e21',\
    '#f28b22','#f28923','#f18724','#f18524','#f18225','#f08026',\
    '#f07e27','#f07c27','#ef7a28','#ef7729','#ee752a','#ee732a',\
    '#ee712b','#ed6e2c','#ed6c2d','#ed6a2d','#ec682e','#ec652f',\
    '#eb6330','#eb6130','#eb5f31','#ea5d32','#ea5a33','#ea5833',\
    '#e95634','#e95435','#e85136','#e84f36','#e84d37','#e74b38',\
    '#e74839','#e74639','#e6443a','#e6423b','#e5403c','#e53d3c',\
    '#e53b3d','#e4393e','#e4373f','#e4343f','#e33240','#e33041',\
    '#e22e42','#e22b42','#e22943','#e12744','#e12545','#e12345',\
    '#e02046','#e01e47','#df1c48','#df1a48','#df1749','#de154a',\
    '#de134b','#de114c','#dd0e4c','#dd0c4d','#dc0a4e','#dc084f',\
    '#dc064f','#db0350','#db0151','#da0052','#d90153','#d70154',\
    '#d60256','#d40257','#d30258','#d2035a','#d0035b','#cf045c',\
    '#cd045e','#cc055f','#cb0560','#c90562','#c80663','#c60664',\
    '#c50766','#c30767','#c20868','#c1086a','#bf096b','#be096c',\
    '#bc096e','#bb0a6f','#ba0a70','#b80b72','#b70b73','#b50c74',\
    '#b40c75','#b30c77','#b10d78','#b00d79','#ae0e7b','#ad0e7c',\
    '#ac0f7d','#aa0f7f','#a90f80','#a71081','#a61083','#a51184',\
    '#a31185','#a21287','#a01288','#9f1389','#9e138b','#9c138c',\
    '#9b148d','#99148f','#981590','#961591','#951693','#941694',\
    '#921695','#911797','#8f1798','#8e1899','#8d189a','#8b199c',\
    '#8a199d','#881a9e','#871aa0','#861aa1','#841ba2','#831ba4',\
    '#811ca5','#801ca4','#7f1ba2','#7f1ba0','#7e1b9e','#7d1a9b',\
    '#7c1a99','#7b1a97','#7a1995','#791993','#781991','#77198f',\
    '#76188d','#75188b','#751889','#741786','#731784','#721782',\
    '#711680','#70167e','#6f167c','#6e167a','#6d1578','#6c1576',\
    '#6b1574','#6b1471','#6a146f','#69146d','#68136b','#671369',\
    '#661367','#651265','#641263','#631261','#62125f','#61115c',\
    '#61115a','#601158','#5f1056','#5e1054','#5d1052','#5c0f50',\
    '#5b0f4e','#5a0f4c','#590f4a','#580e48','#570e45','#570e43',\
    '#560d41','#550d3f','#540d3d','#530c3b','#520c39','#510c37',\
    '#500c35','#4f0b33','#4e0b30','#4d0b2e','#4d0a2c','#4c0a2a',\
    '#4b0a28','#4a0926','#490924','#480922','#470820']

    RGB_T   = np.array([hex2color(x) for x in dkass_cmap])
    tmp_cmap[:,0:3]=RGB_T

    return ListedColormap(tmp_cmap)

def dkass_temp_cmap():
    '''
    Returns a color map that goes from black>purple>blue>green>yellow>orange>red
    Provided by Courtney B.
    '''
    from matplotlib.colors import ListedColormap,hex2color
    tmp_cmap = np.zeros((256,4))
    tmp_cmap [:,3]=1. #set alpha

    dkass_cmap = ['#200000','#230104','#250208','#27040c','#290510',\
    '#2c0614','#2e0718','#30081c','#320a20','#350b24','#370c28',\
    '#390d2c','#3c0f30','#3e1034','#401138','#42123c','#451340',\
    '#471544','#491648','#4b174c','#4e1850','#501954','#521b58',\
    '#541c5c','#571d60','#591e64','#5b2068','#5d216c','#602270',\
    '#622374','#642478','#66267c','#692780','#6b2884','#6d2988',\
    '#6f2a8c','#722c90','#742d94','#762e98','#782f9c','#7b30a0',\
    '#7d32a4','#7f33a9','#8134ad','#8435b1','#8637b5','#8838b9',\
    '#8a39bd','#8d3ac1','#8f3bc5','#913dc9','#933ecd','#963fd1',\
    '#9840d5','#9a41d9','#9c43dd','#9f44e1','#a145e5','#a346e9',\
    '#a548ed','#a849f1','#aa4af5','#ac4bf9','#ae4cfd','#af4eff',\
    '#ad50ff','#aa53ff','#a755ff','#a458ff','#a25aff','#9f5cff',\
    '#9c5fff','#9961ff','#9764ff','#9466ff','#9168ff','#8e6bff',\
    '#8c6dff','#8970ff','#8672ff','#8374ff','#8177ff','#7e79ff',\
    '#7b7cff','#787eff','#7581ff','#7383ff','#7085ff','#6d88ff',\
    '#6a8aff','#688dff','#658fff','#6291ff','#5f94ff','#5d96ff',\
    '#5a99ff','#579bff','#549dff','#52a0ff','#4fa2ff','#4ca5ff',\
    '#49a7ff','#46aaff','#44acff','#41aeff','#3eb1ff','#3bb3ff',\
    '#39b6ff','#36b8ff','#33baff','#30bdff','#2ebfff','#2bc2ff',\
    '#28c4ff','#25c6ff','#23c9ff','#20cbff','#1dceff','#1ad0ff',\
    '#17d3ff','#15d5ff','#12d7ff','#0fdaff','#0cdcff','#0adfff',\
    '#07e1ff','#04e3ff','#01e6ff','#02e7fe','#06e8fa','#0ae8f6',\
    '#0ee8f2','#12e9ee','#16e9ea','#1ae9e6','#1eeae2','#22eade',\
    '#26ebda','#2aebd6','#2eebd2','#32ecce','#36ecca','#3aecc6',\
    '#3eedc2','#42edbe','#46eeba','#4aeeb6','#4eeeb2','#52efae',\
    '#55efaa','#59f0a6','#5df0a1','#61f09d','#65f199','#69f195',\
    '#6df191','#71f28d','#75f289','#79f385','#7df381','#81f37d',\
    '#85f479','#89f475','#8df471','#91f56d','#95f569','#99f665',\
    '#9df661','#a1f65d','#a5f759','#a9f755','#adf751','#b1f84d',\
    '#b5f849','#b9f945','#bdf941','#c1f93d','#c5fa39','#c9fa35',\
    '#cdfa31','#d1fb2d','#d5fb29','#d9fc25','#ddfc21','#e1fc1d',\
    '#e5fd19','#e9fd15','#edfd11','#f1fe0d','#f5fe09','#f8ff05',\
    '#fcff01','#fdfc00','#fdf800','#fef400','#fef000','#feec00',\
    '#fee800','#fee400','#fee000','#fedc00','#fed800','#fed400',\
    '#fed000','#fecc00','#fec800','#fec400','#fec000','#febc00',\
    '#feb800','#feb400','#feb000','#feac00','#fea800','#fea400',\
    '#fea000','#fe9c00','#fe9800','#fe9400','#fe9000','#fe8c00',\
    '#fe8800','#fe8400','#fe8000','#fe7c00','#fe7800','#fe7400',\
    '#fe7000','#fe6c00','#fe6800','#fe6400','#fe6000','#fe5c00',\
    '#fe5800','#fe5400','#ff5000','#ff4c00','#ff4800','#ff4400',\
    '#ff4000','#ff3c00','#ff3800','#ff3400','#ff3000','#ff2c00',\
    '#ff2800','#ff2400','#ff2000','#ff1c00','#ff1800','#ff1400',\
    '#ff1000','#ff0c00','#ff0800','#ff0400','#ff0000']

    RGB_T   = np.array([hex2color(x) for x in dkass_cmap])
    tmp_cmap[:,0:3]=RGB_T

    return ListedColormap(tmp_cmap)

#=========================================================================
#=========================================================================
#=========================================================================

def pretty_print_to_fv_eta(var,varname,nperline=6):
    """
    Print the ak or bk coefficients to copy paste in fv_eta.f90
    Args:
        data: ak or bk data
        varname: the variable name, 'a' or 'b'
        nperline:the number of elements per line
    Returns:
         The print statement ready to copy-paste in fv_eta.f90

    """
    NLAY=len(var)-1
    import sys
    ni=0
    print('') #skip a line

    #===========================================================
    #=====print the piece of code to copy/paste in fv_eta.f90===
    #===========================================================

    #If a, print the variable definitions before the variable content
    if varname=='a':
        print('      real a%i(%i),b%i(%i)'%(NLAY,NLAY+1,NLAY,NLAY+1)  )
        print('')



    #===Initialize the first line===
    print("data %s%i /      &"%(varname,NLAY))
    sys.stdout.write('    ') #first tab
    #===Loop over all elements=====
    for i in range(0,len(var)-1):
        sys.stdout.write('%16.10e, '%var[i])
        ni+=1
        if ni==nperline:
            sys.stdout.write(' &\n    ')
            ni=0
    #===last line===
    sys.stdout.write('%16.10e /\n'%var[NLAY])

    #If b, print the code snippet after the displaying the variable
    if varname=='b':
        ks=0
        while var[ks]==0. :
            ks+=1
        print('')

        #We remove 1 as it takes two boundary points to form one layer


        print('        case (%i)'%(NLAY))
        print('          ks = %i'%(ks-1))
        print('          do k=1,km+1')
        print('            ak(k) = a%i(k)'%(NLAY))
        print('            bk(k) = b%i(k)'%(NLAY))
        print('          enddo  ')
        print(' ')


def replace_dims(Ncvar_dim,vert_dim_name=None):
    '''
    Update the name for the variables dimension to match FV3's
    Args:
        Ncvar_dim: Netcdf variable dimensions, e.g f_Ncdf.variables['temp'].dimensions
        vert_dim_name(optional): 'pstd', 'zstd', or 'zagl' if the vertical dimensions is ambigeous
    Return:
        dims_out: updated dimensions matching FV3's naming convention

    '''
    #Set input dictionary options that would be recognized as FV3 variables
    lat_dic=['lat','lats','latitudes','latitude']
    lon_dic=['lon','lon','longitude','longitudes']
    lev_dic=['pressure','altitude']
    areo_dic=['ls']

    #Desired outputs

    dims_out=list(Ncvar_dim).copy()
    for ii,idim in enumerate(Ncvar_dim):
        if idim in lat_dic: dims_out[ii]='lat' #Rename axis
        if idim in lon_dic: dims_out[ii]='lon'
        if idim in lev_dic:
            #Vertical coordinate: If no input is provided, assume it is standard pressure 'pstd'
            if vert_dim_name is None:
                dims_out[ii]='pstd'
            else: #use provided dimension
                dims_out[ii]=vert_dim_name

    return tuple(dims_out)

def ak_bk_loader(fNcdf):
    '''
    Return the ak and bk. First look in the current netcdf file.
    If not found, this routine will check the XXXXX.fixed.nc in the same directory and then in the XXXXX.fixed.tileX.nc files if present
    Args:
        fNcdf: an opened netcdf file
    Returns:
        ak,bk : the ak, bk values
    ***NOTE***

    This routine will look for both 'ak' and 'pk'.

    There are cases when it is convenient to load the  pk, bk once at the begining of the files in MarsVars.py,
    However the pk, bk may not be used at all in the calculation. This is the case with MarsVars.py XXXXX.atmos_average_psd.nc --add msf (which operates on the _pstd.nc file)


    '''
    #First try to read pk and bk in the current netcdf file:
    allvars=fNcdf.variables.keys()

    #Get Netcdf file and path (for debugging)
    Ncdf_name=get_Ncdf_path(fNcdf) #netcdf file
    filepath,fname=extract_path_basename(Ncdf_name)
    fullpath_name=os.path.join(filepath,fname)
    #Check for ak first, then pk

    if ('pk' in allvars or 'ak' in allvars) and 'bk' in allvars:
        if 'ak' in allvars:
            ak=np.array(fNcdf.variables['ak'])
        else:
            ak=np.array(fNcdf.variables['pk'])
        bk=np.array(fNcdf.variables['bk'])
        print('ak bk in file')

    else:
        try:
            name_fixed=find_fixedfile(fullpath_name)
            f_fixed=Dataset(name_fixed, 'r', format='NETCDF4_CLASSIC')
            allvars=f_fixed.variables.keys()
            #Check for ak firdt, then pk
            if 'ak' in allvars:
                ak=np.array(f_fixed.variables['ak'])
            else:
                ak=np.array(f_fixed.variables['pk'])
            bk=np.array(f_fixed.variables['bk'])
            f_fixed.close()
            print('pk bk in fixed file')
        except:
            prRed('Fixed file does not exist in '\
                            + filepath + ' make sure the fixed '\
                            'file you are referencing matches the '\
                            'FV3 filetype (i.e. fixed.tileX.nc '\
                            'for operations on tile X)')
            exit()

    return ak,bk

def read_variable_dict_amescap_profile(f_Ncdf=None):
    '''
    Inspect a Netcdf file and return the name of the variables and dimensions based on the content of ~/.amescap_profile.
    Calling this function allows to remove hard-coded calls in CAP.
    For example, to f.variables['ucomp'] is replaced by f.variables[model.ucomp], with model.ucomp taking the values of'ucomp', 'U'
    Args:
        f_Ncdf: An opened Netcdf file object
    Returns:
        model: a dictionary with the dimensions and variables, e.g. model.ucomp='U' or model.dim_lat='latitudes'

    ***NOTE***
    The defaut names for variables are defined in () parenthesis in  ~/.amescap_profile :
    'X direction wind        [m/s]                   (ucomp)>'

    The defaut names for dimensions are defined in {} parenthesis in  ~/.amescap_profile :
    Ncdf Y latitude dimension    [integer]          {lat}>lats

    The dimensions (lon,lat,pfull,pstd) are loaded in the dictionary as model.dim_lon, model.dim_lat
    '''

    if f_Ncdf is not None:
        var_list_Ncdf=list(f_Ncdf.variables.keys())
        dim_list_Ncdf=list(f_Ncdf.dimensions.keys())
    else:
        var_list_Ncdf=[]
        dim_list_Ncdf=[]

    all_lines=section_content_amescap_profile('Variable dictionary')
    lines=all_lines.split('\n')
    #Remove empty lines:
    while("" in lines):lines.remove("")

    #Initialize model
    class model(object):
        pass
    MOD=model()

    #Read through all lines in the Variable dictionary section of amesgcm_profile:
    for il in lines:
        var_list=[]
        #e.g. 'X direction wind [m/s]                          (ucomp)>U,u'
        left,right=il.split('>') #Split on either side of '>'

        #If using {var}, current entry is a dimension. If using (var), it is a variable
        if '{' in left:
            sep1='{';sep2='}';type_input='dimension'
        elif '(' in left:
            sep1='(';sep2=')';type_input='variable'

        # First get 'ucomp' from  'X direction wind [m/s]      (ucomp)
        _,tmp=FV3_var=left.split(sep1)
        FV3_var=tmp.replace(sep2,'').strip() #THIS IS THE FV3 NAME OF THE CURRENT VARIABLE
        #Then, get the list of variable on the righ-hand side, e.g.  'U,u'
        all_vars=right.split(',')
        for ii in all_vars:
            var_list.append(ii.strip())  #var_list IS A LIST OF POTENTIAL CORRESPONDING VARIABLES
        #Set the attribute to the dictionary
        #If the list is empty, e.g just [''], use the default FV3 variable presents in () or {}
        if len(var_list)==1 and var_list[0]=='':var_list[0]=FV3_var #var_list IS A LIST OF POTENTIAL CORRESPONDING VARIABLES
        found_list=[]

        #Place the input in the appropriate varialbe () or dimension {} dictionary
        #print('var_list>>>',var_list)
        if type_input=='variable':
            for ivar in var_list:
                if ivar in var_list_Ncdf:found_list.append(ivar)

            if len(found_list)==0:
                setattr(MOD,FV3_var,FV3_var)
            elif  len(found_list)==1:
                setattr(MOD,FV3_var,found_list[0])
            else:
                setattr(MOD,FV3_var,found_list[0])
                prYellow('''***Warning*** more than one possible variable '%s' found in file: %s'''%(FV3_var,found_list))
        if type_input=='dimension':
            for ivar in var_list:
                if ivar in dim_list_Ncdf:found_list.append(ivar)
            if len(found_list)==0:
                setattr(MOD,'dim_'+FV3_var,FV3_var)
            elif  len(found_list)==1:
                setattr(MOD,'dim_'+FV3_var,found_list[0])
            else:
                setattr(MOD,'dim_'+FV3_var,found_list[0])
                prYellow('''***Warning*** more than one possible dimension '%s' found in file: %s'''%(FV3_var,found_list))

    return MOD
