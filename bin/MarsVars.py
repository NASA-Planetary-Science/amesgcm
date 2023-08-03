#!/usr/bin/env python3

# Load generic Python Modules
import argparse   # parse arguments
import os         # access operating systems function
import subprocess # run command
import sys        # system command
import warnings   # suppress certain errors when dealing with NaN arrays

from amescap.FV3_utils import fms_press_calc, fms_Z_calc, dvar_dh, cart_to_azimut_TR
from amescap.FV3_utils import mass_stream, zonal_detrend, spherical_div, spherical_curl, frontogenesis
from amescap.Script_utils import check_file_tape, prYellow, prRed, prCyan, prGreen, prPurple, print_fileContent
from amescap.Script_utils import FV3_file_type, filter_vars, find_fixedfile, get_longname_units, ak_bk_loader
from amescap.Ncdf_wrapper import Ncdf

# Attempt to import specific scientic modules that may or may not
# be included in the default Python installation on NAS.
try:
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib not to use any Xwindows backend.
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

parser = argparse.ArgumentParser(description="""\033[93m MarsVars, variable manager. Add to or remove variables from the diagnostic files. \n'
                                             Use MarsFiles ****.atmos.average.nc to view file content. \033[00m""",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('input_file', nargs='+',  # sys.stdin
                    help='***.nc file or list of ***.nc files ')

parser.add_argument('-add', '--add', nargs='+', default=[],
                    help='Add a new variable to file. Variables that can be added are listed below. \n'
                    '> Usage: MarsVars ****.atmos.average.nc -add varname \n'
                    '\033[96mON NATIVE FILES: \n'
                    'rho              (Density)                         Req. [ps, temp] \n'
                    'theta            (Potential Temperature)           Req. [ps, temp] \n'
                    'pfull3D          (Pressure at layer midpoint)      Req. [ps, temp] \n'
                    'DP               (Layer thickness [pressure])      Req. [ps, temp] \n'
                    'DZ               (layer thickness [altitude])      Req. [ps, temp] \n'
                    'zfull            (Altitude AGL)                    Req. [ps, temp] \n'
                    'w                (Vertical Wind)                   Req. [ps, temp, omega] \n'
                    'wdir             (Horiz. Wind Direction)           Req. [ucomp, vcomp] \n'
                    'wspeed           (Horiz. Wind Magnitude)           Req. [ucomp, vcomp] \n'
                    'N                (Brunt Vaisala Frequency)         Req. [ps, temp] \n'
                    'Ri               (Richardson Number)               Req. [ps, temp] \n'
                    'Tco2             (CO2 Condensation Temperature)    Req. [ps, temp] \n'
                    'scorer_wl        (Scorer Horiz. Wavelength)        Req. [ps, temp, ucomp] \n'
                    'div              (Divergence of Wind)              Req. [ucomp, vcomp] \n'
                    'curl             (Relative Vorticity)              Req. [ucomp, vcomp] \n'
                    'fn               (Frontogenesis)                   Req. [ucomp, vcomp, theta] \n'
                    'dzTau            (Dust Extinction Rate)            Req. [dst_mass_micro, temp] \n'
                    'izTau            (Ice Extinction Rate)             Req. [ice_mass_micro, temp] \n'
                    'dst_mass_micro   (Dust Mass Mixing Ratio)          Req. [dzTau, temp] \n'
                    'ice_mass_micro   (Ice Mass Mixing Ratio)           Req. [izTau, temp] \n'
                    'Vg_sed           (Sedimentation Rate)              Req. [dst_mass_micro, dst_num_micro, temp] \n'
                    'w_net            (Net Vertical Wind (w-Vg_sed))    Req. [w, Vg_sed] \n'
                    ' \n\nNOTE:                    \n'
                    '   MarsVars offers some support on interpolated files. Particularly if pfull3D \n'
                    '              and zfull are added to the file before interpolation. \n'
                    '\033[00m \n'
                    '\033[93mON INTERPOLATED FILES (i.e. _pstd, _zstd, _zagl): \n'
                    'msf              (Mass Stream Function)              Req. [vcomp] \n'
                    'ep               (Wave Potential Energy)             Req. [temp] \n'
                    'ek               (Wave Kinetic Energy)               Req. [ucomp, vcomp] \n'
                    'mx               (Vertical Flux of Zonal Momentum)   Req. [ucomp, w] \n'
                    'my               (Vertical Flux of Merid. Momentum)  Req. [vcomp, w] \n'
                    'ax               (Zonal Wave-Mean Flow Forcing)      Req. [ucomp, w, rho] \n'
                    'ay               (Merid. Wave-Mean Flow Forcing)     Req. [ucomp, w, rho] \n'
                    'tp_t             (Normalized Temp. Perturbation)     Req. [temp] \n'
                    '\033[00m')


parser.add_argument('-zdiff', '--zdiff', nargs='+', default=[],
                    help="""Differentiate a variable w.r.t. the Z axis \n"""
                    """A new a variable d_dz_var in [Unit/m] will be added to the file. \n"""
                    """> Usage: MarsVars ****.atmos.average.nc -zdiff temp \n"""
                    """ \n""")
parser.add_argument('-col', '--col', nargs='+', default=[],
                    help="""Integrate a mixing ratio of a variable through the column. \n"""
                    """A new a variable var_col in [kg/m2] will be added to the file. \n"""
                    """> Usage: MarsVars ****.atmos.average.nc -col ice_mass \n"""
                    """ \n""")

parser.add_argument('-zd', '--zonal_detrend', nargs='+', default=[],
                    help="""Detrend a variable by substracting its zonal mean value. \n"""
                    """A new a variable var_p (for prime) will be added to the file. \n"""
                    """> Usage: MarsVars ****.atmos.average.nc -zd ucomp \n"""
                    """ \n""")

parser.add_argument('-dp_to_dz', '--dp_to_dz', nargs='+', default=[],
                    help="""Convert aerosol opacities [op/Pa] to [op/m] (-dp_to_dz) and [op/m] to [op/Pa] (-dp_to_dz) \n"""
                    """Requires [DP, DZ]. \n"""
                    """A new a variable var_dp_to_dz will be added to the file \n"""
                    """> Usage: MarsVars ****.atmos.average.nc -dp_to_dz opacity \n"""
                    """  Use -dz_to_dp to convert from [op/m] to [op/Pa]\n""")

parser.add_argument('-dz_to_dp', '--dz_to_dp', nargs='+', default=[],
                    help=argparse.SUPPRESS)  # same as --hpf but without the instructions

parser.add_argument('-rm', '--remove', nargs='+', default=[],
                    help='Remove a variable from a file. \n'
                    '> Usage: MarsVars ****.atmos.average.nc -rm rho theta \n')

parser.add_argument('-extract', '--extract', nargs='+', default=[],
                    help='Extract variable(s) to a new _extract.nc file. \n'
                    '> Usage: MarsVars ****.atmos.average.nc -extract ps ts \n')

parser.add_argument('-edit', '--edit', default=None,
                    help="""Edit a variable 'name', 'longname', or 'unit', or scale its values. \n"""
                    """> Use jointly with -rename -longname -unit or -multiply flags \n"""
                    """> Usage: MarsVars.py *.atmos_average.nc --edit temp -rename airtemp \n"""
                    """> Usage: MarsVars.py *.atmos_average.nc --edit ps -multiply 0.01 -longname 'new pressure' -unit 'mbar' \n"""
                    """ \n""")

parser.add_argument('-rename', '--rename', type=str, default=None,
                    help=argparse.SUPPRESS)                             # To be used jointly with --edit

parser.add_argument('-longname', '--longname', type=str, default=None,
                    help=argparse.SUPPRESS)                             # To be used jointly with --edit

parser.add_argument('-unit', '--unit', type=str, default=None,
                    help=argparse.SUPPRESS)                             # To be used jointly with --edit

parser.add_argument('-multiply', '--multiply', type=float,
                    default=None, help=argparse.SUPPRESS)               # To be used jointly with --edit

parser.add_argument('--debug',  action='store_true',
                    help='Debug flag: release the exception')

# =====================================================================
# This is a list of the supported variables for --add (short name, longname, units)
# =====================================================================
VAR = {'rho':               ['density (postprocessed with CAP)', 'kg/m3'],
       'theta':             ['potential temperature (postprocessed with CAP)', 'K'],
       'w':                 ['vertical wind (postprocessed with CAP)', 'm/s'],
       'pfull3D':           ['pressure at layer midpoint (postprocessed with CAP)', 'Pa'],
       'DP':                ['layer thickness (pressure) (postprocessed with CAP)', 'Pa'],
       'zfull':             ['altitude  AGL at layer midpoint (postprocessed with CAP)', 'm'],
       'DZ':                ['layer thickness (altitude) (postprocessed with CAP)', 'm'],
       'wdir':              ['wind direction (postprocessed with CAP)', 'deg'],
       'wspeed':            ['wind speed (postprocessed with CAP)', 'm/s'],
       'N':                 ['Brunt Vaisala frequency (postprocessed with CAP)', 'rad/s'],
       'Ri':                ['Richardson number (postprocessed with CAP)', 'none'],
       'Tco2':              ['CO2 condensation temerature (postprocessed with CAP)', 'K'],
       'div':               ['divergence of the wind field (postprocessed with CAP)', 'Hz'],
       'curl':              ['relative vorticity (postprocessed with CAP)', 'Hz'],
       'scorer_wl':         ['Scorer horizontal wavelength [L=2.pi/sqrt(l**2)] (postprocessed with CAP)', 'm'],
       'msf':               ['mass stream function (postprocessed with CAP)', '1.e8 x kg/s'],
       'ep':                ['wave potential energy (postprocessed with CAP)', 'J/kg'],
       'ek':                ['wave kinetic energy (postprocessed with CAP)', 'J/kg'],
       'mx':                ['vertical flux of zonal momentum (postprocessed with CAP)', 'J/kg'],
       'my':                ['vertical flux of merididional momentum(postprocessed with CAP)', 'J/kg'],
       'ax':                ['zonal wave-mean flow forcing (postprocessed with CAP)', 'm/s/s'],
       'ay':                ['meridional wave-mean flow forcing (postprocessed with CAP)', 'm/s/s'],
       'tp_t':              ['normalized temperature perturbation (postprocessed with CAP)', 'None'],
       'fn':                ['frontogenesis (postprocessed with CAP)', 'K m-1 s-1'],
       'dzTau':             ['dust extinction rate (postprocessed with CAP)', 'km-1'],
       'izTau':             ['ice extinction rate (postprocessed with CAP)', 'km-1'],
       'dst_mass_micro':    ['dust mass mixing ratio (postprocessed with CAP)', 'kg/kg'],
       'ice_mass_micro':    ['ice mass mixing ratio (postprocessed with CAP)', 'kg/kg'],
       'Vg_sed':            ['sedimentation rate (postprocessed with CAP)', 'm/s'],
       'w_net':             ['net vertical wind [w-Vg_sed] (postprocessed with CAP)', 'm/s'],
       }
# =====================================================================
# =====================================================================
# =====================================================================
# TODO : If only one timestep, reshape from (lev, lat, lon) to (time, lev, lat, lon)

# Fill values for NaN. Do not use np.NaN, will raise error when running runpinterp
fill_value = 0.

# Define constants
global rgas, psrf, Tpole, g, R, Rd, rho_air, rho_dst, rho_ice, Qext_dst, Qext_ice, n0, S0, T0,\
    Cp, Na, amu, amu_co2, mass_co2, sigma, M_co2, N, C_dst, C_ice

rgas        = 189.              # Cas cosntant for CO2                  J/(kg-K) (or m2/(s2 K))
psrf        = 610.              # Mars Surface Pressure                 Pa (or kg/ms^2)
Tpole       = 150.              # Polar Temperature                     K
g           = 3.72              # Gravitational Constant for Mars       m/s2
R           = 8.314             # Universal Gas Constant                J/(mol. K)
Rd          = 192.0             # R for dry air on Mars                 J/(kg K)
rho_air     = psrf/(rgas*Tpole) # Air Density                           kg/m3
rho_dst     = 2500.             # Dust Particle Density                 kg/m3
#rho_dst     = 3000              # Dust Particle Density                 kg/m3       Kleinbohl et al. 2009
rho_ice     = 900               # Ice Particle Density                  kg/m3       Heavens et al. 2010
Qext_dst    = 0.35              # Dust Extinction Efficiency (MCS)                  Kleinbohl et al. 2009
Qext_ice    = 0.773             # ice Extinction Efficiency (MCS)                   Heavens et al. 2010
Reff_dst    = 1.06              # Effective Dust Particle Radius        micron      Kleinbohl et al. 2009
Reff_ice    = 1.41              # Effective Ice Particle Radius         micron      Heavens et al. 2010
n0          = 1.37*1.e-5        # Sutherland's Law                      N-s/m2
S0          = 222               # Sutherland's Law                      K
T0          = 273.15            # Sutherland's Law                      K
Cp          = 735.0             # J/K
Na          = 6.022*1.e23       # Avogadro's Number                     per mol
Kb          = R/Na              # Boltzmann Constant                    (m2 kg)/(s2 K)
amu         = 1.66054*1.e-27    # Atomic Mass Unit                      kg/amu
amu_co2     = 44.0              # Molecular Mass of CO2                 amu
mass_co2    = amu_co2*amu       # Mass of 1 CO2 Particle                kg
sigma       = 0.63676           # Gives Effective Variance = 0.5 (Dust)
M_co2       = 0.044             # Molar Mass of CO2                     kg/mol
N           = 0.01              # For the wave potential energy calc    rad/s

C_dst       = (4/3)*(rho_dst/Qext_dst)*Reff_dst     # 12114.286         m-2
C_ice       = (4/3)*(rho_ice/Qext_ice)*Reff_ice     # 2188.874          m-2


# ===========================
def compute_p_3D(ps, ak, bk, shape_out):
    """
    Return the 3D pressure field at the layer midpoint.
    *** NOTE***
    The shape_out argument ensures that when time = 1 (one timestep), results are returned
    as (1, lev, lat, lon) not (lev, lat, lon)
    """
    p_3D = fms_press_calc(ps, ak, bk, lev_type='full')
    # p_3D [lev, tim, lat, lon] ->[tim, lev, lat, lon]
    p_3D = p_3D.transpose(lev_T)
    return p_3D.reshape(shape_out)

# =====================================================================
def compute_rho(p_3D, temp):
    """
    Returns density in [kg/m3].
    """
    return p_3D/(rgas*temp)

# =====================================================================
def compute_xzTau(q, temp, lev, const, f_type):
    """
    Returns dust or ice extinction in [km-1].
    Adapted from Heavens et al. 2011, observations by MCS (JGR).
    """
    if f_type == 'diurn':
        PT = np.repeat(
            lev, (q.shape[0] * q.shape[1] * q.shape[3] * q.shape[4]))
        PT = np.reshape(
            PT, (q.shape[2],  q.shape[0],  q.shape[1],  q.shape[3],   q.shape[4]))
        # (lev, tim, tod, lat, lon) -> (tim, tod, lev, lat, lon)
        P = PT.transpose((1, 2, 0, 3, 4))
    else:
        PT = np.repeat(lev, (q.shape[0] * q.shape[2] * q.shape[3]))
        PT = np.reshape(
            PT, (q.shape[1],  q.shape[0],  q.shape[2],  q.shape[3]))
        # (lev, tim, lat, lon) -> (tim, lev, lat, lon)
        P = PT.transpose(lev_T)

    rho_z = P/(Rd*temp)
    # Converts Mass Mixing Ratio (q) from kg/kg -> ppm (mg/kg)
    # Converts extinction (xzTau) from m-1 -> km-1
    xzTau = (rho_z*(q*1.e6)/const)*1000
    return xzTau

# =====================================================================
def compute_mmr(xTau, temp, lev, const, f_type):
    """
    Return dust or ice mixing ratio [kg/kg]
    Adapted from Heavens et al. 2011. observations by MCS (JGR)
    """
    if f_type == 'diurn':
        PT = np.repeat(
            lev, (xTau.shape[0] * xTau.shape[1] * xTau.shape[3] * xTau.shape[4]))
        PT = np.reshape(
            PT, (xTau.shape[2],  xTau.shape[0],  xTau.shape[1],  xTau.shape[3],   xTau.shape[4]))
        # (lev, tim, tod, lat, lon) -> (tim, tod, lev, lat, lon)
        P = PT.transpose((1, 2, 0, 3, 4))
    else:
        PT = np.repeat(lev, (xTau.shape[0] * xTau.shape[2] * xTau.shape[3]))
        PT = np.reshape(
            PT, (xTau.shape[1],  xTau.shape[0],  xTau.shape[2],  xTau.shape[3]))
        # (lev, tim, lat, lon) -> (tim, lev, lat, lon)
        P = PT.transpose(lev_T)

    rho_z = P/(Rd*temp)
    # Converts extinction (xzTau) from km-1 -> m-1
    # Converts mass mixing ratio (q) from ppm (kg/kg) -> mg/kg
    q = (const*(xTau/1000)/rho_z)/1.e6
    return q

# =====================================================================
def compute_Vg_sed(xTau, nTau, temp):
    """
    Returns the dust sedimentation rate.
    """
    r0 = (((3.*xTau) / (4.*np.pi*rho_dst*nTau))
          ** (1/3) * np.exp(-3*(sigma**2)/2))
    Rp = r0*np.exp(3.5*sigma**2)
    c = (2/9)*rho_dst*(Rp)**2*g
    eta = n0*((temp/T0)**(3/2))*((T0+S0)/(temp+S0))
    v = np.sqrt((3*Kb*temp)/mass_co2)
    mfp = 2*eta/(rho_air*v)
    Kn = mfp/Rp
    alpha = 1.246+0.42*np.exp(-0.87/Kn)
    Vg = c*(1+alpha*Kn)/eta
    return Vg

# =====================================================================
def compute_w_net(Vg, wvar):
    """
    Returns the net vertical wind (subtracts the sedimentation rate (Vg_sed)
    from the vertical wind (w))
    w_net = w - Vg_sed
    """
    w_net = np.subtract(wvar, Vg)
    return w_net

# =====================================================================
def compute_theta(p_3D, ps, temp, f_type):
    """
    Returns the potential temperature in [K].
    """
    theta_exp = R/(M_co2*Cp)
    # Broadcast dimensions
    ps_shape = ps.shape
    if f_type == 'diurn':
        # (time, tod, lat, lon) is transformed into (time, tod, 1, lat, lon)
        ps_shape = [ps_shape[0], ps_shape[1], 1, ps_shape[2], ps_shape[3]]
    else:
        # (time, lat, lon) is transformed into (time, 1, lat, lon)
        ps_shape = [ps_shape[0], 1, ps_shape[1], ps_shape[2]]

    return temp*(np.reshape(ps, ps_shape)/p_3D)**(theta_exp)

# =====================================================================
def compute_w(rho, omega):
    return -omega/(rho*g)

# =====================================================================
def compute_zfull(ps, ak, bk, temp):
    """
    Returns the altitude of the layer midpoints AGL in [m].
    """
    dim_out = temp.shape
    zfull = fms_Z_calc(ps, ak, bk, temp.transpose(
        lev_T), topo=0., lev_type='full')  # (lev, time, tod, lat, lon)
    # p_3D [lev, tim, lat, lon] -> [tim, lev, lat, lon]
    # temp [tim, tod, lev, lat, lon, lev] -> [lev, time, tod,lat, lon]
    zfull = zfull.transpose(lev_T_out)
    return zfull

# =====================================================================
def compute_zhalf(ps, ak, bk, temp):
    """
    Returns the altitude of the layer interfaces AGL in [m]
    """
    dim_out = temp.shape
    # temp: [tim, lev, lat, lon, lev] ->[lev, time,  lat,  lon]
    zhalf = fms_Z_calc(ps, ak, bk, temp.transpose(
        lev_T), topo=0., lev_type='half')
    # p_3D [lev+1, tim, lat, lon] ->[tim, lev+1, lat, lon]
    zhalf = zhalf.transpose(lev_T_out)
    return zhalf

# =====================================================================
def compute_DZ_full_pstd(pstd, temp, ftype='average'):
    """
    Returns the thickness of a layer (distance between two layers) from the
    midpoint of the standard pressure levels ('pstd').

    Args:
        pstd:   1D array of standard pressure in [Pa]
        temp:   3D array of temperature
        ftype: 'daily', 'aveage', or 'diurn'
    Returns:
        DZ_full_pstd: 3D array of thicknesses

    *** NOTE***
    In this context, 'pfull' = 'pstd' with the layer interfaces defined somewhere
    in between successive layers.

    --- Nk --- TOP       ========  phalf
    --- Nk-1 ---
                         --------  pfull = pstd    ^
                                                   | DZ_full_pstd
                         ========  phalf           |
    --- 1 ---            --------  pfull = pstd    v
    --- 0 --- SFC        ========  phalf
                        / / / /
    """
    if ftype == 'diurn':
        axis = 2
    else:
        axis = 1

    temp = np.swapaxes(temp, 0, axis)

    # Create broadcasting array for 'pstd'
    shape_out = temp.shape
    reshape_shape = [1 for i in range(0, len(shape_out))]
    reshape_shape[0] = len(pstd)  # e.g [28, 1, 1, 1]
    pstd_b = pstd.reshape(reshape_shape)

    DZ_full_pstd = np.zeros_like(temp)

    # Use the average temperature for both layers
    DZ_full_pstd[0:-1, ...] = -rgas*0.5 * \
        (temp[1:, ...]+temp[0:-1, ...])/g * \
        np.log(pstd_b[1:, ...]/pstd_b[0:-1, ...])

    # There is nothing to differentiate the last layer with, so copy over the value at N-1.
    # Note that unless you fine-tune the standard pressure levels to match the model top,
    # there is typically data missing in the last few layers. This is not a major issue.

    DZ_full_pstd[-1, ...] = DZ_full_pstd[-2, ...]
    return np.swapaxes(DZ_full_pstd, 0, axis)

# =====================================================================
def compute_N(theta, zfull):
    """
    Returns the Brunt Vaisala freqency in [rad/s].
    """
    dtheta_dz = dvar_dh(theta.transpose(
        lev_T), zfull.transpose(lev_T)).transpose(lev_T)
    return np.sqrt(g/theta*dtheta_dz)

# =====================================================================
def compute_Tco2(P_3D, temp):
    """
    Returns the frost point of CO2 in [K].
    Adapted from Fannale, 1982. Mars: The regolith-atmosphere cap system and climate change. Icarus.
    """
    return np.where(P_3D < 518000, -3167.8/(np.log(0.01*P_3D)-23.23), 684.2-92.3*np.log(P_3D)+4.32*np.log(P_3D)**2)

# =====================================================================
def compute_scorer(N, ucomp, zfull):
    """
    Returns the Scorer wavelength in [m].
    """
    dudz = dvar_dh(ucomp.transpose(lev_T),
                   zfull.transpose(lev_T)).transpose(lev_T)
    dudz2 = dvar_dh(dudz.transpose(lev_T),
                    zfull.transpose(lev_T)).transpose(lev_T)
    scorer2 = N**2/ucomp**2 - 1./ucomp*dudz2
    return 2*np.pi/np.sqrt(scorer2)

# =====================================================================
def compute_DP_3D(ps, ak, bk, shape_out):
    """
    Returns the thickness of a layer in [Pa].
    """
    p_half3D = fms_press_calc(ps, ak, bk, lev_type='half')  # [lev, tim, lat, lon]
    DP_3D = p_half3D[1:, ..., ] - p_half3D[0:-1, ...]
    # p_3D [lev, tim, lat, lon] ->[tim, lev, lat, lon]
    DP_3D = DP_3D.transpose(lev_T)
    out = DP_3D.reshape(shape_out)
    return out

# =====================================================================
def compute_DZ_3D(ps, ak, bk, temp, shape_out):
    """
    Returns the layer thickness in [Pa].
    """
    z_half3D = fms_Z_calc(ps, ak, bk, temp.transpose(
        lev_T), topo=0., lev_type='half')
    # Note the reversed order: Z decreases with increasing levels
    DZ_3D = z_half3D[0:-1, ...]-z_half3D[1:, ..., ]
    # DZ_3D [lev, tim, lat, lon] ->[tim, lev, lat, lon]
    DZ_3D = DZ_3D.transpose(lev_T)
    out = DZ_3D.reshape(shape_out)
    return out

# =====================================================================
def compute_Ep(temp):
    """
    Returns the wave potential energy (Ep) in [J/kg].
    Ep = 1/2 (g/N)**2 (T'/T)**2
    """
    return 0.5*g**2*(zonal_detrend(temp)/(temp*N))**2

# =====================================================================
def compute_Ek(ucomp, vcomp):
    """
    Returns the wave kinetic energy (Ek) in [J/kg].
    Ek= 1/2 (u'**2+v'**2)
    """
    return 0.5*(zonal_detrend(ucomp)**2+zonal_detrend(vcomp)**2)

# =====================================================================
def compute_MF(UVcomp, w):
    """
    Returns the zonal or meridional momentum fluxes (u'w' or v'w').
    """
    return zonal_detrend(UVcomp)*zonal_detrend(w)

# =====================================================================
def compute_WMFF(MF, rho, lev, interp_type):
    """
    Returns the zonal or meridional wave-mean flow forcing
    ax= -1/rho d(rho u'w')/dz in [m/s/s]
    ay= -1/rho d(rho v'w')/dz in [m/s/s]

    For 'pstd':
        [du/dz = (du/dp).(dp/dz)] > [du/dz = -rho g (du/dp)] with dp/dz = -rho g
    """
    # Differentiate the variable
    darr_dz = dvar_dh((rho*MF).transpose(lev_T), lev).transpose(lev_T)

    if interp_type == 'pstd':
        # Computed du/dp, need to multiply by (-rho g) to obtain du/dz
        return g * darr_dz
    else:
        # With 'zagl' and 'zstd', levels already in meters du/dz
        # computation does not need the above multiplier.
        return -1/rho*darr_dz

# =====================================================================
# =====================================================================
# =====================================================================

filepath = os.getcwd()

def main():
    # Load all the .nc files
    file_list       = parser.parse_args().input_file
    add_list        = parser.parse_args().add
    zdiff_list      = parser.parse_args().zdiff
    zdetrend_list   = parser.parse_args().zonal_detrend
    dp_to_dz_list   = parser.parse_args().dp_to_dz
    dz_to_dp_list   = parser.parse_args().dz_to_dp
    col_list        = parser.parse_args().col
    remove_list     = parser.parse_args().remove
    extract_list    = parser.parse_args().extract
    edit_var        = parser.parse_args().edit
    debug           = parser.parse_args().debug

    # An array to swap vertical axis forward and backward:
    # [1, 0, 2, 3]    for [time, lev, lat, lon]      and
    # [2, 1, 0, 3, 4] for [time, tod, lev, lat, lon]
    global lev_T
    global lev_T_out  # Reshape in 'zfull' and 'zhalf' calculation

    # Check if an operation is requested. Otherwise, print file content.
    if not (add_list or zdiff_list or zdetrend_list or remove_list or col_list or extract_list or dp_to_dz_list or dz_to_dp_list or edit_var):
        print_fileContent(file_list[0])
        prYellow(''' ***Notice***  No operation requested. Use '-add', '-zdiff', '-zd', '-col', '-dp_to_dz', '-rm' '-edit' ''')
        exit()  # Exit cleanly

    # For all the files
    for ifile in file_list:
        # First check if file is on the disk (Lou only)
        check_file_tape(ifile)

        # =================================================================
        # ========================= Remove ================================
        # =================================================================
        if remove_list:
            cmd_txt = 'ncks --version'
            try:
                # If ncks is available, use it
                subprocess.check_call(cmd_txt, shell=True, stdout=open(
                    os.devnull, "w"), stderr=open(os.devnull, "w"))
                print('ncks is available. Using it.')
                for ivar in remove_list:
                    print('Creating new file %s without %s:' % (ifile, ivar))
                    cmd_txt = 'ncks -C -O -x -v %s %s %s' % (
                        ivar, ifile, ifile)
                    try:
                        subprocess.check_call(cmd_txt, shell=True, stdout=open(
                            os.devnull, "w"), stderr=open(os.devnull, "w"))
                    except Exception as exception:
                        print(exception.__class__.__name__ +
                              ": " + exception.message)
            except subprocess.CalledProcessError:
                # ncks is not available, use internal method
                print('Using internal method instead.')
                f_IN = Dataset(ifile, 'r', format='NETCDF4_CLASSIC')
                ifile_tmp = ifile[:-3]+'_tmp'+'.nc'
                Log = Ncdf(ifile_tmp, 'Edited postprocess')
                Log.copy_all_dims_from_Ncfile(f_IN)
                Log.copy_all_vars_from_Ncfile(f_IN, remove_list)
                f_IN.close()
                Log.close()
                cmd_txt = 'mv '+ifile_tmp+' '+ifile
                p = subprocess.run(
                    cmd_txt, universal_newlines=True, shell=True)
                prCyan(ifile+' was updated')

        # =================================================================
        # ======================== Extract ================================
        # =================================================================
        if extract_list:
            f_IN = Dataset(ifile, 'r', format='NETCDF4_CLASSIC')
            exclude_list = filter_vars(f_IN, parser.parse_args(
            ).extract, giveExclude=True)  # The variable to exclude
            print()
            ifile_tmp = ifile[:-3]+'_extract.nc'
            Log = Ncdf(ifile_tmp, 'Edited in postprocessing')
            Log.copy_all_dims_from_Ncfile(f_IN)
            Log.copy_all_vars_from_Ncfile(f_IN, exclude_list)
            f_IN.close()
            Log.close()
            prCyan(ifile+' was created')

        # =================================================================
        # ============================ Add ================================
        # =================================================================
        # If the list is not empty, load ak and bk for thepressure calculation.
        # ak and bk are always needed.

        # Check if the variable to be added is currently supported.
        for ivar in add_list:
            if ivar not in VAR.keys():
                prRed("Variable '%s' is not supported and cannot be added to the file. " % (ivar))
            else:
                print('Processing: %s...' % (ivar))
                try:
                    fileNC = Dataset(ifile, 'a', format='NETCDF4_CLASSIC')
                    model=read_variable_dict_amescap_profile(fileNC)
                    f_type, interp_type = FV3_file_type(fileNC)
                    # Load ak and bk for pressure calculation. Usually required.
                    if interp_type == model.pfull:
                        ak, bk = ak_bk_loader(fileNC)
                    # 'temp' and 'ps' always required
                    # Get dimension
                    dim_out = fileNC.variables[model.temp].dimensions
                    temp = fileNC.variables[model.temp][:]
                    shape_out = temp.shape
                    if f_type == 'diurn':
                        # [time, tod, lev, lat, lon] -> [lev, tod, time, lat, lon] -> [time, tod, lev, lat, lon]
                        lev_T = [2, 1, 0, 3, 4]
                        # (0 1 2 3 4) -> (2 1 0 3 4) -> (2 1 0 3 4)
                        lev_T_out = [1, 2, 0, 3, 4]
                        # In 'diurn' file, 'level' is the 3rd axis: (time, tod, lev, lat, lon)
                        lev_axis = 2
                    else:
                        # [tim, lev, lat, lon] -> [lev, time, lat, lon] -> [tim, lev, lat, lon]
                        lev_T = [1, 0, 2, 3]
                        # (0 1 2 3) -> (1 0 2 3) -> (1 0 2 3)
                        lev_T_out = lev_T
                        # In 'average' and 'daily' files, 'level' is the 2nd axis: (time, lev, lat, lon)
                        lev_axis = 1

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # ~~~~~~~~~~~~  Non-Interpolated Files ~~~~~~~~~~~~~~~~~
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # 'level', 'ps', and 'p_3d' are often required.
                    if interp_type == model.pfull:
                        lev  = fileNC.variables[model.pfull][:]
                        ps   = fileNC.variables[model.ps][:]
                        p_3D = compute_p_3D(ps, ak, bk, shape_out)

                    # If file interpolated to 'pstd', calculate the 3D pressure field.
                    # This is quick and easy:
                elif interp_type == model.pstd:
                        lev = fileNC.variables[model.pstd][:]
                        reshape_shape = [1 for i in range(
                            0, len(shape_out))]  # (0 1 2 3)
                        reshape_shape[lev_axis] = len(lev)  # e.g [1, 28, 1, 1]
                        p_3D = lev.reshape(reshape_shape)
                    # If requested interp_type is 'zstd', or 'zagl', 'pfull3D' is required before interpolation.
                    # Some computations (e.g. wind speed) do not require 'pfull3D' and will work without it,
                    # so we use a 'try' statement here.
                    else:
                        try:
                            p_3D = fileNC.variables['pfull3D'][:]
                        except:
                            pass

                    if ivar == 'dzTau':
                        if 'dst_mass_micro' in fileNC.variables.keys():
                            q = fileNC.variables['dst_mass_micro'][:]
                        elif 'dst_mass' in fileNC.variables.keys():
                            q = fileNC.variables['dst_mass'][:]
                        OUT = compute_xzTau(q, temp, lev, C_dst, f_type)

                    if ivar == 'izTau':
                        if 'ice_mass_micro' in fileNC.variables.keys():
                            q = fileNC.variables['ice_mass_micro'][:]
                        elif 'ice_mass' in fileNC.variables.keys():
                            q = fileNC.variables['ice_mass'][:]
                        OUT = compute_xzTau(q, temp, lev, C_ice, f_type)

                    if ivar == 'dst_mass_micro':
                        xTau = fileNC.variables['dzTau'][:]
                        OUT = compute_mmr(xTau, temp, lev, C_dst, f_type)

                    if ivar == 'ice_mass_micro':
                        xTau = fileNC.variables['izTau'][:]
                        OUT = compute_mmr(xTau, temp, lev, C_ice, f_type)

                    if ivar == 'Vg_sed':
                        if 'dst_mass_micro' in fileNC.variables.keys():
                            xTau = fileNC.variables['dst_mass_micro'][:]
                            nTau = fileNC.variables['dst_num_micro'][:]
                        elif 'dst_mass' in fileNC.variables.keys():
                            xTau = fileNC.variables['dst_mass'][:]
                            nTau = fileNC.variables['dst_num'][:]
                        OUT = compute_Vg_sed(xTau, nTau, temp)

                    if ivar == 'w_net':
                        Vg = fileNC.variables['Vg_sed'][:]
                        wvar = fileNC.variables['w'][:]
                        OUT = compute_w_net(Vg, wvar)

                    if ivar == 'pfull3D':
                        OUT = p_3D

                    if ivar == 'DP':
                        OUT = compute_DP_3D(ps, ak, bk, shape_out)

                    if ivar == 'rho':
                        OUT = compute_rho(p_3D, temp)

                    if ivar == 'theta':
                        OUT = compute_theta(p_3D, ps, temp, f_type)

                    if ivar == 'w':
                        omega = fileNC.variables['omega'][:]
                        rho = compute_rho(p_3D, temp)
                        OUT = compute_w(rho, omega)

                    if ivar == 'zfull':
                        # TODO not with _pstd
                        OUT = compute_zfull(ps, ak, bk, temp)

                    if ivar == 'DZ':
                        OUT = compute_DZ_3D(ps, ak, bk, temp, shape_out)

                    if ivar == 'wspeed' or ivar == 'wdir':
                        ucomp = fileNC.variables['ucomp'][:]
                        vcomp = fileNC.variables['vcomp'][:]
                        theta, mag = cart_to_azimut_TR(
                            ucomp, vcomp, mode='from')
                        if ivar == 'wdir':
                            OUT = theta
                        if ivar == 'wspeed':
                            OUT = mag

                    if ivar == 'N':
                        theta = compute_theta(p_3D, ps, temp, f_type)
                        # TODO incompatible with 'pstd' files
                        zfull = compute_zfull(ps, ak, bk, temp)
                        OUT = compute_N(theta, zfull)

                    if ivar == 'Ri':
                        theta = compute_theta(p_3D, ps, temp, f_type)
                        # TODO incompatible with 'pstd' files
                        zfull = compute_zfull(ps, ak, bk, temp)
                        N = compute_N(theta, zfull)

                        ucomp = fileNC.variables['ucomp'][:]
                        vcomp = fileNC.variables['vcomp'][:]
                        du_dz = dvar_dh(ucomp.transpose(
                            lev_T), zfull.transpose(lev_T)).transpose(lev_T)
                        dv_dz = dvar_dh(vcomp.transpose(
                            lev_T), zfull.transpose(lev_T)).transpose(lev_T)
                        OUT = N**2/(du_dz**2+dv_dz**2)

                    if ivar == 'Tco2':
                        OUT = compute_Tco2(p_3D, temp)

                    if ivar == 'scorer_wl':
                        ucomp = fileNC.variables['ucomp'][:]
                        theta = compute_theta(p_3D, ps, temp, f_type)
                        zfull = compute_zfull(ps, ak, bk, temp)
                        N = compute_N(theta, zfull)
                        OUT = compute_scorer(N, ucomp, zfull)

                    if ivar in ['div', 'curl', 'fn']:
                        lat = fileNC.variables['lat'][:]
                        lon = fileNC.variables['lon'][:]
                        ucomp = fileNC.variables['ucomp'][:]
                        vcomp = fileNC.variables['vcomp'][:]

                    if ivar == 'div':
                        OUT = spherical_div(
                            ucomp, vcomp, lon, lat, R=3400*1000., spacing='regular')

                    if ivar == 'curl':
                        OUT = spherical_curl(
                            ucomp, vcomp, lon, lat, R=3400*1000., spacing='regular')

                    if ivar == 'fn':
                        theta = fileNC.variables['theta'][:]
                        OUT = frontogenesis(
                            ucomp, vcomp, theta, lon, lat, R=3400*1000., spacing='regular')

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    # ~~~~~~~~~~~~~~~~~ Interpolated files ~~~~~~~~~~~~~~~~~~~
                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # All interpolated files have the following
                    if interp_type != 'pfull':
                        lev = fileNC.variables[interp_type][:]

                    if ivar == 'msf':
                        vcomp = fileNC.variables['vcomp'][:]
                        lat = fileNC.variables['lat'][:]
                        if f_type == 'diurn':
                            # [lev, lat, time, tod, lon] -> [time, tod, lev, lat, lon]
                            # (0 1 2 3 4) -> (2 3 0 1 4) -> (2 3 0 1 4)
                            OUT = mass_stream(vcomp.transpose(
                                [2, 3, 0, 1, 4]), lat, lev, type=interp_type).transpose([2, 3, 0, 1, 4])
                        else:
                            OUT = mass_stream(vcomp.transpose(
                                [1, 2, 3, 0]), lat, lev, type=interp_type).transpose([3, 0, 1, 2])
                            # [time, lev, lat, lon] -> [lev, lat, lon, time]  ->  [time, lev, lat, lon]
                            # (0 1 2 3) -> (1 2 3 0) -> (3 0 1 2)

                    if ivar == 'ep':
                        OUT = compute_Ep(temp)

                    if ivar == 'ek':
                        ucomp = fileNC.variables['ucomp'][:]
                        vcomp = fileNC.variables['vcomp'][:]
                        OUT = compute_Ek(ucomp, vcomp)

                    if ivar == 'mx':
                        OUT = compute_MF(
                            fileNC.variables['ucomp'][:], fileNC.variables['w'][:])

                    if ivar == 'my':
                        OUT = compute_MF(
                            fileNC.variables['vcomp'][:], fileNC.variables['w'][:])

                    if ivar == 'ax':
                        mx = compute_MF(
                            fileNC.variables['ucomp'][:], fileNC.variables['w'][:])
                        rho = fileNC.variables['rho'][:]
                        OUT = compute_WMFF(mx, rho, lev, interp_type)

                    if ivar == 'ay':
                        my = compute_MF(
                            fileNC.variables['vcomp'][:], fileNC.variables['w'][:])
                        rho = fileNC.variables['rho'][:]
                        OUT = compute_WMFF(my, rho, lev, interp_type)

                    if ivar == 'tp_t':
                        OUT = zonal_detrend(temp)/temp

                    # Filter out NANs in the native files
                    if interp_type == 'pfull':
                        OUT[np.isnan(OUT)] = fill_value

                    # Add NANs to the interpolated files
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                "ignore", category=RuntimeWarning)
                            OUT[OUT > 1.e30] = np.NaN
                            OUT[OUT < -1.e30] = np.NaN

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(ivar, 'f4', dim_out)
                    var_Ncdf.long_name = VAR[ivar][0]
                    var_Ncdf.units = VAR[ivar][1]
                    var_Ncdf[:] = OUT
                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % (ivar))

                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow(
                            """Delete the existing variables %s with 'MarsVars.py %s -rm %s'""" % (ivar, ifile, ivar))

        # =================================================================
        # ================== Vertical Differentiation =====================
        # =================================================================
        for idiff in zdiff_list:
            fileNC = Dataset(ifile, 'a', format='NETCDF4_CLASSIC')
            f_type, interp_type = FV3_file_type(fileNC)

            if interp_type == 'pfull':
                ak, bk = ak_bk_loader(fileNC)

            if idiff not in fileNC.variables.keys():
                prRed("zdiff error: variable '%s' is not present in %s" %
                      (idiff, ifile))
                fileNC.close()
            else:
                print('Differentiating: %s...' % (idiff))
                if f_type == 'diurn':
                    lev_T = [2, 1, 0, 3, 4]
                else:  # [time, lat, lon]
                    lev_T = [1, 0, 2, 3]  # [tim, lev, lat, lon]
                try:
                    var = fileNC.variables[idiff][:]
                    longname_txt, units_txt = get_longname_units(fileNC, idiff)
                    # Remove the last ']' to update the units (e.g '[kg]' to '[kg/m]')
                    newUnits = units_txt[:-2]+'/m]'
                    newLong_name = 'vertical gradient of '+longname_txt
                    # Alex's version of the above 2 lines:
                    # remove the last ']' to update units, (e.g '[kg]' to '[kg/m]')
                    #newUnits = getattr(fileNC.variables[idiff],'units','')[:-2]+'/m]'
                    #newLong_name = 'vertical gradient of ' + getattr(fileNC.variables[idiff], 'long_name', '')

                    # 'temp' and 'ps' are always required
                    # Get dimension
                    dim_out = fileNC.variables['temp'].dimensions
                    if interp_type == 'pfull':
                        if 'zfull' in fileNC.variables.keys():
                            zfull = fileNC.variables['zfull'][:]
                        else:
                            temp = fileNC.variables['temp'][:]
                            ps = fileNC.variables['ps'][:]
                            zfull = fms_Z_calc(ps, ak, bk, temp.transpose(
                                lev_T), topo=0., lev_type='full')  # Z is the first axis
                        # 'average' file: zfull = (lev, time, lat, lon)
                        # 'diurn' file:   zfull = (lev, tod, time, lat, lon)
                        # Differentiate the variable w.r.t. Z:
                        darr_dz = dvar_dh(var.transpose(
                            lev_T), zfull).transpose(lev_T)

                    elif interp_type == 'pstd':
                        # If 'pstd', requires 'zfull'
                        if 'zfull' in fileNC.variables.keys():
                            zfull = fileNC.variables['zfull'][:]
                            darr_dz = dvar_dh(var.transpose(
                                lev_T), zfull.transpose(lev_T)).transpose(lev_T)
                        else:
                            lev = fileNC.variables[interp_type][:]
                            temp = fileNC.variables['temp'][:]
                            dzfull_pstd = compute_DZ_full_pstd(lev, temp)
                            darr_dz = dvar_dh(var.transpose(
                                lev_T)).transpose(lev_T)/dzfull_pstd

                    elif interp_type in ['zagl', 'zstd']:
                        lev = fileNC.variables[interp_type][:]
                        darr_dz = dvar_dh(var.transpose(
                            lev_T), lev).transpose(lev_T)

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(
                        'd_dz_'+idiff, 'f4', dim_out)
                    var_Ncdf.long_name = newLong_name
                    var_Ncdf.units = newUnits
                    var_Ncdf[:] = darr_dz
                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % ('d_dz_'+idiff))
                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow("""Delete the existing variable %s with 'MarsVars %s -rm %s'""" %
                                 ('d_dz_'+idiff, ifile, 'd_dz_'+idiff))

        # =================================================================
        # ====================== Zonal Detrending =========================
        # =================================================================
        for izdetrend in zdetrend_list:
            fileNC = Dataset(ifile, 'a', format='NETCDF4_CLASSIC')
            f_type, interp_type = FV3_file_type(fileNC)
            if izdetrend not in fileNC.variables.keys():
                prRed("zdiff error: variable '%s' is not in %s" %
                      (izdetrend, ifile))
                fileNC.close()
            else:
                print('Detrending: %s...' % (izdetrend))

                try:
                    var = fileNC.variables[izdetrend][:]
                    longname_txt, units_txt = get_longname_units(
                        fileNC, izdetrend)
                    newLong_name = 'zonal perturbation of '+longname_txt
                    # Alex's version of the above (and below) lines:
                    #newUnits = getattr(fileNC.variables[izdetrend], 'units', '')
                    #newLong_name = 'zonal perturbation of ' + getattr(fileNC.variables[izdetrend], 'long_name', '')

                    # Get dimension
                    dim_out = fileNC.variables[izdetrend].dimensions

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(
                        izdetrend+'_p', 'f4', dim_out)
                    var_Ncdf.long_name = newLong_name
                    var_Ncdf.units = units_txt
                    #var_Ncdf.units = newUnits # alex's version
                    var_Ncdf[:] = zonal_detrend(var)
                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % (izdetrend+'_p'))
                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow("""Delete the existing variable %s with 'MarsVars %s -rm %s'""" %
                                 ('d_dz_'+idiff, ifile, 'd_dz_'+idiff))

        # =================================================================
        # ========= Opacity Conversion (dp_to_dz and dz_to_dp) ============
        # =================================================================
        # ========= Case 1: dp_to_dz
        for idp_to_dz in dp_to_dz_list:
            fileNC = Dataset(ifile, 'a', format='NETCDF4_CLASSIC')
            f_type, interp_type = FV3_file_type(fileNC)
            if idp_to_dz not in fileNC.variables.keys():
                prRed("dp_to_dz error: variable '%s' is not in %s" %
                      (idp_to_dz, ifile))
                fileNC.close()
            else:
                print('Converting: %s...' % (idp_to_dz))

                try:
                    var = fileNC.variables[idp_to_dz][:]
                    newUnits = getattr(
                        fileNC.variables[idp_to_dz], 'units', '')+'/m'
                    newLong_name = getattr(
                        fileNC.variables[idp_to_dz], 'long_name', '')+' rescaled to meter-1'
                    # Get dimension
                    dim_out = fileNC.variables[idp_to_dz].dimensions

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(
                        idp_to_dz+'_dp_to_dz', 'f4', dim_out)
                    var_Ncdf.long_name = newLong_name
                    var_Ncdf.units = newUnits
                    var_Ncdf[:] = var*fileNC.variables['DP'][:] / \
                        fileNC.variables['DZ'][:]
                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % (idp_to_dz+'_dp_to_dz'))
                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow("""Delete the existing variable %s with 'MarsVars %s -rm %s'""" %
                                 (idp_to_dz+'_dp_to_dz', ifile, idp_to_dz+'_dp_to_dz'))

       # ========= Case 2: dz_to_dp
        for idz_to_dp in dz_to_dp_list:
            fileNC = Dataset(ifile, 'a', format='NETCDF4_CLASSIC')
            f_type, interp_type = FV3_file_type(fileNC)
            if idz_to_dp not in fileNC.variables.keys():
                prRed("dz_to_dp error: variable '%s' is not in %s" %
                      (idz_to_dp, ifile))
                fileNC.close()
            else:
                print('Converting: %s...' % (idz_to_dp))

                try:
                    var = fileNC.variables[idz_to_dp][:]
                    newUnits = getattr(
                        fileNC.variables[idz_to_dp], 'units', '')+'/m'
                    newLong_name = getattr(
                        fileNC.variables[idz_to_dp], 'long_name', '')+' rescaled to Pa-1'
                    # Get dimension
                    dim_out = fileNC.variables[idz_to_dp].dimensions

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(
                        idz_to_dp+'_dz_to_dp', 'f4', dim_out)
                    var_Ncdf.long_name = newLong_name
                    var_Ncdf.units = newUnits
                    var_Ncdf[:] = var*fileNC.variables['DZ'][:] / \
                        fileNC.variables['DP'][:]
                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % (idz_to_dp+'_dz_to_dp'))
                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow("""Delete the existing variable %s with 'MarsVars.py %s -rm %s'""" %
                                 (idp_to_dz+'_dp_to_dz', ifile, idp_to_dz+'_dp_to_dz'))

        # =================================================================
        # ====================== Column Integration =======================
        # =================================================================
        """
                          z_top
                          ⌠
        We have col=      ⌡ var (rho dz)  with [(dp/dz) = (-rho g)] => [(rho dz) = (-dp/g)]
                          0

                      ___ p_sfc
             >  col = \
                      /__ var (dp/g)
                        p_top
        """

        for icol in col_list:
            fileNC = Dataset(ifile, 'a')  # , format='NETCDF4_CLASSIC
            f_type, interp_type = FV3_file_type(fileNC)
            if interp_type == 'pfull':
                ak, bk = ak_bk_loader(fileNC)

            if icol not in fileNC.variables.keys():
                prRed("column integration error: variable '%s' is not in %s" % (
                    icol, ifile))
                fileNC.close()
            else:
                print('Performing column integration: %s...' % (icol))

                try:
                    var = fileNC.variables[icol][:]
                    longname_txt, units_txt = get_longname_units(fileNC, icol)
                    newUnits = units_txt[:-3]+'/m2'  # turn 'kg/kg'> to 'kg/m2'
                    newLong_name = 'column integration of '+longname_txt
                    # Alex's version of the above 2 lines:
                    #newUnits = getattr(fileNC.variables[icol], 'units', '')[:-3]+'/m2' # 'kg/kg' -> 'kg/m2'
                    #newLong_name = 'column integration of '+getattr(fileNC.variables[icol], 'long_name', '')

                    # 'temp' and 'ps' always required
                    # Get dimension
                    dim_in = fileNC.variables['temp'].dimensions
                    shape_in = fileNC.variables['temp'].shape
                    # TODO edge cases where time = 1
                    if f_type == 'diurn':
                        # [time, tod, lat, lon]
                        lev_T = [2, 1, 0, 3, 4]  # [time, tod, lev, lat, lon]
                        dim_out = tuple(
                            [dim_in[0], dim_in[1], dim_in[3], dim_in[4]])
                        # In 'diurn', 'level' is the 3rd axis: (time, tod, lev, lat, lon)
                        lev_axis = 2
                    else:  # [time, lat, lon]
                        lev_T = [1, 0, 2, 3]  # [time, lev, lat, lon]
                        dim_out = tuple([dim_in[0], dim_in[2], dim_in[3]])
                        lev_axis = 1

                    ps = fileNC.variables['ps'][:]
                    DP = compute_DP_3D(ps, ak, bk, shape_in)
                    out = np.sum(var*DP/g, axis=lev_axis)

                    # Log the variable
                    var_Ncdf = fileNC.createVariable(
                        icol+'_col', 'f4', dim_out)
                    var_Ncdf.long_name = newLong_name
                    var_Ncdf.units = newUnits
                    var_Ncdf[:] = out

                    fileNC.close()

                    print('%s: \033[92mDone\033[00m' % (icol+'_col'))
                except Exception as exception:
                    if debug:
                        raise
                    if str(exception) == 'NetCDF: String match to name in use':
                        prYellow("""***Error*** Variable already exists in file.""")
                        prYellow("""Delete the existing variable %s with 'MarsVars %s -rm %s'""" %
                                 (icol+'_col', ifile, icol+'_col'))
        if edit_var:
            f_IN = Dataset(ifile, 'r', format='NETCDF4_CLASSIC')
            ifile_tmp = ifile[:-3]+'_tmp.nc'
            Log = Ncdf(ifile_tmp, 'Edited in postprocessing')
            Log.copy_all_dims_from_Ncfile(f_IN)
            # Copy all variables but this one
            Log.copy_all_vars_from_Ncfile(f_IN, exclude_var=edit_var)
            # Read value, longname, units, name, and log the new variable
            var_Ncdf = f_IN.variables[edit_var]

            name_txt = edit_var
            vals = var_Ncdf[:]
            dim_out = var_Ncdf.dimensions
            longname_txt = getattr(var_Ncdf, 'long_name', '')
            units_txt = getattr(var_Ncdf, 'units', '')
            cart_txt = getattr(var_Ncdf, 'cartesian_axis', '')

            if parser.parse_args().rename:
                name_txt = parser.parse_args().rename
            if parser.parse_args().longname:
                longname_txt = parser.parse_args().longname
            if parser.parse_args().unit:
                units_txt = parser.parse_args().unit
            if parser.parse_args().multiply:
                vals *= parser.parse_args().multiply

            if cart_txt == '':
                Log.log_variable(name_txt, vals, dim_out,
                                 longname_txt, units_txt)
            else:
                Log.log_axis1D(name_txt, vals, dim_out,
                               longname_txt, units_txt, cart_txt)

            f_IN.close()
            Log.close()

            # Rename the new file
            cmd_txt = 'mv '+ifile_tmp+' '+ifile
            subprocess.call(cmd_txt, shell=True)

            prCyan(ifile+' was updated')


if __name__ == '__main__':
    main()
#
