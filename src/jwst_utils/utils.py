import numpy as np
import os
from astropy.io import fits
from astropy.visualization import LogStretch, AsinhStretch, ImageNormalize, ManualInterval, make_lupton_rgb

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from scipy.signal import medfilt,savgol_filter, find_peaks
import copy

# define a function read fits file
def read_fits(filename: str):

    '''
    Read a FITS file and return the data, header, and error.
    Parameters:
    -----------
    filename : str
        The name of the FITS file to read
    Returns:
    -----------
    data : astropy.io.fits.FITS_rec
        The data from the FITS file
    header : astropy.io.fits.Header
        The header from the FITS file
    err : astropy.io.fits.FITS_rec
        The error data from the FITS file
    '''

    hdul = fits.open(filename)
    data = hdul[1].data
    header = hdul[1].header
    err = hdul[2].data
    hdul.close()
    return data, header ,err

# define a function to create a dictionary to store data
def create_data_dict(fitls: list, name:list | None = None, check_header:bool = False, 
                     data_dir:str|None = None, check_file_header:dict | None = None):

    '''
    Create a dictionary to store data from FITS files.
    Parameters:
    -----------
    fitls : list
        List of FITS file names
    name : list, optional
        List of source names corresponding to the FITS files
    check_header : bool, optional
        If True, print the header of FITS file for verification
    data_dir : str, optional
        The directory where the FITS files are located
    check_file_header : str, optional
        The FITS file name to check the header if check_header is True
    
    Returns:
    -----------
    data_dict : dict
        Dictionary containing data, header, and error for each source
    '''
    data_dict = {}

    # Whether to check the header of a specific FITS file
    if (check_header is True):

        
        # Check the header of the specified FITS file
        if (check_file_header is None):
            print("Error: Please specify the FITS file name to check the header.")
            return None

        
        # Read and print the header of the specified FITS file
        if (check_file_header is not None):
            if data_dir is not None:
                check_file_header = os.path.join(data_dir, check_file_header)
            data, header, err = read_fits(check_file_header)
            print(repr(header))
    else:
        pass

    if name is not None:
        # Check the length of the name list is equal to fitls
        try:
            assert len(name) == len(fitls)
        except AssertionError:
            print("Error: The length of the name list must be equal to the length of the fitls list.")
            return None
        
        for i, file in enumerate(fitls):

            # The following two lines make sure that file variable only contains the file name instead of the full path
            file_path = os.path.join(data_dir, file) if data_dir is not None else file

            source = name + '.' + file.split('.')[-1]
            data, header, err = read_fits(file_path)
            data_dict['%s' %source] = {'header': header, 'data': data, 'error': err}
    else:
        for i, file in enumerate(fitls):

            file_path = os.path.join(data_dir, file) if data_dir is not None else file
            name = file.split('-')[0] # this is only useful for TMC1A JWST files
            suffix = file.split('.')[0][-3:]
            #print(name+'.'+suffix)
            source = name + '.' +suffix
            data, header, err = read_fits(file_path)
            data_dict['%s' %source] = {'header': header, 'data': data, 'error': err}
    print('data_dict keys:', data_dict.keys())
    
    return data_dict

def optical_veocity(wave: np.ndarray, header:dict, restwave: float = 1.644, c:float = 3.e5):

    '''
    Convert wavelength array to optical velocity (units - km/s).
    Parameters:
    -----------
    wave : 1D array
        Wavelength array
    header : astropy.io.fits.Header
        The header containing velocity correction information
    restwave : float, optional
        Rest wavelength of the emission line for velocity calculation. Here we assume the line is FeII 1.644 micron meter.
    c : float, optional
        Speed of light in km/s
    Returns:
    -----------
    vopt : 1D array
        Optical velocity array
    '''

    v_barycorr = header['VELOSYS']
    vopt = (wave - restwave)/restwave * c + v_barycorr/1000
    return vopt

# define a function to plot an image
def plot_image(data: np.ndarray, header: dict, vmin: float, vmax: float, a: float = 1e-2, text:str = '-70km/s', savefig = False, save_name:str = 'overview.png') -> None:
    
    '''
    Plot an image with given data and header on the provided axis.
    Parameters:
    -----------
    data : 2D array
        The image data to plot
    header : astropy.io.fits.Header
        The header containing WCS information
    vmin : float
        Minimum value for color scaling
    vmax : float
        Maximum value for color scaling
    a : float, optional
        Parameter for AsinhStretch
    text : str, optional
        Text (the velocity or channel) to display on the image
    savefig : bool, optional
        If True, save the figure
    save_name : str, optional
        The name of the file to save the figure as if savefig is True.
    Returns:
    -----------
    cbar1 : matplotlib.colorbar.Colorbar
        The colorbar associated with the image
    '''

    fig = plt.figure(figsize = (8,10))

    widths = [0.05,1]
    heights = [0.05,1]
    gs = fig.add_gridspec(2, 2, width_ratios=widths,height_ratios=heights)
    gs.update(left=0.05, right=0.95, bottom=0.08, top=0.85, wspace=0.02, hspace=0.02)
    
    pixel = header['CDELT2']*3600  # arcsec/pixel
    ax1 = fig.add_subplot(gs[1,1])
    divider1 = make_axes_locatable(ax1)
    cbar = divider1.append_axes("top", size="5%", pad=0.1)

    # Change the ticks to arcsec
    def axis_transfer(pos,val):
        return f'{pos*pixel:.2f}'

    norm1 = ImageNormalize(data, vmin= vmin, vmax = vmax, stretch=AsinhStretch(a = a))
    plt1 = ax1.imshow(data, origin='lower', norm = norm1, cmap='inferno')
    cb1 = Colorbar(ax = cbar, mappable = plt1, orientation = 'horizontal', ticklocation = 'top')
    
    ax1.xaxis.set_major_formatter(FuncFormatter(axis_transfer))
    ax1.yaxis.set_major_formatter(FuncFormatter(axis_transfer))
    ax1.set_xlabel(r'$\Delta \alpha$' +' [arcsec]', fontsize = 15)
    ax1.set_ylabel(r'$\Delta \delta$'+' [arcsec]',fontsize = 15)
    ax1.text(0.05,0.95, s = text, transform = ax1.transAxes, fontsize = 15)
    cbar.set_title(r'$I_\nu$'+' [MJy/Sr]',fontsize = 25)
    plt.show()

    if savefig == True:
        fig.savefig(save_name)
    
def calc_spec(data:np.ndarray, header:dict):
    ''' 
    Calculate the flux in each channel without continuum subtraction by masking out the Nan value.
    Parameters:
    ------------
    data: 3D array
        Data cube
    header: astropy.io.fits.Header
        The header containing pixel information
    
    Returns:
    ------------
    wave: 1D Wavelength array.
    flux: 1D Flux array.
        
    '''
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    
    pixel = header['CDELT2']*3600  # arcsec/pixel
    omega = (pixel/3600*np.pi/180)**2 # Set the solid angle to pixel
    print('Sr in pixel:', omega)
    flux = []
    wave = []
    wave0 = header['CRVAL3']
    delta_wave = header['CDELT3']
    print('Begin to calculate spectrum flux without continuum subtraction...')
    # mask out the Nan value in each channel of data
    for i in tqdm(range(len(data)), desc="Calculating spectrum flux"):
        mask_valid = ~np.isnan(data)
        masked_slice = np.where(mask_valid[i], data[i], 0.0)
        f_w = np.sum(masked_slice) * 1e6 * omega
    
        flux.append(f_w)
        w = wave0 + i * delta_wave
        wave.append(w)

    wave = np.asarray(wave)
    flux = np.asarray(flux)
    print('Finish calculate spectrum flux without continuum subtraction!!\n' \
    'Return the wave and flux arrays.')

    return wave, flux

def plot_line_spectrum(wave:np.ndarray, header: dict, flux:np.ndarray, xlim:tuple, ylim:tuple, 
                       text:tuple = ('1.644', 'FeII'), restwave:float = 1.644, plot_all_spec:bool = False,
                        savefig:bool = False, save_name:str = 'FeII1644_spectrum.png') -> None:
    
    '''
    Plot the specific emmission line's spectrum or all spectrum. Remember to specify the rest wavelength in the optical_velocity function.

    Parameters:
    -----------
    wave : 1D array
        Wavelength array
    header : astropy.io.fits.Header
        The header containing velocity correction information
    flux : 1D array
        Flux array
    xlim : tuple
        x-axis limits for the line spectrum plot
    ylim : tuple
        y-axis limits for the line spectrum plot
    text : tuple, optional
        Text (wavelength and line name) to display on the plot title
    plot_all_spec : bool, optional
        If True, plot the full spectrum in wavelength space
    savefig : bool, optional
        If True, save the figure
    save_name : str, optional
        The name of the file to save the figure as if savefig is True.

    Returns:
    -----------
    None. Show the plots.
    '''

    if not isinstance(wave, np.ndarray):
        raise TypeError("wave must be a numpy.ndarray")
    
    if not isinstance(flux, np.ndarray):
        raise TypeError("flux must be a numpy.ndarray")
    
    # Plot the full spectrum in wavelength space
    if plot_all_spec == True:
        
        fig = plt.figure(figsize = (9,5))
        plt.step(wave, np.array(flux), where = 'mid')
        #plt.xlim(1.64,1.65)
        #plt.ylim(0,50)
        plt.xlabel('wavelength ' + r'$[\mu m]$', fontsize = 15)
        plt.ylabel(r'$F_\lambda$'+' [Jy]', fontsize = 15)
        plt.title('Spectrum', fontsize = 15)
        plt.show()

        if savefig == True:
            fig.savefig('full_spectrum.png')
    
    # Plot the line spectrum in velocity space

    v = optical_veocity(wave, header, restwave = restwave) # convert wavelength to velocity
    xmin, xmax = xlim[0], xlim[1]
    ymin, ymax = ylim[0], ylim[1]

    fig = plt.figure(figsize = (9,5))
    plt.step(v, flux, where='mid')
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)
    plt.xlabel('velocity ' + r'$[km/s]$', fontsize = 15)
    plt.ylabel(r'$F_\lambda$'+' [Jy]', fontsize = 15)
    plt.title('%s ' %(text[0]) + r'$\mu m$'+' %s' %(text[1]) + ' Spectrum'  , fontsize = 15)
    plt.show()

    if savefig == True:
        fig.savefig(save_name)

def calc_cont(wave:np.ndarray,flux:np.ndarray, niter:int =3, boxsize:int = 9, exclude:list = None, 
              threshold: float =0.998, offset=0., spike_threshold:float =None):
    '''2025Fall_ISM_Project
    Calculate the continuum of a spectrum using an iterative median filtering method and Savitzky-Golay smoothing.
    Parameters:
    -----------
    wave : 1D array
        Wavelength array
    flux : 1D array
        Flux array
    niter : int, optional
        Number of iterations for continuum fitting
    boxsize : int, optional
        Size of the median filter box
    exclude : list of tuples, optional
        List of wavelength ranges to exclude from fitting (e.g., [(start1, end1), (start2, end2)])
    threshold : float, optional
        Threshold for setting the anchor points
    offset : float, optional
        Offset to add to the final continuum
    spike_threshold : float, optional
        Threshold for identifying and removing negative spikes
    Returns:
    -----------
    cont : 1D array
        The calculated continuum array
    '''

    flux_tmp= copy.deepcopy(flux)

    #Remove negative spikes from consideration
    if(spike_threshold is not None):
        bad_pix, _ = find_peaks(-1*flux_tmp, prominence=spike_threshold)
        flux_tmp[bad_pix] = np.nan

    #Exclude regions    
    if(exclude is not None):
        for myexclude in exclude:  #Exclude regions from fitting
            localbool=((wave>myexclude[0]) & (wave<myexclude[1]))
            flux_tmp[localbool]=np.nan   
            
    #Perform continuum determination        
    cont = copy.deepcopy(flux_tmp)  

    # Fill NaNs by linear interp (only if we have anchors)
    good = np.isfinite(cont)
    if good.sum() >= 2:
        cont[~good] = np.interp(wave[~good], wave[good], cont[good])
    else:
        return np.full_like(cont, np.nan)  
    
    for _ in range(max(1, int(niter))):
        smooth = medfilt(cont.astype(np.float64), boxsize)
        valid  = np.isfinite(smooth) & np.isfinite(cont)
        anchor = valid & (smooth > cont * float(threshold))

        # If too few anchors, relax once; if still too few, fall back to smooth
        #if anchor.sum() < 2:
        #    anchor = valid & (smooth > cont * 0.99)
        #if anchor.sum() < 2:
        #    cont = smooth
        #    break

        cont = np.interp(wave, wave[anchor], cont[anchor])

    # The following makes sure that the window is large enough but not larger than the total length
    sg_window = min(boxsize if  boxsize% 2 else (boxsize+1), max(boxsize*3, len(flux_tmp) - (1 - len(flux_tmp) % 2)))
    if sg_window >= 10 and sg_window <= len(flux_tmp):
        # 10 is just a number to limit the lower bound
        cont = savgol_filter(cont, sg_window, polyorder=1,mode='interp')
    
    #Apply offset
    cont+=offset
    
    return cont

def generate_line_cont(data:np.ndarray, wave:np.ndarray, boxsize:int = 9,  exclude=[(1.432, 1.476), (1.869, 1.884)]):
    '''
    Generate continuum and line emission data cubes from the input data cube.
    Parameters:
    -----------
    data : 3D array
        The input data cube with shape (wavelength, y, x)
    wave : 1D array
        The wavelength array corresponding to the first dimension of the data cube
    boxsize : int, optional
        Size of the median filter box for continuum calculation
    Returns:
    -----------
    cont_spaxel : 3D array
        The continuum emission data cube
    line_spaxel : 3D array
        The line emission data cube
    '''
    cont_spaxel = np.zeros_like(data)
    line_spaxel = np.zeros_like(data)

    for j in tqdm(range(data.shape[1]), desc="Generating continuum and line emission cubes"):
        for i in range(data.shape[2]):
            check = data[:,j,i]
            #print('pre: ',check)
            if np.all(check == 0):
                continue
            #print('post: ',check)
            cont_spaxel[:,j,i] = calc_cont(wave, data[:,j,i], boxsize = boxsize, exclude= exclude)
            line_spaxel[:,j,i] = data[:,j,i] - cont_spaxel[:,j,i]
    
    return cont_spaxel, line_spaxel

# Plot the spectrum with conntinuum subtraction, g140
def calc_contsubtract_spec(line_spaxel: np.ndarray, cont_spaxel: np.ndarray, header:dict):
    ''' 
    Calculate the flux in each channel by masking out the Nan value.
    Parameters:
    ------------
    line_spaxel: 3D array
        Line emission data cube
    cont_spaxel: 3D array
        Continuum emission data cube
    header: astropy.io.fits.Header
        The header containing pixel information
    
    Returns:
    ------------
    wave: 1D Wavelength array.
    flux: 1D Flux array.
        
    '''
    if not isinstance(line_spaxel, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    
    pixel = header['CDELT2']*3600  # arcsec/pixel
    omega = (pixel/3600*np.pi/180)**2 # Set the solid angle to pixel
    print('Sr in pixel:', omega)
    flux_line = []
    noise = []
    cont = []

    # mask out the Nan value in each channel of data
    for i in tqdm(range(len(line_spaxel)), desc="Calculating spectrum flux with continuum subtraction"):
        mask_valid = ~np.isnan(line_spaxel)
        cont_mask_valid = ~np.isnan(cont_spaxel)
        masked_slice = np.where(mask_valid[i], line_spaxel[i], 0.0)
        cont_masked_slice = np.where(cont_mask_valid[i],cont_spaxel[i], 0.0)
        f_w = np.sum(masked_slice) * 1e6 * omega # flux at each wave
        cont_w = np.sum(cont_masked_slice) * 1e6 * omega

        flux_line.append(f_w)
        cont.append(cont_w)

    flux_line = np.asarray(flux_line)
    cont = np.asarray(cont)

    return cont, flux_line

# Plot the comparison between line and continuum emissions
def plot_comparison_line_continuum(v:np.ndarray, wave:np.ndarray, flux:np.ndarray, flux_line:np.ndarray, cont: np.ndarray, plot_all_spec = False, savefig = False, save_name = 'Comparison_line_cont.png') -> None:
    '''
    Plot the comparison between line and continuum emissions in both velocity and wavelength space.
    Parameters:
    -----------
    v : 1D array
        Optical velocity array
    wave : 1D array
        Wavelength array
    flux : 1D array
        Flux array
    flux_line : 1D array
        Continuum-subtracted flux array
    cont : 1D array
        Continuum flux array
    Returns:
    -----------
    None. Show the plots.
    '''
    fig = plt.figure(figsize = (9,5))
    plt.step(v, np.array(flux), where = 'mid', label = 'spectrum')
    plt.step(v, np.array(flux_line), where = 'mid', label = 'continuum-subtracted')
    plt.plot(v, np.array(cont), label = 'continuum')
    plt.xlim(-1000,800) # adjust velocity limits as needed
    plt.ylim(0,1.2e-2)
    plt.xlabel('velocity ' + r'$[km/s]$', fontsize = 15)
    plt.ylabel(r'$F_\lambda$'+' [Jy]', fontsize = 15)
    plt.title('Comparison between line and continuum emissions', fontsize = 15)
    plt.legend()
    plt.show()

    if savefig == True:
        fig.savefig(save_name)
    
    # Plot the full spectrum in wavelength space
    if plot_all_spec == True:
        fig = plt.figure(figsize = (9,5))
        plt.step(wave, np.array(flux), where = 'mid', label = 'spectrum')
        plt.step(wave, np.array(flux_line), where = 'mid', label = 'continuum-subtracted')
        plt.plot(wave, np.array(cont), label = 'continuum')
        #plt.xlim(-800,800)
        #plt.ylim(0,1.2e-2)
        plt.xlabel('wavelength ' + r'$[\mu m]$', fontsize = 15)
        plt.ylabel(r'$F_\lambda$'+' [Jy]', fontsize = 15)
        plt.title('Comparison between line and continuum emissions', fontsize = 15)
        plt.legend()
        plt.show()
        if savefig == True:
            fig.savefig('Comparison_line_cont_allspec.png')


def add_numbers(a, b):
    """Simple example function."""
    return a + b