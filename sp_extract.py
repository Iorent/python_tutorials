from __future__ import division

from struct import unpack

import sys
import numpy as np
from matplotlib.collections import LineCollection
from pylab import *
import argparse
import os
import math

import scipy.stats as ss

#Photometric Constants
global c1,c2,c3,l,cosi,cose,xl_fixed
c1 = -0.019
c2 = 0.000242
c3 = -.00000146
l = 1.0 + (c1*(30)) + (c2*(30**2)) + (c3*(30**3))
cosi = math.cos(math.radians(30))
cose = math.cos(math.radians(0))
xl_fixed= ((2*l*(cosi /(cosi + cose)))) + ((1-l)*cosi)

def openreferenceimage(dirpath, fname):
    from osgeo import gdal
    imagepath = os.path.join(dirpath, fname)
    imagepath = imagepath.split('.')[0]

    try:
        ds = gdal.Open(imagepath + 'P.jpg')
    except Exception:
        ds = gdal.Open(imagepath +'.jpg')
    bandcount = ds.RasterCount
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    img = ds.GetRasterBand(1).ReadAsArray()

    return ds, img

def openspc(input_data):

    """
    Parameters
    ----------

    input_data : string
                 This is the .spc file that contains the label and data.

    Returns
    --------
    wavelength :  array
                  An array of wavelengths from all 3 detectors

    radiance :  array
                An array of radiance values over the image.
                This is binned into n observations.

    reflectance : array
                  An array of reflectance values over the image.
                  This is binned into n observations.

    #TODO: Rewrite to use regex - this is old skool ghetto
    """

    label = open(input_data, 'r+b')
    for line in label:
        if "^SP_SPECTRUM_WAV" in line:
            wav_offset = int(line.split('=')[1].split(" ")[1])
        if "^SP_SPECTRUM_RAD" in line:
            rad_offset = int(line.split('=')[1].split(" ")[1])
        if "^SP_SPECTRUM_REF" in line:
            ref_offset = int(line.split('=')[1].split(" ")[1])
        if "^SP_SPECTRUM_QA" in line:
            qa_offset = int(line.split('=')[1].split(" ")[1])
        if "^L2D_RESULT_ARRAY" in line:
            l2d_offset = int(line.split('=')[1].split(" ")[1])
        if "OBJECT                               = SP_SPECTRUM_RAD" in line:
            line = label.next()
            rad_lines = int(line.split('=')[1])
        if "OBJECT                               = SP_SPECTRUM_REF" in line:
            line = label.next()
            ref_lines = int(line.split('=')[1])
        if 'NAME                         = "EMISSION_ANGLE"' in line:
            line = label.next();line = label.next(); line=label.next()
            emission_offset = int(line.split("=")[1])
        if 'NAME                         = "INCIDENCE_ANGLE"' in line:
            line = label.next();line = label.next(); line=label.next()
            incidence_offset = int(line.split("=")[1])
        if 'NAME                         = "PHASE_ANGLE"' in line:
            line = label.next();line = label.next(); line=label.next()
            phase_offset = int(line.split("=")[1])
        if 'NORMAL_SP_POINT_NUM' in line:
            num_observations = int(line.split("=")[1])
        if 'ROW_BYTES' in line:
            row_bytes = int(line.split("=")[1])
        if "^ANCILLARY_AND_SUPPLEMENT_DATA" in line:
            ancillary_offset = int(line.split("=")[1].split("<")[0])
        if "OBJECT                               = SP_SPECTRUM_QA" in line:
            line = label.next()
            qa_lines = int(line.split('=')[1])
        if "UPPER_LEFT_LATITUDE" in line:
            ullat = float(line.split("=")[1].split()[0])
        if "LOWER_LEFT_LATITUDE" in line:
            lllat = float(line.split("=")[1].split()[0])
        if "END_OBJECT                           = L2D_RESULT_ARRAY" in line:
            #Last line before binary
            break


    #Wavelength
    label.seek(wav_offset-1) #Seek to the wavelength section
    array = np.fromstring(label.read(296*2), dtype='>H')
    wv_array = array.astype(np.float64)
    wv_array *= 0.1

    #Radiance
    label.seek(rad_offset-1)
    array = np.fromstring(label.read(rad_lines*296*2), dtype='>H')
    rad_array = array.astype(np.float64)
    rad_array *= 0.01
    rad_array = rad_array.reshape(rad_lines,296)
    #print rad_array

    #Reflectance
    label.seek(ref_offset-1) #Seek to the wavelength section
    array = np.fromstring(label.read(ref_lines*296*2), dtype='>H')
    ref_array = array.astype(np.float64)
    ref_array *= 0.0001
    ref_array = ref_array.reshape(ref_lines,296)

    #QA
    label.seek(qa_offset-1)
    array = np.fromstring(label.read(qa_lines*296*2), dtype='>H')
    qa_array = array.astype(np.float64)
    qa_array *= 1.0  # Offset - should be dynamic?
    qa_array = qa_array.reshape(qa_lines, 296)
    #Parse the binary to get i, e, and phase for each observation
    angles = []
    for n in range(num_observations):
        #Emission Angle
        label.seek(ancillary_offset + (n*row_bytes-1) +  (emission_offset-1))
        emission_angle = unpack('>f', label.read(4))[0]
        #Incidence Angle
        label.seek(ancillary_offset + (n*row_bytes-1) + (incidence_offset-1))
        incidence_angle = unpack('>f', label.read(4))[0]
        #Phase Angle
        label.seek(ancillary_offset + (n*row_bytes-1) + (phase_offset-1))
        phase_angle = unpack('>f', label.read(4))[0]
        angles.append([incidence_angle, emission_angle,  phase_angle])
    angles = np.asarray(angles)

    return wv_array, rad_array, ref_array, angles, qa_array, ullat, lllat

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def getbandnumbers(wavelengths, *args):
    '''
    This parses the wavelenth list,finds the mean wavelength closest to the
    provided wavelength, and returns the index of that value.  One (1) is added
    to the index to grab the correct band.

    Parameters
    ----------
    wavelengths: A list of wavelengths, 0 based indexing
    *args: A variable number of input wavelengths to map to bands

    Returns
    -------
    bands: A variable length list of bands.  These are in the same order they are
    provided in.  Beware that altering the order will cause unexpected results.

    '''
    bands = []
    for x in args:
        bands.append(min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i]-x)))
    return bands

def parse_coefficients(coefficient_table):
    '''
    Parameters
    ----------

    coefficient_table     type: file path
                          The CSV file to be parsed

    Returns
    -------
    supplemental          type: list of lists
                          List of coefficients where index is the sequentially increasing wavelength.  This data is 'cleaned'.  The r_{mean} at 1003.6 is set to -999, a NoDataValue.
    '''
    d = open(coefficient_table)
    supplemental = []
    for line in d:
        line = line.split(",")
        supplemental.append([float(s) for s in line[1:]])

    return supplemental

def photometric_correction(wv, ref_vec,coefficient_table, angles):
    '''
    TODO: Docs here
    This function performs the photometric correction.
    '''
    incidence_angle = angles[:,0]
    emission_angle = angles[:,1]
    phase_angle = angles[:,2]


    def _phg(g, phase_angle):
        '''This function allows positive and neg. g to be passed in'''
        phg = (1.0-g**2) / (1.0+g**2-2.0*g*np.cos(np.radians(phase_angle))**(1.5))
        return phg

    #The ref_array runs to the detector limit, but the coefficient table truncates at 1652.1, we therefore only correct the wavelengths that we know the coefficents for.
    #Column  = ref_array[:,wv]
    b_naught = coefficient_table[wv][0]
    h = coefficient_table[wv][1]
    c = coefficient_table[wv][2]
    g = coefficient_table[wv][3]

    #Compute the phase function with fixed values
    p = ((1-c)/2) * _phg(g,30) + ((1+c)/2) * _phg((-1 * g),30)
    b = b_naught / (1+(np.tan(np.radians(30/2.0))/h))
    f_fixed = (1+b)*p

    #Compute the phase function with the observation phase
    p = (((1-c)/2) * _phg(g,phase_angle)) + (((1+c)/2)* _phg((-1 * g),phase_angle))
    b = b_naught / (1+(np.tan(np.radians(phase_angle/2.0))/h))
    f_observed = (1+b)*p

    f_ratio = f_fixed / f_observed

    #Compute the lunar lambert function
    l = 1.0 + (c1*phase_angle) + (c2*phase_angle**2) + (c3*phase_angle**3)
    cosi = np.cos(np.radians(incidence_angle))
    cose = np.cos(np.radians(emission_angle))
    xl_observed = 2 * l * (cosi / (cosi + cose)) + ((1-l)*cosi)
    xl_ratio = xl_fixed / xl_observed

    #Compute the photometrically corrected reflectance
    ref_vec = ref_vec * xl_ratio * f_ratio
    return ref_vec

def continuum_correction(bands, mask_ref, masked_wv, obs_id):
    y2 = mask_ref[obs_id][bands[1]]
    y1 = mask_ref[obs_id][bands[0]]
    wv2 = masked_wv[bands[1]]
    wv1 =masked_wv[bands[0]]

    m = (y2-y1) / (wv2 - wv1)
    b =  y1 - (m * wv1)
    y = m * masked_wv + b

    continuum_corrected_ref_array = mask_ref[obs_id] / y
    return continuum_corrected_ref_array, y

def regression_correction(wavelength, reflectance):
    m, b, r_value, p_value, stderr = ss.linregress(wavelength, reflectance)
    regressed_continuum = m * wavelength + b
    return reflectance / regressed_continuum

def horgan_correction(wavelengths, reflectance, a, b, c):
    numwv = len(wavelengths)
    maxa = reflectance[:a].argmax()
    maxb = reflectance[b:c + 1].argmax() + b
    maxc = reflectance[numwv-10:numwv-3].argmax() + numwv-10
    iterating = True
    while iterating:
        reflectance.dtype = np.float64
        x = np.asarray([wavelengths[maxa], wavelengths[maxb], wavelengths[maxc]])
        y = np.asarray([reflectance[maxa], reflectance[maxb], reflectance[maxc]])
        fit = np.polyfit(x,y,2)
        horgan_continuum = np.polyval(fit, wavelengths)
        horgan_correction = reflectance / horgan_continuum
        iterating = False
    return horgan_correction

def save_reflectance(wv_array, rad_array, ref_array, qa_array, outname):
    nobs = ref_array.shape[0]
    header = 'wavelength\tquality\t'
    for i in range(nobs):
        header += 'rad{}\tref{}\t'.format(i, i)
    ncols = nobs * 2 + 2
    stacked = np.empty((ref_array.shape[1], ncols))
    stacked[:,0] = wv_array
    #This assumes that the QS is static across all observations
    stacked[:,1] = qa_array[0]
    alt_shape = stacked[:,2::2].shape
    stacked[:,2::2] = rad_array.reshape(rad_array.size, order='F').reshape((rad_array.shape[1], rad_array.shape[0]))
    stacked[:,3::2] = ref_array.reshape(ref_array.size, order='F').reshape((ref_array.shape[1], ref_array.shape[0]))
    np.savetxt(outname + '.txt', stacked, fmt='%10.5f', header=header, delimiter='\t')
def observation_list(nrows, ncols, nobs):
    """
    Given the size of an input image and the number of observations
    evenly space said observations down the center ot the image.

    Parameters
    ----------
    nrows : int
            the number of rows
    ncols : int
            the number of columns
    nobs : int the number of observations

    Returns
    -------
    x : array
        constant array at the center of the image
    y : array
        y value for plotting observations
    pt_to_obs : dict
                label locations (id : y value)
    """
    midpoint = ncols / 2.0
    obs_interval = float(nrows) / nobs
    x = np.empty(nobs)
    x[:] = midpoint
    y = np.empty(nobs)
    y[:] = obs_interval
    y[0] = obs_interval / 2
    y = np.cumsum(y)
    labels = np.arange(nobs,dtype=np.int)
    c = 0
    pt_to_obs = {}
    for i, j, k in zip(x,y,labels):
        pt_to_obs[j] = k
    return x,y, pt_to_obs

def cleandata(qa_array, wv_array, ref_array):
    '''
    masked_wv = wv_array[np.where(qa_array[0] < 2000)[0]]
    mask_size = len(np.where(qa_array[0] < 2000)[0])
    mask_ref = np.empty((ref_array.shape[0], mask_size), dtype=np.float64)
    for i, v in enumerate(ref_array):
        mask_ref[i] = v[np.where(qa_array[i] < 2000)[0]]
    masked_wv.dtype = np.float64
    '''
    masked_wv = wv_array[np.where(qa_array[0] < 2000)[0]]
    mask_size = len(masked_wv)
    mask_ref = np.empty((ref_array.shape[0], mask_size), dtype=np.float64)
    for i, v in enumerate(ref_array):
        mask_ref[i] = v[np.where(qa_array[0] < 2000)[0]]
    return masked_wv, mask_ref

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spectral Profiler Reflectance Extraction Tool')
    parser.add_argument('input_data', action='store', help='The ".spc" file shipped with the SP data.')
    parser.add_argument('albedo_tab', action='store', help='The albedo table for the chosen overall reflectance (high, medium, or low).')
    parser.add_argument('-w', action='store',dest='wv_limits', default=1652, nargs=1, help='The limit wavelength to visualize to.')
    parser.add_argument('-s', '--save', default=False, action='store_true', dest='save', help='Save output to a CSV file.')
    parser.add_argument('-o', '--outputname', dest='outputname', help='Custom output name for the CSV file.')
    parser.add_argument('-p', default=True, action='store_false', dest='check_photometric', help='Disable photometric correction')
    parser.add_argument('observation', default=0,type=int, nargs='+', help='The range of observations to visualize.')
    args = parser.parse_args()

    #Read in the spc file, extract necessary info, and clean the data
    wv_array, rad_array, ref_array, angles, qa_array = openspc(args.input_data, args.save)
    #Using the QA data, mask the array.
    if args.save is False:
        masked_wv, mask_ref = cleandata(qa_array, wv_array, ref_array)

        maxwv = int(args.wv_limits)
        extent = np.where(masked_wv<= maxwv)
        #Copy the unphotometrically corrected array
        input_refarray = np.copy(mask_ref)


    #Parse the supplemental table to get photometric correction coefficients
    coefficient_table = parse_coefficients(args.albedo_tab)

    if args.check_photometric is True:
        #Perform the photometric correction
        for wv in range(len(coefficient_table)):
            mask_ref[:,wv] = photometric_correction(wv, mask_ref[:,wv], coefficient_table, angles)

        #Copy the photometrically corrected array
        photometrically_corrected_ref_array = np.copy(mask_ref)
        continuum_slope_array = np.empty(mask_ref.shape)

    #Continuum correction
    if args.save is True:
        if args.outputname != None:
            out = args.outputname
        else:
            out = args.input_data.split('/')[-1].split('.')[0]
        save_reflectance(wv_array, rad_array, ref_array, qa_array, out)

    else:
        #Continuum correct all observations
        for obs_id in range(len(ref_array)):
            bands = getbandnumbers(masked_wv, 752.8, 1547.7)
            mask_ref[obs_id],continuum_slope_array[obs_id] = continuum_correction(bands, mask_ref, obs_id)

        for obs in range(len(args.observation)):
            #Do the plotting
            fig = plt.figure(args.observation[obs], figsize=(8,12))
            fig.subplots_adjust(hspace=0.75)

            ax1 = subplot(411)
            grid(alpha=.5)
            plot(masked_wv[extent],input_refarray[obs][extent], linewidth=1.5)
            xlabel('Wavelength', fontsize=10)
            ax1.set_xticks(masked_wv[extent][::4])
            ax1.set_xticklabels(masked_wv[extent][::4], rotation=45, fontsize=8)
            ax1.set_xlim(masked_wv[extent].min()-10, masked_wv[extent].max()+10)
            ylabel('Reflectance', fontsize=10)
            ax1.set_yticklabels(input_refarray[obs][extent],fontsize=8)
            title('Level 2B2 Data', fontsize=12)

            ax2 = subplot(412)
            grid(alpha=.5)
            plot(masked_wv[extent],photometrically_corrected_ref_array[obs][extent], linewidth=1.5)
            xlabel('Wavelength', fontsize=10)
            ax2.set_xticks(masked_wv[extent][::4])
            ax2.set_xticklabels(masked_wv[extent][::4], rotation=45, fontsize=8)
            ax2.set_xlim(masked_wv[extent].min()-10, masked_wv[extent].max()+10)
            ylabel('Reflectance', fontsize=10)
            ax2.set_yticklabels(input_refarray[obs][extent],fontsize=8)
            title('Photometrically Corrected Data', fontsize=12)

            ax3 = subplot(413)
            grid(alpha=.5)
            plot(masked_wv[extent],photometrically_corrected_ref_array[obs][extent], label='Photometrically Corrected Spectrum', linewidth=1.5)
            plot(masked_wv[extent], continuum_slope_array[obs][extent],'r--', label='Spectral Continuum', linewidth=1.5)
            xlabel('Wavelength', fontsize=10)
            ax3.set_xticks(masked_wv[extent][::4])
            ax3.set_xticklabels(masked_wv[extent][::4], rotation=45, fontsize=8)
            ax3.set_xlim(masked_wv[extent].min()-10, masked_wv[extent].max()+10)
            ylabel('Reflectance', fontsize=10)
            ax3.set_yticklabels(input_refarray[obs][extent],fontsize=8)
            title('Continuum Slope', fontsize=12)

            ax4 = subplot(414)
            grid(alpha=.5)
            plot(masked_wv[extent], mask_ref[obs][extent], linewidth=1.5)
            xlabel('Wavelength', fontsize=10)
            ax4.set_xticks(masked_wv[extent][::4])
            ax4.set_xticklabels(masked_wv[extent][::4], rotation=45, fontsize=8)
            ax4.set_xlim(masked_wv[extent].min()-10, masked_wv[extent].max()+10)
            ylabel('Reflectance', fontsize=10)
            #ax4.set_yticklabels(mask_ref[obs][extent],fontsize=8)
            title('Continuum Removed Spectrum', fontsize=12)

            draw()

            fig2 = plt.figure(args.observation[obs] + 1, figsize=(8,8))
            grid(alpha=.5)
            plot(masked_wv[extent], mask_ref[obs][extent], linewidth=1.5)
            xlabel('Wavelength', fontsize=10)
            xticks(masked_wv[extent][::4], rotation=90)

            xlim(masked_wv[extent].min()-10, masked_wv[extent].max()+10)
            ylabel('Reflectance', fontsize=10)
            title('Continuum Removed Spectrum', fontsize=12)
    show()

