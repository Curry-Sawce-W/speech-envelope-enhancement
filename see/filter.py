import numpy as np
from scipy.signal import lfilter
import see.envelope as envelope

def Hz_to_ERB(f):
    """
    Convert frequency in Hz to ERB number
    """
    return 21.4 * np.log10(1 + f * 0.00437)

def ERB_to_Hz(erb_num):
    """
    Convert ERB number to frequency in Hz
    """

    return (10**(erb_num / 21.4) - 1) * 1000 / 4.37

def ERB_cf(f_low, f_high, n):

    """
    Calculate n centre frequencies linearly spaced along ERB scale within range (f_low, f_high)

    Parameters:
    f_low   (float):    lower bound of frequency range (Hz)
    f_high  (float):    upper bound of frequency range (Hz)
    n       (int):      number of centre frequencies to calculate

    Returns:
    cf      (list):     list of n centre frequencies linearly spaced on ERB scale in given range
    """

    erb_low = Hz_to_ERB(f_low)
    erb_high = Hz_to_ERB(f_high)
    erb_cf = np.linspace(erb_low, erb_high, n)
    cf = [ERB_to_Hz(erb) for erb in erb_cf]

    return cf

def ERB_x_over(f_low, f_high, n):

    """
    Calculate n crossover frequencies linearly spaced along ERB scale within range (f_low, f_high)

    Parameters:
    f_low   (float):    lower bound of frequency range (Hz)
    f_high  (float):    upper bound of frequency range (Hz)
    n       (int):      number of crossover frequencies to calculate

    Returns:
    cf      (list):     list of n crossover frequencies linearly spaced on ERB scale in given range
    """

    erb_low = Hz_to_ERB(f_low)
    erb_high = Hz_to_ERB(f_high)
    erb_x = np.linspace(erb_low, erb_high, n + 1)
    x_over = [ERB_to_Hz(erb) for erb in erb_x]

    return x_over

def ERB_cf_to_x_over(cf):
    """
    Calculate crossover frequencies based on ERB scale.

    Parameters:
    - cf: List of center frequencies in Hz, spaced on the ERB scale

    Returns:
    - List of crossover frequencies in Hz
    """
    erb_numbers = [Hz_to_ERB(f) for f in cf]
    erb_crossovers = [(erb_numbers[i] + erb_numbers[i+1]) / 2 for i in range(len(erb_numbers) - 1)]
    crossover_freqs = [ERB_to_Hz(erb) for erb in erb_crossovers]
    return crossover_freqs


def biquad(filter_type, cutoff, fs, Q=0.707):

    """
    Calculates coefficients for a biquad lowpass, highpass, allpass or bandpass filter using standard formulas.

    Parameters:
    - filter_type (str): 'lowpass', 'highpass', 'allpass' or 'bandpass'
    - cutoff (float): Cutoff frequency in Hz
    - fs (float): Sampling rate in Hz
    - Q (float): Quality factor (default is 1/sqrt(2) for Butterworth-like response)

    Returns:
    - b (np.ndarray): Numerator coefficients [b0, b1, b2]
    - a (np.ndarray): Denominator coefficients [1, a1, a2]
    """

    if filter_type not in ['lowpass', 'highpass', 'allpass', 'bandpass']:
        raise ValueError("filter_type must be 'lowpass', 'highpass', 'allpass' or 'bandpass'")

    omega = 2.0 * np.pi * cutoff / fs
    cos_omega = np.cos(omega)
    alpha = np.sin(omega) / (2.0 * Q)

    a0 = 1 + alpha

    a1 = -2 * cos_omega / a0
    a2 = (1 - alpha) / a0

    if filter_type == 'lowpass':
        b0 = (1 - cos_omega) / 2 / a0
        b1 = (1 - cos_omega) / a0
        b2 = (1 - cos_omega) / 2 / a0

    elif filter_type == 'highpass':
        b0 = (1 + cos_omega) / 2 / a0
        b1 = -(1 + cos_omega) / a0
        b2 = (1 + cos_omega) / 2 / a0
    
    elif filter_type == 'allpass':
        b0 = (1 - alpha) / a0
        b1 = -2 * cos_omega / a0
        b2 = (1 + alpha) / a0

    elif filter_type == 'bandpass':
        b0 = alpha
        b1 = 0
        b2 = -alpha
    
    b = np.array([b0, b1, b2])
    a = np.array([1, a1, a2])
    
    return b, a

def bandpass_low_high(f_low, f_high, fs):

    f0 = np.sqrt(f_low * f_high)
    BW = f_high - f_low
    Q = f0 / BW

    return biquad('bandpass', f0, fs, Q), f0
    
def BPF_data(data, f_low, f_high, fs):

    """
    Apply a bandpass filter with biquad by specifying high and low cutoff rather than centre frequency.

    Parameters:
    data    (arraylike):    signal to be filtered
    f_low   (float):        Low cutoff for band to be passed
    f_high  (float):        High cutoff for band to be passed
    fs      (int):          Samplerate

    returns:
    y       (arraylike):    Array containing the bandpass filtered signal.
    """

    b, a = bandpass_low_high(f_low, f_high, fs)

    y = lfilter(b,a, data)

    return y

def apply_filter(data, b, a):
    return lfilter(b, a, data)

def filter_data(data, filter_type, cutoff, fs, Q=0.707):

    b, a = biquad(filter_type, cutoff, fs, Q)

    """
    ###############################################################################
    # Original sample by sample filtering function. 
    # Commented out to replace with more efficient C implementation from scipy.
    # Left in place incase useful for later adaption to real-time implementation 
    y = np.zeros_like(data)
    x1 = x2 = y1 = y2 = 0.0

    for n in range(len(data)):
        x0 = data[n]
        y[n] = b[0]*x0 + b[1]*x1 + b[2]*x2 - a[1]*y1 - a[2]*y2
        x2, x1 = x1, x0
        y2, y1 = y1, y[n]
 
    """

    y = lfilter(b, a, data)

    return y

def LWR_filter_design(filter_type, cutoff, fs):

    """
    Calculates coefficients for a 4th-order Linkwitz-Riley filter using two cascading 2nd-order Butterworth filters.

    Parameters:
    - filter_type (str): 'lowpass' or 'highpass'
    - cutoff (float): Cutoff frequency in Hz
    - fs (float): Sampling rate in Hz

    Returns:
    - np.ndarray: stage1 and stage2 coefficients
    """

    Q = 1 / np.sqrt(2)

    stages=[]

    for _ in range(2):
        stages.append(biquad(filter_type, cutoff, fs, Q))

    return stages



def LWR_crossover_coefs(x_over_freqs, fs):

    """
    Calculates coefs for 4th-order Linkwitz-Riley filter to be appliedy cascading two 2nd-order Butterworth filters.

    Parameters:
    - data (np.ndarray): Input signal
    - filter_type (str): 'lowpass' or 'highpass'
    - cutoff (float): Cutoff frequency in Hz
    - fs (float): Sampling rate in Hz

    Returns:
    - np.ndarray: Filtered signal
    """

    Q =  1 / np.sqrt(2)

    LP_coefs = [biquad('lowpass', cutoff, fs, Q) for cutoff in x_over_freqs]
    AP_coefs = [biquad('allpass', cutoff, fs, Q) for cutoff in x_over_freqs]

    return LP_coefs, AP_coefs

def invert_list(data):

    y = []
    for i, x in enumerate(data):

        if np.sign(x)==1:
            y.append(-x)
        elif np.sign(x)==-1:
            y.append(abs(x))
        else:
            y.append(x)
    
    return y


def subtract_list(list_pos, list_neg):

    """
    Perform element-wise subtraction on two lists

    parameters:
    - list_plus (list): positive list
    - list_minus (list): list of values to be subtracted from list_plus

    returns:
    - list_result (list): list of values resulting from calculation
    """

    list_result = []

    inverted = invert_list(list_neg)

    for i, j in zip(list_pos, inverted):

        result = i + j

        list_result.append(result)

    return list_result


def subtract_array(pos_array, neg_array):

    """
    Perform element-wise subtraction on two lists

    parameters:
    - pos_array (np.ndarray): positive array
    - list_minus (np.ndarray): array of values to be subtracted from pos_array

    returns:
    - list_result (list): list of values resulting from calculation
    """

    inverted = neg_array * -1.0

    y = pos_array + inverted

    return y


def filter_data_LWR(data, filter_type, cutoff, fs):
    
    """
    Applies a 4th-order Linkwitz-Riley filter by cascading two 2nd-order Butterworth filters.

    Parameters:
    - data (np.ndarray): Input signal
    - filter_type (str): 'lowpass' or 'highpass'
    - cutoff (float): Cutoff frequency in Hz
    - fs (float): Sampling rate in Hz

    Returns:
    - np.ndarray: Filtered signal
    """

    Q = 1 / np.sqrt(2)

    b, a = biquad(filter_type, cutoff, fs, Q)

    for _ in range(2):

        data = lfilter(b, a, data)

    return data


def reconstruct_signal(Y_channels):

    Y_summed = np.zeros_like(Y_channels[0])

    for chan in Y_channels:
        Y_summed += chan

    Y = np.stack(Y_summed, axis=-1) if len(Y_summed) > 1 else Y_summed[0]

    #output = np.int16(np.clip(Y, -1.0, 1.0) * 32767)

    return Y

def apply_LWR_crossover(data, LP_coefs, AP_coefs):

    if len(data.shape)==1:
        X = [data]
    else:
        X = [data[:,x] for x in range(data.shape[1])]

    high_data = X

    channels = []

    for lp_coef, ap_coef in zip(LP_coefs, AP_coefs):

        lp_b = lp_coef[0]
        lp_a = lp_coef[1]

        ap_b = ap_coef[0]
        ap_a = ap_coef[1]

        #low_data = lfilter(lp_b, lp_a, lfilter(lp_b, lp_a, high_data))
        #high_data = lfilter(ap_b, ap_a, lfilter(ap_b, ap_a, high_data))

        low_data = [lfilter(lp_b, lp_a, x) for x in high_data]
        #low_data = [lfilter(lp_b, lp_a, x) for x in low_data]

        high_data = [lfilter(ap_b, ap_a, x) for x in high_data]
        #high_data = [lfilter(ap_b, ap_a, x) for x in high_data]

        channels = [[lfilter(ap_b, ap_a, x) for x in chan] for chan in channels]
        #channels = [lfilter(ap_b, ap_a, x) for x in channels]

        channels.append(low_data)

    channels.append(high_data)

    # Reverse channel order and iterate through from highest freq bin to lowest freq bin and HPF each channel by subtracting data from next channel down. 
    rev_chans = list(reversed(channels))

    Y = []


    for i, _ in enumerate(rev_chans[:-1]):

        y = rev_chans[i][0] + (rev_chans[i+1][0]*-1)

        Y.append(y)
    """

    for i in range(len(channels[:-1])):

        ii = i+1

        y = channels[-i][0] + (channels[-ii][0] * -1)

        Y.append(y)

    #Y.append(channels[0][0])
    """
    Y = list(reversed(Y))

    return np.array(Y)

def LWR_filterbank(data, x_over_freqs, fs):

    # Calculate coefficients for Linkwitz Riley filters for allpass cross over filterbank
    LP_coefs, AP_coefs = LWR_crossover_coefs(x_over_freqs, fs)

    # Pass data through filterbank to return array of shape (n_frequency_bands, n_channels, n_samples)
    channels = apply_LWR_crossover(data, LP_coefs, AP_coefs)

    return channels

def get_mod_spectrum(data, x_over_freqs, fs, n=None):

    filtered = LWR_filterbank(data, x_over_freqs, fs)

    n_bands, n_samps = filtered.shape

    if n == None:
        n = n_samps

    mod_spectrum = np.zeros((n_bands, n))

    for i, band in enumerate(filtered):
        mod_spectrum[i] = np.abs(np.fft.fft(envelope.get_envelope(band), n))

    ERB_x_overs = [Hz_to_ERB(x) for x in x_over_freqs]

    ERB_cf = np.zeros(n_bands)

    for i, x in enumerate(x_over_freqs[:-1]):

        cf = (x_over_freqs[i+1]-x)/2+x
        ERB_cf[i] = cf

    cf = np.array([ERB_to_Hz(erb) for erb in ERB_cf])
    
    mod_freqs = np.fft.fftfreq(n, d=1/fs)

    freqs = (cf, mod_freqs)

    return mod_spectrum, freqs