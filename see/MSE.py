import numpy as np
import see.filter as filter
import math
import see.envelope as envelope

def intensity(x):
    
    return  10 * np.log10(x**2 + 1e-6)


def enhancement_ceiling(cf, beta):

    bark = 7 * math.asinh(cf/650)

    return 1 + (10**(beta/20)-1) * (0.5 - 0.5 * np.cos((np.pi*bark)/13))

def modulation_filter_window(f_low, f_high, freqs):
    
    alpha = np.log(2)

    Hf = np.exp(-((alpha*freqs)/f_high)**2) - np.exp(-((alpha*freqs)/f_low)**2)

    return Hf


def deepen_band_mod(data, fs, lo_cut, hi_cut, gamma, alpha=0.5):
    """
    Boosts the amplitude of frequencies in a specified range in the input signal.

    Parameters:
    - data: Input time-domain signal (1D numpy array)
    - fs: Sampling frequency of the signal
    - f_low: Lower bound of the frequency range to boost
    - f_high: Upper bound of the frequency range to boost
    - gamma: Amplitude boost/enhancement factor

    Returns:
    - output_signal: Time-domain signal after frequency boost
    """
    N = len(data)

    env = intensity(data)
    fft_output = np.fft.fft(env)
    freqs = np.fft.fftfreq(N, d=1/fs)

    # Create window
    window = modulation_filter_window(lo_cut, hi_cut, freqs)
    
    filtered = fft_output * window

    # Convert filtered signal back to time domain and convert to power
    real_filtered = np.real(np.fft.ifft(filtered))
    power = 10**((real_filtered) / 2)

    enhance_factor = np.sqrt(1 / (1/power + 1/gamma))

    enhanced_env = env*enhance_factor

    gain = (alpha*enhanced_env + (1-alpha)*env) / env
    output = data * gain

    return output

def apply_MSE(data, fs, cf, lo_cut, hi_cut, beta, alpha):

    gamma = enhancement_ceiling(cf, beta)
    enhanced = deepen_band_mod(data, fs, lo_cut, hi_cut, gamma, alpha)
    
    return enhanced


def MSE(data, fs, f_low, f_high, lo_cut=3.0, hi_cut=30.0, beta=20.0, alpha=0.75, n_bands=22, process_low=False, process_high=False):

    """
    Apply multi-band modulation spectrum enhancement (MSE) to an audio signal.

    Parameters:
    -----------
    data : np.ndarray
        Input audio signal (mono or stereo, shape: [samples] or [samples, channels]).
    fs : float
        Sampling frequency of the input signal in Hz.
    f_low : float
        Lower frequency bound for the filterbank (Hz).
    f_high : float
        Upper frequency bound for the filterbank (Hz).
    lo_cut : float
        Lower frequency for modulation boosting (Hz).
    hi_cut : float
        Upper frequency for modulation boosting (Hz).
    beta : float
        dB enhancement value
    n_bands : int, optional
        Number of frequency bands to split the signal into (default: 10).
    process_low : bool, optional
        If True, apply DRC to the lowest frequency band. If False, leave it unprocessed (default: False).
    process_high : bool, optional
        If True, apply DRC to the highest frequency band. If False, leave it unprocessed (default: False).

    Returns:
    --------
    enhanced : np.ndarray
        Output audio signal after multi-band MSE, same shape as input.
    """

    cf = filter.ERB_cf(f_low, f_high, n=n_bands)
    x_over_freqs = filter.ERB_x_over(f_low, f_high, n=n_bands)

    bands = filter.LWR_filterbank(data, x_over_freqs, fs)

    MSE_bands = []

    for i, band in enumerate(bands):

        if i == 0 and not process_low:
            y = band

        elif i == len(bands)-1 and not process_high:
            y = band

        else:
            y = apply_MSE(band, fs, cf[i], lo_cut, hi_cut, beta, alpha)
            

        MSE_bands.append(y)
    
    enhanced = filter.reconstruct_signal(MSE_bands)

    return enhanced

