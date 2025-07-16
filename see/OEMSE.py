
import numpy as np
import see.MSE as MSE
import see.envelope as envelope
import see.filter as filter


def apply_OEMSE(data, fs, cf, cutoff=20.0, slow_boost=1.5, peak='ratio', peak_boost=1.0, frame_size=128, lo_cut=3.0, hi_cut=20.0, beta=20.0, env_type='smooth'):

    """
    Onset enhancer based on algorithm described by Koning & Wouters (2012). 
    Original used an FFT based method but this function applies the idea to the output of a filterbank.

    Parameters:
    data (arraylike):   list or array of speech data to be enhanced
    fs (int):           Sample Rate
    slow_freq (float):  Cutoff frequency used to filter slow envelope (lower cutoff produces more boosted segments and earlier boosting of onsets)
    slow_boost (float): Gain boost applied to slow envelope to ensure it has greater amplitude than fast_freq during quasi-stationary sections. (Greater boost reduces sensitivity to onsets and the relative gain applied in enhancement)
    frame_size (int):   Size of the frame used to calculate the peak hold envelope. (Larger frame size produces smoother envelope but may miss some onsets)
    env_type (str):     Type of envelope to use, either 'peak' or 'hilbert'. 'peak' uses a peak hold envelope, 'hilbert' uses the Hilbert transform to calculate the envelope.
    
    Returns:
    enhanced (array):   Enhanced Speech signal with boosted acoustic onsets.

    """

    gamma = MSE.enhancement_ceiling(cf, beta)
    mse = MSE.deepen_band_mod(data, fs, lo_cut, hi_cut, gamma)

    epsilon = 1e-6  # small value to avoid division by zero

    if env_type == 'peak':
        env = envelope.get_peak_hold(data, frame_size)
    elif env_type == 'smooth':
        env = envelope.get_smooth_env(data, frame_size)
    elif env_type == 'hilbert':
        env = envelope.get_envelope(data)
    else:
        raise ValueError("env_type must be 'peak', 'smooth', or 'hilbert'")
    
    if env_type == 'peak':
        mse_env = envelope.get_peak_hold(mse, frame_size)
    elif env_type == 'smooth':
        mse_env = envelope.get_smooth_env(mse, frame_size)
    elif env_type == 'hilbert':
        mse_env = envelope.get_envelope(mse)
    else:
        raise ValueError("env_type must be 'peak', 'smooth', or 'hilbert'")

    env_slow = filter.filter_data(data=env, 
                           filter_type='lowpass', 
                           cutoff=cutoff, 
                           fs=fs, 
                           Q=0.707
    )

    env_slow_boost = env_slow * slow_boost

    gain = np.zeros_like(env)

    if peak == 'ratio':

        env_peak = (env - env_slow_boost) * peak_boost
        env_peak = np.clip(env_peak, 0.0, None)

        env_enhanced = env_peak + mse_env
        gain = env_enhanced / (env  + epsilon) # to avoid division by zero

    enhanced = data * gain

    return np.array(enhanced)

def apply_MSEOE(data, fs, cf, cutoff=20.0, slow_boost=1.5, peak='ratio', peak_boost=1.0, frame_size=128, lo_cut=3.0, hi_cut=20.0, beta=20.0, env_type='smooth'):

    """
    Onset enhancer based on algorithm described by Koning & Wouters (2012). 
    Original used an FFT based method but this function applies the idea to the output of a filterbank.

    Parameters:
    data (arraylike):   list or array of speech data to be enhanced
    fs (int):           Sample Rate
    slow_freq (float):  Cutoff frequency used to filter slow envelope (lower cutoff produces more boosted segments and earlier boosting of onsets)
    slow_boost (float): Gain boost applied to slow envelope to ensure it has greater amplitude than fast_freq during quasi-stationary sections. (Greater boost reduces sensitivity to onsets and the relative gain applied in enhancement)
    frame_size (int):   Size of the frame used to calculate the peak hold envelope. (Larger frame size produces smoother envelope but may miss some onsets)
    env_type (str):     Type of envelope to use, either 'peak' or 'hilbert'. 'peak' uses a peak hold envelope, 'hilbert' uses the Hilbert transform to calculate the envelope.
    
    Returns:
    enhanced (array):   Enhanced Speech signal with boosted acoustic onsets.

    """

    gamma = MSE.enhancement_ceiling(cf, beta)
    mse = MSE.deepen_band_mod(data, fs, lo_cut, hi_cut, gamma)

    epsilon = 1e-6  # small value to avoid division by zero

    if env_type == 'peak':
        env = envelope.get_peak_hold(data, frame_size)
    elif env_type == 'smooth':
        env = envelope.get_smooth_env(data, frame_size)
    elif env_type == 'hilbert':
        env = envelope.get_envelope(data)
    else:
        raise ValueError("env_type must be 'peak', 'smooth', or 'hilbert'")
    
    if env_type == 'peak':
        mse_env = envelope.get_peak_hold(mse, frame_size)
    elif env_type == 'smooth':
        mse_env = envelope.get_smooth_env(mse, frame_size)
    elif env_type == 'hilbert':
        mse_env = envelope.get_envelope(mse)
    else:
        raise ValueError("env_type must be 'peak', 'smooth', or 'hilbert'")

    env_slow = filter.filter_data(data=env, 
                           filter_type='lowpass', 
                           cutoff=cutoff, 
                           fs=fs, 
                           Q=0.707
    )

    env_slow_boost = env_slow * slow_boost

    gain = np.zeros_like(env)

    if peak == 'ratio':

        env_peak = (env - env_slow_boost) * peak_boost
        env_peak = np.clip(env_peak, 0.0, None)

        env_enhanced = env_peak + mse_env
        gain = env_enhanced / (env  + epsilon) # to avoid division by zero

    enhanced = data * gain

    return np.array(enhanced)

def OEMSE(data, fs, f_low, f_high, cutoff=20.0, slow_boost=1.5, peak='ratio', peak_boost=1.0, frame_size=128, lo_cut=3.0, hi_cut=30.0, beta=20.0, env_type='smooth', n_bands=22, process_low=False, process_high=False):

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

    OEMSE_bands = []

    for i, band in enumerate(bands):

        if i == 0 and not process_low:
            y = band

        elif i == len(bands)-1 and not process_high:
            y = band

        else:
            y = apply_OEMSE(data, fs, cf[i], cutoff, slow_boost, peak, peak_boost, frame_size, lo_cut, hi_cut, beta, env_type)

        OEMSE_bands.append(y)
    
    enhanced = filter.reconstruct_signal(OEMSE_bands)

    return enhanced


def MSEOE(data, fs, f_low, f_high, cutoff=20.0, slow_boost=1.5, peak='ratio', peak_boost=1.0, frame_size=128, lo_cut=3.0, hi_cut=30.0, beta=20.0, env_type='smooth', n_bands=22, process_low=False, process_high=False):

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

    OEMSE_bands = []

    for i, band in enumerate(bands):

        if i == 0 and not process_low:
            y = band

        elif i == len(bands)-1 and not process_high:
            y = band

        else:
            y = apply_MSEOE(data, fs, cf[i], cutoff, slow_boost, peak, peak_boost, frame_size, lo_cut, hi_cut, beta, env_type)

        OEMSE_bands.append(y)
    
    enhanced = filter.reconstruct_signal(OEMSE_bands)

    return enhanced