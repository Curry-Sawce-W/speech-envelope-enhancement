import numpy as np
import see.envelope as envelope
import see.filter as filter

def log_limit(array):

    log_array = 2*np.log(1+(array/2))

    return log_array

def apply_OE(data, fs, cutoff=20.0, slow_boost=1.5, peak='additive', peak_boost=1.0, frame_size=128, env_type='smooth', attack_ms=5.0, release_ms=5.0):

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

    epsilon = 1e-6  # small value to avoid division by zero

    if env_type == 'peak':
        env = envelope.get_peak_hold(data, frame_size)
    elif env_type == 'smooth':
        env = envelope.get_smooth_env(data, frame_size)
    elif env_type == 'hilbert':
        env = envelope.get_envelope(data)
    elif env_type == 'block':
        env = envelope.get_block_env(data, fs, attack_ms, release_ms, frame_size, block_size=64)
    else:
        raise ValueError("env_type must be 'peak', 'smooth', or 'hilbert'")

    """
    env_fast = filter.filter_data(data=env, 
                           filter_type='lowpass', 
                           cutoff=fast_cutoff, 
                           fs=fs, 
                           Q=0.707
    )
    """
    
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
        env_peak = -np.clip(env_peak, 0.0, None)

        env_enhanced = env_peak + env

        gain = env_enhanced / (env  + epsilon) # to avoid division by zero

    elif peak == 'additive':

        env_peak = (env - env_slow_boost) * peak_boost
        env_peak = np.clip(env_peak, 0.0, None)
        gain = env_peak + 1

    enhanced = data * gain

    return np.array(enhanced)


def OE(data, fs, f_low, f_high, cutoff=20.0, slow_boost=1.5, peak='additive', peak_boost=4.0, n_bands=22, frame_size=128, env_type='smooth', attack_ms=5.0, release_ms=5.0, process_low=False, process_high=False):

    """
    Apply multi-band onset enhancement (OE) to an audio signal.

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
    cutoff: float
        frequency of LPF used to create control signal. Onsets faster than this frequency will be boosted.
    slow_boost: float
        amplification factor used to boost low frequency envelope to be at higher level during stationary parts. Greater boost = reduced sensitivity to onsets.
    peak_boost: float
        amplification factor used to boost onsets by.
    n_bands : int, optional
        Number of frequency bands to split the signal into (default: 10).
    process_low : bool, optional
        If True, apply DRC to the lowest frequency band. If False, leave it unprocessed (default: False).
    process_high : bool, optional
        If True, apply DRC to the highest frequency band. If False, leave it unprocessed (default: False).

    Returns:
    --------
    enhanced : np.ndarray
        Output audio signal after multi-band DRC, same shape as input.
    """

    x_over_freqs = filter.ERB_x_over(f_low, f_high, n_bands)

    bands = filter.LWR_filterbank(data, x_over_freqs, fs)

    OE_bands = np.zeros_like(bands)

    for i, band in enumerate(bands):

        if i == 0:
            OE_bands[i] = band

        elif i == len(bands)-1:
            OE_bands[i] = band

        else:
            OE_bands[i] = apply_OE(band, fs, cutoff, slow_boost, peak, peak_boost, frame_size, env_type)

        #OE_bands.append(y)
    
    enhanced = filter.reconstruct_signal(OE_bands)

    return enhanced
