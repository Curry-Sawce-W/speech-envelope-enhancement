import numpy as np
import see.filter as filter
import see.envelope as envelope

def IOEC(x):

    if x <= -30.0:
        y = x
    elif x <= -25.0:
        y = 4 * x + 90
    elif x <= -15.0:
        y = (3 / 4) * x + 8.75
    elif x <= -10.0:
        y = 0.5 * x + 5
    else:
        y = 0.0

    return y

def apply_IOEC(data):

    return np.array([IOEC(x) for x in data])

def apply_DRC(data, fs, slow_env=160.0, fast_env=750.0, attack=0.0001, release=0.15):

    env = envelope.get_envelope(data)

    fast_env = filter.filter_data(env, 'lowpass', 750.0, fs, Q=0.707)
    slow_env = filter.filter_data(env, 'lowpass', 160.0, fs, Q=0.707)

    e_hat = 0
    
    static_comp = np.zeros_like(env)

    for i in range(len(env)):

        e = slow_env[i]

        if e < e_hat:
            e_hat = release * e_hat + (1 - release) * e
        else:
            e_hat = attack * e_hat + (1 - attack) * e

        static_comp[i] = e_hat
    
    e0 = 0.75 * np.max(fast_env)

    e_in = 20*np.log(10)*(static_comp / e0)

    e_out = apply_IOEC(e_in)

    gain = 10**((e_out - e_in) / 20)

    Y = data * gain

    return Y


def DRC(data, fs, f_low, f_high, n_bands=22, process_low=False, process_high=False):

    """
    Apply multi-band dynamic range compression (DRC) to an audio signal.

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

    cf = filter.ERB_cf(f_low, f_high, n=n_bands)

    x_over_freqs = filter.ERB_x_over(cf)

    bands = filter.LWR_filterbank(data, x_over_freqs, fs)

    DRC_bands = []

    slow_env = np.linspace(80, 375, len(cf))
    fast_env = np.linspace(375, 750, len(cf))

    for i, band in enumerate(bands):

        if i == 0 and not process_low:
            y = filter.filter_data(band, 'allpass', slow_env[i], fs, Q=0.707)

        elif i == (len(bands)-1) and not process_high:
            y = filter.filter_data(band, 'allpass', slow_env[i], fs, Q=0.707)

        else:
            y = apply_DRC(band, fs, slow_env[i], fast_env, attack=0.0001, release=0.15)

        DRC_bands.append(y)
    
    enhanced = filter.reconstruct_signal(DRC_bands)

    return enhanced
