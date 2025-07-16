import numpy as np
from scipy.signal import hilbert
import pyloudnorm as pyln

def normalize_output(data, reference_loudness, meter):

    output_loudness = meter.integrated_loudness(data)
    normal_output = pyln.normalize.loudness(data, output_loudness, reference_loudness)

    return normal_output


def get_envelope(data):

    return np.abs(hilbert(data))

def peak_envelope(data, frame_size=128):

    n_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        n_frames += 1       

    envelope = np.zeros_like(data)

    for t in range(n_frames):
        start = t * frame_size
        end = start + frame_size
        if end > len(data):
            end = len(data)
        envelope[start:end] = np.max(abs(data[start : end]))

    return envelope

def get_peak_hold(data, frame_size=128):

    n_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        n_frames += 1       

    envelope = np.zeros_like(data)

    state = 0.0

    for t in range(n_frames):
        start = t * frame_size
        end = start + frame_size
        block_len = frame_size
        if end > len(data):
            end = len(data)
            block_len = end - start
        frame_max = np.max(abs(data[start : end]))
        envelope[start:end] = np.linspace(state, frame_max, block_len)
        state = frame_max

    return envelope

def get_smooth_env(data, frame_size=128, window='hann'):

    # TODO: Add support for other window types
    if window == 'hann':
        wind = np.hanning(frame_size)
    else:
        raise ValueError("Currently only 'hann' window is supported.")

    overlap = 2
    advance = int(frame_size / overlap)

    n_frames = int(len(data) // advance)
    
    if len(data) % frame_size != 0:
        n_frames += 1       

    envelope = np.zeros_like(data)

    state = 0.0

    for t in range(n_frames):
        start = t * advance
        end = start + frame_size
        block_len = frame_size
        if end > len(data):
            end = len(data)
            block_len = end - start
            wind = np.hanning(block_len)
        try:
            frame_max = np.max(abs(data[start:end]))
            output_block = np.linspace(state, frame_max, block_len) * wind
            envelope[start:end] += output_block
            state = frame_max
        except ValueError:
            print('Value Error occured when calculating envelope windowing function')
            break

    return envelope

def get_half_hann_env(data, frame_size=128):

    n_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        n_frames += 1       

    curve = half_hann(frame_size)
    envelope = np.zeros_like(data)

    state = 0.0
    block_len = frame_size

    for t in range(n_frames):
        start = t * frame_size
        end = start + frame_size
        if end > len(data):
            end = len(data)
            block_len = end - start
            curve = half_hann(block_len)
        frame_max = np.max(abs(data[start : end]))
        frame_diff = frame_max - state
        smoothed = (curve * frame_diff) + state
        envelope[start:end] = smoothed
        state = frame_max

    return envelope

def get_sigmoid_env(data, frame_size=128):

    n_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        n_frames += 1       

    curve = sigmoid_smooth(frame_size)
    envelope = np.zeros_like(data)

    state = 0.0
    block_len = frame_size

    for t in range(n_frames):
        start = t * frame_size
        end = start + frame_size
        if end > len(data):
            end = len(data)
            block_len = end - start
            curve = sigmoid_smooth(block_len)
        frame_max = np.max(abs(data[start : end]))
        frame_diff = frame_max - state
        smoothed = (curve * frame_diff) + state
        envelope[start:end] = smoothed
        state = frame_max

    return envelope

def sigmoid_smooth(N):
    block = np.linspace(-6, 6, N)
    w = 1 / (1 + np.exp(-block))
    return w

def half_hann(N):

    double = N * 2

    window = np.zeros(double)

    for i in range(double):
        window[i] = 0.5 - 0.5 * np.cos(2 * np.pi * i / (double-1))

    half_window = window[:N]

    return half_window

def MIN_MAX(x, min, max):

    if x > max:
        y = max
    elif x < min:
        y = min
    else:
        y = x

    return y

def ms_to_alpha(time, fs):

    time_in_secs = time / 1000

    alpha = np.power(0.01, 1/(time_in_secs*fs*0.001))

    return MIN_MAX(alpha, 0.0, 1.0)

def block_max(block):

    block_len = len(block)
    env = 0.0

    for i in block:
        if i > env:
            env = i

    return env

def get_block_env(data, fs, attack_ms, release_ms, frame_size=256, block_size=64):

    curve = sigmoid_smooth(block_size)

    attack_alpha = ms_to_alpha(attack_ms, fs)
    release_alpha = ms_to_alpha(release_ms, fs)

    n_frames = len(data) // frame_size
    if len(data) % frame_size != 0:
        n_frames += 1       

    abs_data = abs(hilbert(data))
    envelope = np.zeros_like(data)

    state = 0.0

    frame_len = frame_size

    for t in range(n_frames):

        frame_start = t * frame_size
        frame_end = frame_start + frame_size

        if frame_end > len(data):
            frame_end = len(data)
            frame_len = frame_end - frame_start

        frame = abs_data[frame_start:frame_end]

        n_blocks = int(frame_len / block_size)
        output_frame = np.zeros(frame_size)

        if len(frame) % block_size != 0:
            n_blocks += 1       

        for i in range(n_blocks):
        
            block_start = block_size * i
            block_len = block_size
            block_end = block_start + block_size

            if block_end > len(frame):
                block_end = len(frame)
                block_len = block_end - block_start

            block = frame[block_start:block_end]

            env = block_max(block)
            output_block = np.zeros_like(block)

            if env > state:
                alpha = attack_alpha
            else:
                alpha = release_alpha

            env = alpha * (state - env) + env

            delta = env - state

            output_frame[block_start:block_end] = curve * delta + state

            state = env
            
        envelope[frame_start:frame_end] = output_frame

    return envelope