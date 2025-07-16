from pathlib import Path

import numpy as np

import soundfile as sf
import librosa
import pyloudnorm as pyln

from IPython.display import display, Audio
import ipywidgets as widgets

ROOT = Path.cwd()
WAV_PATH = ROOT / 'audio'
SPEECH_PATH = WAV_PATH / 'speech'
NOISE_PATH = WAV_PATH / 'noise'


def mix_and_display(SNR, processing, sample, noise):

    speech_LUFS = -23
    noise_LUFS = speech_LUFS - SNR

    algos = {
    'MSE-A':'Modulation Spectrum Enhanced A',
    'MSE-B':'Modulation Spectrum Enhanced B', 
    'OE-A':'Onset Enhanced A', 
    'OE-B':'Onset Enhanced B', 
    'OEMSE':'Onset and Modulation Spectrum Enhanced'
    }

    UP_path = SPEECH_PATH / f'{sample}_UP.wav'
    UP_speech, fs = sf.read(UP_path)

    SEE_speech_path = SPEECH_PATH / f'{sample}_{processing}.wav'
    SEE_speech, fs = sf.read(SEE_speech_path)

    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(SEE_speech)
    
    SEE_speech = pyln.normalize.loudness(SEE_speech, loudness, speech_LUFS)
    UP_speech = pyln.normalize.loudness(UP_speech, loudness, speech_LUFS)
    

    if noise==None:
        noise = np.zeros_like(UP_speech)
        fs_noise = fs
    else:
        noise_path = NOISE_PATH / f'{noise}.wav'
        noise, fs_noise = librosa.load(noise_path, sr=fs)
        noise_loudness = meter.integrated_loudness(noise)
        noise = pyln.normalize.loudness(noise, noise_loudness, noise_LUFS)

        try:
            noise = noise.sum(axis=1) / 2
        except:
            noise = noise

        # Ensure both signals have the same sample rate
        if fs != fs_noise:
            raise ValueError("Sample rates of speech and noise must match.")
        

        
        # Trim or pad noise to match speech length
        if len(noise) < len(SEE_speech):
            repeats = int(np.ceil(len(SEE_speech) / len(noise)))
            noise = np.tile(noise, (repeats, 1) if noise.ndim > 1 else repeats)
        noise = noise[:len(SEE_speech)]

    # Mix speech and noise
    SEE_mixed = SEE_speech + noise
    UP_mixed = UP_speech + noise

    # Create a Label widget
    UP_label = widgets.Label(value="Unprocessed:")
    SEE_label = widgets.Label(value=f"{algos[processing]}:")

    display(UP_label, Audio(SEE_mixed, rate=fs))
    display(SEE_label, Audio(UP_mixed, rate=fs))


def mix_to_snr(speech_path, noise_path, snr, output_path, speech_LUFS=-23.0):
    
    # Load audio files
    speech, fs = sf.read(speech_path)
    noise, fs_noise = librosa.load(noise_path, sr=fs)

    try:
        noise = noise.sum(axis=1) / 2
    except:
        noise = noise

    # Ensure both signals have the same sample rate
    if fs != fs_noise:
        raise ValueError("Sample rates of speech and noise must match.")
    
    meter = pyln.Meter(fs)
    loudness = meter.integrated_loudness(speech)
    noise_loudness = meter.integrated_loudness(noise)
    noise_LUFS = speech_LUFS - snr

    speech = pyln.normalize.loudness(speech, loudness, speech_LUFS)
    noise = pyln.normalize.loudness(noise, noise_loudness, noise_LUFS)
    
    # Trim or pad noise to match speech length
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, (repeats, 1) if noise.ndim > 1 else repeats)
    noise = noise[:len(speech)]

    # Mix and save
    mixed = speech + noise
    sf.write(output_path, mixed, fs)
    print(f"Mixed audio saved to {output_path}")