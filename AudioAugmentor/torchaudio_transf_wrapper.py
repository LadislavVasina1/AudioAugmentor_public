"""
File containing functions that adds randmization to the torchaudio transforms.

@author: Ladislav Vašina, github: LadislavVasina1
"""

import torch
import numpy as np
import torchaudio.transforms as T
from random import choice


def randomize_parameters(**kwargs):
    '''
    Get parameters in form [min, max, steps] and return randomized values
    Args: 
        **kwargs: Arguments of the augmentations with paramaters in form [min, max, steps]
    Returns: 
        randomized_values (dict): Dictionary of randomized parameters.
    '''

    randomized_values = {}
    for param, values in kwargs.items():
        if isinstance(values, list):
            param_values = np.arange(values[0], values[1], values[2])
            randomized_value = choice(param_values)
            randomized_values[param] = randomized_value
        else:
            randomized_values[param] = values

    return randomized_values


def vol(
        gain,
        gain_type='amplitude'
):
    '''
    Wrapper for torchaudio.transforms.Vol
    https://pytorch.org/audio/main/generated/torchaudio.transforms.Vol.html
    
    Args:
        gain (list): Gain in form [min, max, steps] - RANDOMIZABLE
        gain_type (str): Type of gain - One of: amplitude, power, db (Default: amplitude)

    Returns:
        torchaudio.transforms.Vol - Volume transfomation from the torchaudio library with randomized parameters.
    '''

    randomized_parameters = randomize_parameters(gain=gain)
    return T.Vol(**randomized_parameters, gain_type=gain_type)


def speed(
        orig_freq,
        factor
):
    '''
    Wrapper for torchaudio.transforms.Speed
    https://pytorch.org/audio/main/generated/torchaudio.transforms.Speed.html

    Args:
        orig_freq (int): Original frequency of the audio
        factor (list): Factor of the speed change in form [min, max, steps] - RANDOMIZABLE

    Returns:
        torchaudio.transforms.Speed - Speed transfomation from the torchaudio library with randomized parameters.
    '''

    randomized_factor = randomize_parameters(factor=factor)
    return T.Speed(orig_freq, **randomized_factor)


def pitchshift(
        sample_rate,
        n_steps,
        bins_per_octave=12,
        n_fft=512,
        win_length=None,
        hop_length=None,
        window_fn=None):
    '''
    Wrapper for torchaudio.transforms.PitchShift
    https://pytorch.org/audio/main/generated/torchaudio.transforms.PitchShift.html

    Args:
        sample_rate (int): Sample rate of waveform.
        n_steps (int): The (fractional) steps to shift waveform.
        bins_per_octave (int, optional): The number of steps per octave (Default : 12).
        n_fft (int, optional): Size of FFT, creates n_fft // 2 + 1 bins (Default: 512).
        win_length (int or None, optional): Window size. If None, then n_fft is used. (Default: None).
        hop_length (int or None, optional): Length of hop between STFT windows. 
            If None, then win_length // 4 is used (Default: None).
        window_fn (Tensor or None, optional): Window tensor that is applied/multiplied to each frame/window. 
            If None, then torch.hann_window(win_length) is used (Default: None)

    Returns:
        torchaudio.transforms.PitchShift - Pitch shift transfomation from the torchaudio library with randmized parameters
    '''
    parameters = {
        'n_steps': n_steps,
        'bins_per_octave': bins_per_octave,
        'n_fft': n_fft,
        'win_length': win_length,
        'hop_length': hop_length,
        'window_fn': window_fn
    }

    if window_fn is None:
        parameters.pop('window_fn')

    randomized_parameters = randomize_parameters(**parameters)
    
    return T.PitchShift(sample_rate, **randomized_parameters)
