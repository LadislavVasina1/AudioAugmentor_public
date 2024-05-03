"""
File containing functions that generate transformations from the user input.

@author: Ladislav Vašina, github: LadislavVasina1
"""

import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.io as TIO
import os
from IPython.display import Audio, display
import torch_audiomentations as TA
import audiomentations as AA
import soundfile as sf
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import torch.multiprocessing as mp

# Build pip package with this uncommented
from . import torchaudio_transf_wrapper as TTW
from .rir_setup import ApplyRIR, RandomRIRKwargs, get_all_materials_info, create_random_rir_kwargs

# Test sources with this uncommented
# import torchaudio_transf_wrapper as TTW
# from rir_setup import ApplyRIR, RandomRIRKwargs, get_all_materials_info, create_random_rir_kwargs


PRA_TRANSF = [
    'ApplyRIR'
]

TTW_TRANSF = [
    'pitchshift',
    'speed',
    'vol',
    'frequencymasking',
    'timemasking',
    'spectrogram',
    'melspectrogram',
]

# MelSpectrogram, Spectrogram, TimeMasking, FrequencyMasking augmentations
# should be used as last augmentation method
T_TRANSF = [
    'PitchShift',               # https://pytorch.org/audio/main/generated/torchaudio.transforms.PitchShift.html
    'Speed',                    # https://pytorch.org/audio/main/generated/torchaudio.transforms.Speed.html
    'Vol',                      # https://pytorch.org/audio/main/generated/torchaudio.transforms.Vol.html               
    'FrequencyMasking',         # https://pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html
    'TimeMasking',              # https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html
    'Spectrogram',              # https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html
    'MelSpectrogram',           # https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html
]

TA_TRANSF = [                       
    'AddColoredNoise',          # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/colored_noise.py#L49-L74
    'AddBackgroundNoise',       # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/background_noise.py#L32-L51
    'BandPassFilter',           # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/band_pass_filter.py#L25-L46
    'BandStopFilter',           # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/band_stop_filter.py#L16-L38
    'HighPassFilter',           # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/high_pass_filter.py#L15-L31
    'ApplyImpulseResponse',     # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/impulse_response.py#L33-L55
    'LowPassFilter',            # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/low_pass_filter.py#L27-L42   
    'PeakNormalization',        # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/peak_normalization.py#L34-L36
    'PolarityInversion',        # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/polarity_inversion.py#L30-L32
    'Shift',                    # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/shift.py#L66-L93
    'TimeInversion',            # https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/time_inversion.py#L28-L39
]

AA_TRANSF = [
    'AddGaussianNoise',         # https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_noise/
    'AddShortNoises',           # https://iver56.github.io/audiomentations/waveform_transforms/add_short_noises/
    'AdjustDuration',           # https://iver56.github.io/audiomentations/waveform_transforms/adjust_duration/
    'AirAbsorption',            # https://iver56.github.io/audiomentations/waveform_transforms/air_absorption/
    'ClippingDistortion',       # https://iver56.github.io/audiomentations/waveform_transforms/clipping_distortion/
    'Gain',                     # https://iver56.github.io/audiomentations/waveform_transforms/gain/
    'GainTransition',           # https://iver56.github.io/audiomentations/waveform_transforms/gain_transition/
    'HighShelfFilter',          # https://iver56.github.io/audiomentations/waveform_transforms/high_shelf_filter/
    'Limiter',                  # https://iver56.github.io/audiomentations/waveform_transforms/limiter/
    'LoudnessNormalization',    # https://iver56.github.io/audiomentations/waveform_transforms/loudness_normalization/
    'LowShelfFilter',           # https://iver56.github.io/audiomentations/waveform_transforms/low_shelf_filter/
    'Mp3Compression',           # https://iver56.github.io/audiomentations/waveform_transforms/mp3_compression/
    'Normalize',                # https://iver56.github.io/audiomentations/waveform_transforms/normalize/
    'Padding',                  # https://iver56.github.io/audiomentations/waveform_transforms/padding/
    'PeakingFilter',            # https://iver56.github.io/audiomentations/waveform_transforms/peaking_filter/
    'SevenBandParametricEQ',    # https://iver56.github.io/audiomentations/waveform_transforms/seven_band_parametric_eq/
    'TanhDistortion',           # https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/
    'TimeStretch',              # https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/
]

# Codecs should be used as last augmentation method.
# Only exception is using SpecAugment, in that case SpecAugment should be used as last method.
CODECS = [
    "ac3",              # torchaudio
    "adpcm_ima_wav",    # ---| |----
    "adpcm_ms",         # ---| |----
    "adpcm_yamaha",     # ---| |----
    "eac3",             # ---| |----
    "flac",             # ---| |----
    "libmp3lame",       # ---| |----
    "mp2",              # ---| |----
    "pcm_alaw",         # ---| |----
    "pcm_f32le",        # ---| |----
    "pcm_f64le",        # ---| |----
    "pcm_mulaw",        # ---| |----
    "pcm_s16le",        # ---| |----
    "pcm_s24le",        # ---| |----
    "pcm_s32le",        # ---| |----
    "pcm_u8",           # ---| |----
    "wmav1",            # ---| |----
    "wmav2",            # torchaudio
    'g726',             # ffmpeg - supported bitrates 16k, 24k, 32k, 40k
    'gsm',              # ffmpeg - GSM-FR = 13k bitrate
    'amr',              # ffmpeg - supported bitrates 4.75k, 5.15k, 5.90k, 6.70k, 7.40k, 7.95k, 10.20k, 12.20k
]

CODECS_AFFECTORS_LIST = []

for codec in CODECS:
    globals()[f"{codec}_effector"] = TIO.AudioEffector(
        format="wav", encoder=codec)
    CODECS_AFFECTORS_LIST.append(globals()[f"{codec}_effector"])


def verbose_out(key, value, transf_list):
    print(f'ADDED: {get_transf(key, transf_list)}, \n\t\t{value}\n')


def get_transf(user_input, list_of_transformations):
    '''
    Args:
        user_input: user input
        list_of_transformations: list of transformations
    Returns:
        specific transformation from the list of available transformations listed above
        or None if the transformation is not in the list

    This function takes user input and returns the name of the transformation.
    It can handle user input in any case (for better UX) and returns the name of the transformation.
    '''

    user_input_lower = user_input.lower()
    for item in list_of_transformations:
        if item.lower() == user_input_lower:
            return item
    return None


def transf_gen(verbose=False, **kwargs):
    '''
    Args:
        kwargs: dictionary containing transformation with its arguments

    Returns: transfoarmations chain (list)
    '''

    transformations = []

    for key, value in kwargs.items():
        # Handle string input of the transformation parameters
        if type(value) is str:
            value = eval('dict(' + value + ')')

        # Check the type of the current transformation from the user
        pra_flag = get_transf(key, PRA_TRANSF) in PRA_TRANSF
        t_flag = get_transf(key, T_TRANSF) in T_TRANSF
        ta_flag = get_transf(key, TA_TRANSF) in TA_TRANSF
        aa_flag = get_transf(key, AA_TRANSF) in AA_TRANSF
        codec_flag = get_transf(key, CODECS) in CODECS

        # Handle list input of the transformation parameters
        match key:
            # Handle transformations from PYROOMACOUSTICS
            case key if pra_flag:
                random_rir_kwargs_unique_keys = ['x_range', 'y_range', 'num_vertices_range', 'mic_height']
                if any(key in value for key in random_rir_kwargs_unique_keys):
                    rir_kwargs = RandomRIRKwargs(**value)
                else:
                    rir_kwargs = value
                transformations.append(
                    eval(get_transf(key, PRA_TRANSF))(rir_kwargs)
                )
                verbose_out(key, value, PRA_TRANSF) if verbose else None

            # Handle transformations from TORCH_AUDIOMENTATIONS
            case key if ta_flag:
                transformations.append(
                    getattr(TA, get_transf(key, TA_TRANSF))(**value)
                )
                verbose_out(key, value, TA_TRANSF) if verbose else None

            # Handle transformations from AUDIOMENTATIONS
            case key if aa_flag:
                transformations.append(
                    getattr(AA, get_transf(key, AA_TRANSF))(**value)
                )
                verbose_out(key, value, AA_TRANSF) if verbose else None

            # Handle transformations from TORCHAUDIO
            case key if t_flag:
                # check if values contain "p" key
                if 'p' in value.keys():
                    # save the p value and delete it from value dictionary
                    p = value['p']
                    del value['p']
                    transformations.append(
                        {get_transf(key, TTW_TRANSF): {**value}, 'p': p}
                    )

                elif 'p' not in value.keys():
                    transformations.append(
                        getattr(T, get_transf(key, T_TRANSF))(**value)
                    )

                verbose_out(key, value, T_TRANSF) if verbose else None

            # Handle codecs
            case key if codec_flag:
                if key != 'g726' and key != 'gsm' and key != 'amr':
                    transformations.append(
                        CODECS_AFFECTORS_LIST[CODECS.index(key)]
                    )
                else:
                    if key == 'g726':
                        # Check if the bitrate is correct
                        correct_g726_bitrates = ['16k', '24k', '32k', '40k']
                        if value['audio_bitrate'] not in correct_g726_bitrates:
                            raise ValueError(
                                f'''!!!INVALID BITRATE FOR G726: {value["audio_bitrate"]}!!!\n
                                Bitrate must be one of: 16k, 24k, 32k, 40k'''
                            )
                    elif key == 'amr':
                        # Check if the bitrate is correct
                        correct_amr_bitrates = [
                            '4.75k', '5.15k', '5.90k', '6.70k', '7.40k', '7.95k', '10.20k', '12.20k']
                        if value['audio_bitrate'] not in correct_amr_bitrates:
                            raise ValueError(
                                f'''!!!INVALID BITRATE FOR AMR: {value["audio_bitrate"]}!!!\n
                                Bitrate must be one of: 4.75k, 5.15k, 5.90k, 6.70k, 7.40k, 7.95k, 10.20k, 12.20k'''
                            )

                    transformations.append((key, value))

                verbose_out(key, value, CODECS) if verbose else None

            # Handle unknown transformation
            case _:
                raise ValueError(f'!!!UNKNOWN TRANSFORMATION: {key} !!!')
    # Verbose out
    print(
        f'\n\nFINAL TRANSFORMATIONS LIST:\n' +
        '\n'.join(str(transformation) for transformation in transformations)
    ) if verbose else None

    return transformations
