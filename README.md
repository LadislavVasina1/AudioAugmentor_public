# AudioAugmentor
### Python library for augmenting audio data
[![EXAMPLE 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1IMVLl6LCUU5gaYAz0IMSAcHh7Iq7NZr7?usp=sharing)


This library is designed to augment audio data for machine learning purposes. 
It combines several tools and libraries for audio data augmentation and provides a unified interface that can be used to apply a large set of audio augmentations in one place.

The library is designed to be used with the [PyTorch](https://pytorch.org) machine learning framework.
It can also work solely on just simple audio waveforms and augment those.

This library specifically combines these libraries and tools:

- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [audiomentations](https://github.com/iver56/audiomentations)
- [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
- [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)


### Available augmentations
Table below shows which library was used to apply specific audio augmentation/codec.

|                                                                      | [audiomentations](https://iver56.github.io/audiomentations/) | [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) | [torchaudio](https://pytorch.org/audio/stable/index.html) | [pyroomacoustics](https://pyroomacoustics.readthedocs.io/en/pypi-release/index.html) | [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) |
|----------------------------------------------------------------------|:---------------:|:---------------------:|:----------:|:---------------:|:------:|
| [AddBackgroundNoise](#addbackgroundnoise)                            |                 |           ✅          |            |                 |        |
| [AddColoredNoise](#addcolorednoise) |                 |           ✅          |            |                 |        |
| [AddGaussianNoise](#addgaussiannoise)                                |        ✅       |                       |            |                 |        |
| [AddShortNoises](#addshortnoises)                                    |        ✅       |                       |            |                 |        |
| [AdjustDuration](#adjustduration)                                    |        ✅       |                       |            |                 |        |
| [AirAbsorption](#airabsorption)                                      |        ✅       |                       |            |                 |        |
| [ApplyImpulseResponse](#applyimpulseresponse)                        |                 |           ✅          |            |                 |        |
| [BandPassFilter](#bandpassfilter)                                    |                 |           ✅          |            |                 |        |
| [BandStopFilter](#bandstopfilter)                                    |                 |           ✅          |            |                 |        |
| [ClippingDistortion](#clippingdistortion)                            |        ✅       |                       |            |                 |        |
| [FrequencyMasking](#frequencymasking)                                |                 |                       |     ✅     |                 |        |
| [Volume / Gain](#volume--gain)                                       |                 |                       |     ✅     |                 |        |
| [GainTransition](#gaintransition)                                    |        ✅       |                       |            |                 |        |
| [HighPassFilter](#highpassfilter)                                    |                 |           ✅          |            |                 |        |
| [HighShelfFilter](#highshelffilter)                                  |        ✅       |                       |            |                 |        |
| [Limiter](#limiter)                                                  |        ✅       |                       |            |                 |        |
| [LoudnessNormalization](#loudnessnormalization)                      |        ✅       |                       |            |                 |        |
| [LowPassFilter](#lowpassfilter)                                      |                 |           ✅          |            |                 |        |
| [LowShelfFilter](#lowshelffilter)                                    |        ✅       |                       |            |                 |        |
| [Mp3Compression](#mp3compression)                                    |        ✅       |                       |            |                 |        |
| [MelSpectrogram](#melspectrogram)                                    |                 |                       |     ✅     |                 |        |
| [Normalize](#normalize)                                              |        ✅       |                       |            |                 |        |
| [Padding](#padding)                                                  |        ✅       |                       |            |                 |        |
| [PeakNormalization](#peaknormalization)                              |                 |           ✅          |            |                 |        |
| [PeakingFilter](#peakingfilter)                                      |        ✅       |                       |            |                 |        |
| [PitchShift](#pitchshift)                                            |                 |                       |     ✅     |                 |        |
| [PolarityInversion](#polarityinversion)                              |                 |           ✅          |            |                 |        |
| [Time inversion](#time-inversion)                                    |                 |           ✅          |            |                 |        |
| [ApplyRIR (RoomSimulator)](#applyrir)                                |                 |                       |            |        ✅       |        |
| [SevenBandParametricEQ](#sevenbandparametriceq)                      |       ✅        |                       |            |                 |        |
| [Shift](#shift)                                                      |                 |           ✅          |            |                 |        |
| [Speed](#speed)                                                      |                 |                       |     ✅     |                 |        |
| [Spectrogram](#spectrogram)                                          |                 |                       |     ✅     |                 |        |
| [TanhDistortion](#tanhdistortion)                                    |       ✅        |                       |            |                 |        |
| [TimeMasking](#timemasking)                                          |                 |                       |     ✅     |                 |        |
| [TimeStretch](#timestretch)                                          |       ✅        |                       |            |                 |        |
| [ac3](#codecs-using-torchaudio)                                      |                 |                       |     ✅     |                 |        |
| [adpcm_ima_wav](#codecs-using-torchaudio)                            |                 |                       |     ✅     |                 |        |
| [adpcm_ms](#codecs-using-torchaudio)                                 |                 |                       |     ✅     |                 |        |
| [adpcm_yamaha](#codecs-using-torchaudio)                             |                 |                       |     ✅     |                 |        |
| [eac3](#codecs-using-torchaudio)                                     |                 |                       |     ✅     |                 |        |
| [flac](#codecs-using-torchaudio)                                     |                 |                       |     ✅     |                 |        |
| [libmp3lame](#codecs-using-torchaudio)                               |                 |                       |     ✅     |                 |        |
| [mp2](#codecs-using-torchaudio)                                      |                 |                       |     ✅     |                 |        |
| [pcm_alaw](#codecs-using-torchaudio)                                 |                 |                       |     ✅     |                 |        |
| [pcm_f32le](#codecs-using-torchaudio)                                |                 |                       |     ✅     |                 |        |
| [pcm_mulaw](#codecs-using-torchaudio)                                |                 |                       |     ✅     |                 |        |
| [pcm_s16le](#codecs-using-torchaudio)                                |                 |                       |     ✅     |                 |        |
| [pcm_s24le](#codecs-using-torchaudio)                                |                 |                       |     ✅     |                 |        |
| [pcm_s32le](#codecs-using-torchaudio)                                |                 |                       |     ✅     |                 |        |
| [pcm_u8](#codecs-using-torchaudio)                                   |                 |                       |     ✅     |                 |        |
| [wmav1](#codecs-using-torchaudio)                                    |                 |                       |     ✅     |                 |        |
| [wmav2](#codecs-using-torchaudio)                                    |                 |                       |     ✅     |                 |        |
| [g726](#g726)                                                        |                 |                       |            |                 |   ✅   |
| [gsm](#gsm)                                                          |                 |                       |            |                 |   ✅   |
| [amr](#amr)                                                          |                 |                       |            |                 |   ✅   |


## Usage
For a more complex example see [example colab notebook above](#python-library-for-augmenting-audio-data).
Or see jupyter notebook `AudioAugmentor_Usage_Example.ipynb` in the `examples` directory within this repository.

`Note: AudioAugmentor was mainly tested using Python 3.11.8 and Fedora 38 (Google Colab uses Python 3.10 and Ubuntu)`


**0. You need to install the library and necessary packages first**

**!!!You may need to run the following commands with sudo!!!**

If so install these packages manually in terminal.
```bash
pip install -U pip
pip install AudioAugmentor
dnf install -y sox                # FEDORA
dnf install -y sox-devel          # FEDORA
dnf install -y ffmpeg             # FEDORA
# apt-get install -y sox          # UBUNTU
# apt-get install -y libsox-dev   # UBUNTU
# apt-get install -y ffmpeg       # UBUNTU
```

**1. Import necessary libraries**
```python
import torch
import torchaudio
import numpy as np
import audiomentations as AA
from IPython.display import Audio, display

from AudioAugmentor import transf_gen
from AudioAugmentor import sox_parser
from AudioAugmentor import core
from AudioAugmentor import rir_setup
from AudioAugmentor import torchaudio_transf_wrapper as TTW
```
**2. Define the augmentations you want to apply to your audio data.**

You have **3** options of how to define the augmentations:

**a)** Use `transf_gen.transf_gen` function to generate list of transformations.

See [supported transformation table](#available-augmentations) and examples of every augmentation, so you know what parameters are needed for each augmentation method.

You can enter augmentation parameters as a string or as a dictionary.

`PitchShift='sample_rate=16000, n_steps=[1, 1.5, 0.1], p=1.0'`

`PitchShift={'sample_rate': 16000, 'n_steps': [1, 1.5, 0.1], 'p': 1.0}`
```php
transformations = transf_gen.transf_gen(verbose=True,
                                        PitchShift='sample_rate=16000, n_steps=[1, 1.5, 0.1], p=1.0',
                                        Speed={'orig_freq': 16000, 'factor': [0.9, 1.5, 0.1], 'p': 1},
                                        LowPassFilter={'min_cutoff_freq': 700, 'max_cutoff_freq': 800, 'sample_rate': sampling_rate, 'p': 1},
)
```
**b)** Use pseudo SoX command.
SoX command **must** be in this format:

`--sox="norm gain 0 highpass 1000 phaser 0.5 0.6 1 0.45 0.6 -s"`

(When you don't want to apply some codec after applying SoX effects)

OR

`--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k`

(In this case, you want to apply codec after applying SoX effects -> Codec is entered in the form `codec_name` `codec_parameter_name` `codec_parameter_value` directly after the SoX effects command)
```python
example_sox = '--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k'
```

**c)** Use a file with multiple pseudo SoX commands. Random SoX command from this file will be chosen and applied to your data.

File **must** to be loaded using `sox_parser.load_sox_file` function. 
```php
sox_file_content_to_write = '''--sox="norm gain 0 highpass 1000 phaser 0.5 0.6 1 0.45 0.6 -s"
#--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s"
--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" gsm
--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k
'''
with open('sox_file_example.txt', 'w') as f:
    f.write(sox_file_content_to_write)

sox_file_content = sox_parser.load_sox_file('sox_file_example.txt')
print('SOX FILE LOADED:', sox_file_content, type(sox_file_content))
```



**3. Apply augmentations**

**a)** Use generated the `transformations` list, `single SoX command` or `loaded SoX file content` while initializing `Collator` class. 

Use this initiated class as an argument for the `collate_fn` parameter of PyTorch's dataloader.
```php
collate_fn = core.Collator(
    transformations=transformations, device='cpu', sox_effects=None, sample_rate=sampling_rate, verbose=True,
    #transformations=None, device='cpu', sox_effects='--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k', sample_rate=sampling_rate, verbose=False,
    #transformations=None, device='cpu', sox_effects=sox_file_content, sample_rate=sampling_rate, verbose=False,
)

dataset = torchaudio.datasets.LIBRISPEECH("../data", url="train-clean-100", download=True)
aug_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    num_workers=0,
    collate_fn=collate_fn,
)
augmented_record_from_dataset = next(iter(aug_dataloader))
display(Audio(augmented_record_from_dataset[0].squeeze(0).squeeze(0).squeeze(0).cpu(), rate=sampling_rate))
```
`OR`

**b)** Use generated the `transformations` list, `single SoX command` or `loaded SoX file content` while initializing `AugmentWaveform` class and apply the augmentations to the audio signal.
```php
augment = core.AugmentWaveform(
    transformations=transformations, device='cpu', sox_effects=None, sample_rate=16000, verbose=False,
    #transformations=None, device='cpu', sox_effects='--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k', sample_rate=16000, verbose=False,
    #transformations=None, device='cpu', sox_effects=sox_file_content, sample_rate=16000, verbose=False,
)
# Load test wav file
signal, fs = torchaudio.load('../data/test.wav')
# Apply transformations
waveform = augment(signal.numpy()[0])
display(Audio(waveform, rate=fs))
```

**c)** Use generated the `transformations` list, `single SoX command` or `loaded SoX file content` while initializing `AugmentLocalAudioDataset` class and apply the augmentations to the local audio dataset.
```php
augment = core.AugmentLocalAudioDataset(
    transformations=transformations, device='cpu', sox_effects=None, sample_rate=16000, verbose=False,
    #transformations=None, device='cpu', sox_effects='--sox="norm gain 20 highpass 300 phaser 0.5 0.6 1 0.45 0.6 -s" amr audio_bitrate 4.75k', sample_rate=16000, verbose=False,
    #transformations=None, device='cpu', sox_effects=sox_file_content, sample_rate=16000, verbose=False,
)
augment(input_dir='../data/test-input-folder', output_dir='../data/test-output-folder')
```


# EXAMPLES OF AVAILABLE AUGMENTATIONS
## !!!Put following examples as an argument for `transf_gen.transf_gen` function to generate a list of transformations!!!

Like this:
```php
transformations = transf_gen.transf_gen(verbose=True,
                                        AddBackgroundNoise=f'background_paths="../data/musan/noise/free-sound", min_snr_in_db=10, max_snr_in_db=20, p=1, sample_rate={sampling_rate}',
                                        AddColoredNoise=f'min_snr_in_db=9, max_snr_in_db=10, p=1, sample_rate={sampling_rate}',
                                        )
```
You can enter augmentation parameters as a string or as a dictionary.

`PitchShift='sample_rate=16000, n_steps=[1, 1.5, 0.1], p=1.0'`

`PitchShift={'sample_rate': 16000, 'n_steps': [1, 1.5, 0.1], 'p': 1.0}`

<a id="addbackgroundnoise"></a>
### [⬆️](#available-augmentations) AddBackgroundNoise [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/background_noise.py#L32-L51)
```python
AddBackgroundNoise=f'''background_paths="../data/musan/noise/free-sound",
                       min_snr_in_db=10, 
                       max_snr_in_db=20, 
                       p=1, 
                       sample_rate={sampling_rate}''',
```
<a id="addcolorednoise"></a>
### [⬆️](#available-augmentations) AddColoredNoise [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/colored_noise.py#L49-L74)
```python
AddColoredNoise=f'''min_snr_in_db=9,
                    max_snr_in_db=10, 
                    p=1, 
                    sample_rate={sampling_rate}''',
```
<a id="addgaussiannoise"></a>
### [⬆️](#available-augmentations) AddGaussianNoise [docs](https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_noise/)
```python
AddGaussianNoise={'min_amplitude': 0.001, 
                  'max_amplitude': 0.015, 
                  'p': 1},
```
<a id="addshortnoises"></a>
### [⬆️](#available-augmentations) AddShortNoises [docs](https://iver56.github.io/audiomentations/waveform_transforms/add_short_noises/)
```python
AddShortNoises={'sounds_path': "../data/musan/noise/free-sound",
                'min_snr_in_db': 3.0,
                'max_snr_in_db': 30.0,
                'noise_rms': "relative_to_whole_input",
                'min_time_between_sounds': 2.0,
                'max_time_between_sounds': 8.0,
                'noise_transform': AA.PolarityInversion(),
                'p': 1.0},
```
<a id="adjustduration"></a>
### [⬆️](#available-augmentations) AdjustDuration [docs](https://iver56.github.io/audiomentations/waveform_transforms/adjust_duration/)
```python
AdjustDuration={'duration_seconds': 3.5, 
                'padding_mode': 'silence', 
                'p': 1},
```
<a id="airabsorption"></a>
### [⬆️](#available-augmentations) AirAbsorption [docs](https://iver56.github.io/audiomentations/waveform_transforms/air_absorption/)
```python
AirAbsorption={'min_distance': 10.0, 
               'max_distance': 50.0, 
               'min_humidity': 80.0, 
               'max_humidity': 90.0, 
               'min_temperature': 10.0, 
               'max_temperature': 20.0, 
               'p': 1.0},
```
<a id="applyimpulseresponse"></a>
### [⬆️](#available-augmentations) ApplyImpulseResponse [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/impulse_response.py#L33-L55)
```python
ApplyImpulseResponse=f'''ir_paths="../data/Rir.wav", 
                         p=1, 
                         sample_rate={sampling_rate}''',
```
<a id="bandpassfilter"></a>
### [⬆️](#available-augmentations) BandPassFilter [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/band_pass_filter.py#L25-L46)
```python
BandPassFilter=f'''min_center_frequency=200, 
                   max_center_frequency=4000, 
                   min_bandwidth_fraction=0.5, 
                   max_bandwidth_fraction=1.99, 
                   sample_rate={sampling_rate}, 
                   p=1''',
```
<a id="bandstopfilter"></a>
### [⬆️](#available-augmentations) BandStopFilter [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/band_stop_filter.py#L16-L38)
```python
BandStopFilter=f'''min_center_frequency=200, 
                   max_center_frequency=4000, 
                   min_bandwidth_fraction=0.5, 
                   max_bandwidth_fraction=1.99, 
                   sample_rate={sampling_rate}, 
                   p=1''',
```
<a id="clippingdistortion"></a>
### [⬆️](#available-augmentations) ClippingDistortion [docs](https://iver56.github.io/audiomentations/waveform_transforms/clipping_distortion/)
```python
ClippingDistortion={'min_percentile_threshold': 10, 
                    'max_percentile_threshold': 30, 
                    'p': 1},
```
<a id="frequencymasking"></a>
### [⬆️](#available-augmentations) FrequencyMasking [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.FrequencyMasking.html)
```python
FrequencyMasking={'freq_mask_param': 80},
```
<a id="volume--gain"></a>
### [⬆️](#available-augmentations) Volume / Gain [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.Vol.html)
```python
Vol={'gain': [2.5, 3, 0.1], 
     'p': 1.0},
```
<a id="gaintransition"></a>
### [⬆️](#available-augmentations) GainTransition [docs](https://iver56.github.io/audiomentations/waveform_transforms/gain_transition/)
```python
GainTransition={'min_gain_db': 30, 
                'max_gain_db': 40, 
                'min_duration': 5, 
                'max_duration': 16, 
                'duration_unit': 'seconds', 
                'p': 1},
```
<a id="highpassfilter"></a>
### [⬆️](#available-augmentations) HighPassFilter [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/high_pass_filter.py#L15-L31)
```python
HighPassFilter=f'''min_cutoff_freq=700,
                   max_cutoff_freq=800,
                   sample_rate={sampling_rate},
                   p=1''',
```
<a id="highshelffilter"></a>
### [⬆️](#available-augmentations) HighShelfFilter [docs](https://iver56.github.io/audiomentations/waveform_transforms/high_shelf_filter/)
```python
HighShelfFilter={'min_center_freq': 2000, 
                 'max_center_freq': 5000, 
                 'min_gain_db': 10.0, 
                 'max_gain_db': 16.0, 
                 'min_q': 0.5, 
                 'max_q': 1.0, 
                 'p': 1},
```
<a id="limiter"></a>
### [⬆️](#available-augmentations) Limiter [docs](https://iver56.github.io/audiomentations/waveform_transforms/limiter/)
```python
Limiter='''min_threshold_db=-24, 
           max_threshold_db=-2,
           min_attack=0.0005, 
           max_attack=0.025, 
           min_release=0.05, 
           max_release=0.7, 
           threshold_mode="relative_to_signal_peak", 
           p=1''',
```
<a id="loudnessnormalization"></a>
### [⬆️](#available-augmentations) LoudnessNormalization [docs](https://iver56.github.io/audiomentations/waveform_transforms/loudness_normalization/)
```python
LoudnessNormalization={'min_lufs': -31, 
                       'max_lufs': -13, 
                       'p': 1},
```
<a id="lowpassfilter"></a>
### [⬆️](#available-augmentations) LowPassFilter [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/low_pass_filter.py#L27-L42)
```python
LowPassFilter={'min_cutoff_freq': 700, 
               'max_cutoff_freq': 800, 
               'sample_rate': sampling_rate, 
               'p': 1},
```
<a id="lowshelffilter"></a>
### [⬆️](#available-augmentations) LowShelfFilter [docs](https://iver56.github.io/audiomentations/waveform_transforms/low_shelf_filter/)
```python
LowShelfFilter={'min_center_freq': 20, 
                'max_center_freq': 600, 
                'min_gain_db': -16.0, 
                'max_gain_db': 16.0, 
                'min_q': 0.5, 
                'max_q': 1.0, 
                'p': 1},
```
<a id="mp3compression"></a>
### [⬆️](#available-augmentations) Mp3Compression [docs](https://iver56.github.io/audiomentations/waveform_transforms/mp3_compression/)
```python
Mp3Compression={'min_bitrate': 8, 
                'max_bitrate': 8, 
                'backend': 'pydub', 
                'p': 1},
```
<a id="melspectrogram"></a>
### [⬆️](#available-augmentations) MelSpectrogram [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.MelSpectrogram.html)
```python
MelSpectrogram={'sample_rate': 16000},
```
<a id="normalize"></a>
### [⬆️](#available-augmentations) Normalize [docs](https://iver56.github.io/audiomentations/waveform_transforms/normalize/)
```python
Normalize={'p': 1},
```
<a id="padding"></a>
### [⬆️](#available-augmentations) Padding [docs](https://iver56.github.io/audiomentations/waveform_transforms/padding/)
```python
Padding={'mode': 'silence', 
         'min_fraction': 0.02, 
         'max_fraction': 0.8, 
         'pad_section': 'start', 
         'p': 1},
```
<a id="peaknormalization"></a>
### [⬆️](#available-augmentations) PeakNormalization [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/peak_normalization.py#L34-L36)
```python
PeakNormalization={'p': 1, 
                   'sample_rate': sampling_rate},
```
<a id="peakingfilter"></a>
### [⬆️](#available-augmentations) PeakingFilter [docs](https://iver56.github.io/audiomentations/waveform_transforms/peaking_filter/)
```python
PeakingFilter={'min_center_freq': 51, 
               'max_center_freq': 7400, 
               'min_gain_db': -22, 
               'max_gain_db': 22, 
               'min_q': 0.5, 
               'max_q': 1.0, 
               'p': 1},
```
<a id="pitchshift"></a>
### [⬆️](#available-augmentations) PitchShift [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.PitchShift.html)
```python
PitchShift={'sample_rate': 16000, 
            'n_steps': [1, 1.5, 0.1],
            'bins_per_octave': 12, 
            'n_fft': 512, 
            'win_length':512, 
            'hop_length': 512//4, 
            'p': 1.0},
```
<a id="polarityinversion"></a>
### [⬆️](#available-augmentations) PolarityInversion [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/polarity_inversion.py#L30-L32)
```python
PolarityInversion={'p': 1, 
                   'sample_rate': sampling_rate},
```
<a id="time-inversion"></a>
### [⬆️](#available-augmentations) Time inversion [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/time_inversion.py#L28-L39)
```python
TimeInversion={'p': 1, 
               'sample_rate': sampling_rate},
```
<a id="applyrir"></a>
### [⬆️](#available-augmentations) ApplyRIR
```php
# Use this to see available materials you can use as walls_mat, floor_mat and ceiling_mat argument
# from AudioAugmentor import rir_setup
# rir_setup.get_all_materials_info()

# This way you set up parameters when you want to generate random room parameter
rir_kwargs = {
    'audio_sample_rate': 16000,
    'x_range': (0, 100), 
    'y_range': (0, 100), 
    'num_vertices_range': (3, 6),
    'mic_height': 1.5,
    'source_height': 1.5,
    'walls_mat': 'curtains_cotton_0.5',
    'room_height': 2.0,
    'max_order': 3,
    'floor_mat': 'carpet_cotton',
    'ceiling_mat': 'hard_surface',
    'ray_tracing': True,
    'air_absorption': True,
}
# This way you set up parameters when you want to generate specific room
rir_kwargs = {
    'audio_sample_rate': 16000,
    'corners_coord': [[0, 0], [0, 3], [5, 3], [5, 1], [3, 1], [3, 0]],
    'walls_mat': 'curtains_cotton_0.5',
    'room_height': 2.0,
    'max_order': 3,
    'floor_mat': 'carpet_cotton',
    'ceiling_mat': 'hard_surface',
    'ray_tracing': True,
    'air_absorption': True,
    'source_coord': [[1.0], [1.0], [0.5]],
    'microphones_coord': [[3.5], [2.0], [0.5]],
}
transformations = transf_gen.transf_gen(verbose=True,
                                        ApplyRIR=rir_kwargs,
                                        )
```
<a id="sevenbandparametriceq"></a>
### [⬆️](#available-augmentations) SevenBandParametricEQ [docs](https://iver56.github.io/audiomentations/waveform_transforms/seven_band_parametric_eq/)
```python
SevenBandParametricEQ={'min_gain_db': -10, 
                       'max_gain_db': 10, 
                       'p': 1},
```
<a id="shift"></a>
### [⬆️](#available-augmentations) Shift [docs](https://github.com/asteroid-team/torch-audiomentations/blob/9baf5c516a44651025bd7e8d8ead35888b58bbdc/torch_audiomentations/augmentations/shift.py#L66-L93)
```python
Shift={'min_shift': 1, 
       'max_shift': 2, 
       'p': 1, 
       'sample_rate': sampling_rate},
```
<a id="speed"></a>
### [⬆️](#available-augmentations) Speed [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.Speed.html)
```python
Speed={'orig_freq': 16000, 
       'factor': [0.9, 1.5, 0.1], 
       'p': 1},
```
<a id="spectrogram"></a>
### [⬆️](#available-augmentations) Spectrogram [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.Spectrogram.html)
```python
Spectrogram={'sample_rate': 16000},
```
<a id="tanhdistortion"></a>
### [⬆️](#available-augmentations) TanhDistortion [docs](https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/)
```python
TanhDistortion={'min_distortion': 0.1, 
                'max_distortion': 0.8, 
                'p': 1},
```
<a id="timemasking"></a>
### [⬆️](#available-augmentations) TimeMasking [docs](https://pytorch.org/audio/main/generated/torchaudio.transforms.TimeMasking.html)
```python
TimeMasking={'time_mask_param': 80},
```
<a id="timestretch"></a>
### [⬆️](#available-augmentations) TimeStretch [docs](https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/)
```python
TimeStretch='''min_rate=0.9, 
               max_rate=1.1, 
               p=0.2, 
               leave_length_unchanged=False''',
```
<a id="codecs-using-torchaudio"></a>
### [⬆️](#available-augmentations) Codecs using torchaudio
You can select just one. No need to use them all. :)
```php
transformations = transf_gen.transf_gen(verbose=True,
                                        ac3=True,
                                        adpcm_ima_wav=True,
                                        adpcm_ms=True,
                                        adpcm_yamaha=True,
                                        eac3=True,
                                        flac=True,
                                        libmp3lame=True,
                                        mp2=True,
                                        pcm_alaw=True,
                                        pcm_f32le=True,
                                        pcm_mulaw=True,
                                        pcm_s16le=True,
                                        pcm_s24le=True,
                                        pcm_s32le=True,
                                        pcm_u8=True,
                                        wmav1=True,
                                        wmav2=True,
                                        )
```
<a id="g726"></a>
### [⬆️](#available-augmentations) g726
```python
g726={'audio_bitrate': '40k'},
```
<a id="gsm"></a>
### [⬆️](#available-augmentations) gsm
```python
gsm=True,
```
<a id="amr"></a>
### [⬆️](#available-augmentations) amr
```python
amr={'audio_bitrate': '4.75k'},
```
