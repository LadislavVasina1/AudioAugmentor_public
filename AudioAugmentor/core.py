"""
File containing Classes that are used to apply transfomation on data
or as a collate function for the DataLoader.
Handles propoerly applying transformations and sox effects to the data.

@author: Ladislav Vašina, github: LadislavVasina1
"""

import os
import tempfile
import inspect
import ffmpeg
import numpy as np
import torch
import torchaudio
import torchaudio.io as TIO
import audiomentations as AA
import torch_audiomentations as TA
import torchaudio.transforms as T
import shutil
from pathlib import Path
from glob import glob
from scipy.io.wavfile import read, write
from datasets.formatting.formatting import LazyRow
import random

# Build pip package with this uncommented
# from . import torchaudio_transf_wrapper as TTW
# from .transf_gen import T_TRANSF, TTW_TRANSF, get_transf
# from .sox_parser import select_random_sox, parse
# from . import rir_setup

# Test sources with this uncommented
import torchaudio_transf_wrapper as TTW
from transf_gen import T_TRANSF, TTW_TRANSF, get_transf
from sox_parser import select_random_sox, parse
import rir_setup


def torch_randomizer(
        transformation,
        p
):
    """
    This function decides whether to apply the transformation or not based on the probability p.
    Args:
        transformation: Transformation which user wants to decide wheter to use.
        p (float): Float between 0.0 and 1.0 stating the probability

    Returns:
        transformation or None based on probability
    """

    if transformation is None:
        raise ValueError('Transformation must be specified!')
    if (p < 0) or (p > 1):
        raise ValueError('Probability p must be in range [0, 1]!')

    if random.random() < p:
        return transformation
    else:
        return None


def apply_g726(tmp_input_path, tmp_output_path, audio_bitrate):
    """
    This function applies g726 codec using ffmpeg-python library to the input file and saves it to the output file.
    Args:
        tmp_input_path (str): Path to the input file.
        tmp_output_path (str): Path to the output file.
        audio_bitrate (int): Audio bitrate of the output file.

    Returns:
        None
    """

    (
        ffmpeg
        .input(tmp_input_path)
        .output(
            tmp_output_path,
            acodec='g726',
            audio_bitrate=audio_bitrate,
            ar=8000,
            loglevel="quiet",
        )
        .run(overwrite_output=True)
    )


def apply_gsm(tmp_input_path, tmp_output_path):
    """
    This function applies gsm codec using ffmpeg-python library to the input file and saves it to the output file.
    Args:
        tmp_input_path (str): Path to the input file.
        tmp_output_path (str): Path to the output file.

    Returns:
        None
    """

    (
        ffmpeg
        .input(tmp_input_path)
        .output(
            tmp_output_path,
            ar=8000,
            acodec='gsm',
            loglevel="quiet",
        )
        .run(overwrite_output=True)
    )


def apply_amr(tmp_input_path, tmp_output_path, audio_bitrate):
    """
    This function applies amr codec using ffmpeg-python library to the input file and saves it to the output file.
    Args:
        tmp_input_path (str): Path to the input file.
        tmp_output_path (str): Path to the output file.
        audio_bitrate (int): Audio bitrate of the output file.

    Returns:
        None
    """

    (
        ffmpeg
        .input(tmp_input_path)
        .output(
            tmp_output_path,
            ar=8000,
            audio_bitrate=audio_bitrate,
            format='amr',
            loglevel="quiet",
        )
        .run(overwrite_output=True)
    )


def sox_codec_handler(sox_effects_to_apply, data, sample_rate):
    """
    This function applies codecs using sox library to the input file and returns the output file.
    Args:
        sox_effects_to_apply (list): List of lists, where inner lists contain sox effects with their parameters.
        data (torch.tensor): Input data to which the user wants to apply the transformations.
        sample_rate (int): Sampling rate on which the input data are sampled.

    Returns:
        data: Output data with applied sox effects.
    """

    if sox_effects_to_apply[1]:
        if sox_effects_to_apply[1][0] == 'mp3':
            data = AA.Mp3Compression(
                min_bitrate=int(sox_effects_to_apply[1][2]), max_bitrate=int(sox_effects_to_apply[1][2]), p=1
            )(data.squeeze(0).cpu().detach().numpy(), sample_rate=sample_rate)
            data = torch.from_numpy(data).unsqueeze(0)

        elif sox_effects_to_apply[1][0] in ('g726', 'gsm', 'amr'):
            if sox_effects_to_apply[1][0] == 'g726':
                suffix = '.wav'
            elif sox_effects_to_apply[1][0] == 'gsm':
                suffix = '.gsm'
            elif sox_effects_to_apply[1][0] == 'amr':
                suffix = '.amr'
            # Create temporary file for the output
            fd, tmp_output_path = tempfile.mkstemp(
                suffix=suffix)
            loaded_tensor_with_codec = None
            tmp_input_path = None
            try:
                # Create temporary file for the input
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
                    torchaudio.save(
                        tmp_input.name, data.to(
                            'cpu'), sample_rate
                    )
                    tmp_input_path = tmp_input.name
                    with os.fdopen(fd, 'w') as tmp:
                        if sox_effects_to_apply[1][0] == 'g726':
                            g726_audio_bitrate = sox_effects_to_apply[1][2]
                            apply_g726(
                                tmp_input_path,
                                tmp_output_path,
                                g726_audio_bitrate
                            )

                        elif sox_effects_to_apply[1][0] == 'gsm':
                            apply_gsm(
                                tmp_input_path,
                                tmp_output_path
                            )

                        elif sox_effects_to_apply[1][0] == 'amr':
                            amr_audio_bitrate = sox_effects_to_apply[1][2]
                            apply_amr(
                                tmp_input_path,
                                tmp_output_path,
                                amr_audio_bitrate
                            )

                        # Delete the temporary input file
                        os.remove(tmp_input_path)
                        loaded_tensor_with_codec, fs = torchaudio.load(
                            tmp_output_path)
                        loaded_tensor_with_codec = T.Resample(
                            orig_freq=fs, new_freq=sample_rate)(loaded_tensor_with_codec)
            finally:
                # Delete the temporary output file
                os.remove(tmp_output_path)
            data = loaded_tensor_with_codec

        else:
            data = TIO.AudioEffector(format="wav", encoder=sox_effects_to_apply[1][0]).apply(
                data.cpu().transpose(0, 1), sample_rate).transpose(0, 1).cpu()

    return data


class AugmentWaveform:
    def __init__(
        self,
        transformations=None,
        device=None,
        sox_effects=None,
        sample_rate=None,
        verbose=False,
    ):
        '''
        This class is used to apply transformations to the input waveform.
        It is supposed to be initiated with the list of
            transformations - list of transformations generated using transf_gen function
            or
            sox_effects - sox effects list generated using sox_parser function.

        When this class is called it handles differences between multiple tools for audio augmentation 
        and depending on the transformation list input it applies the correct transformation to the input data.
        It solves the hassle that the user has to go through while wanting to 
        apply multiple transformations from different libraries and tools together.

        See the examples folder for examples of how to use this class.

        Args:
            transformations (list): List of transformations to be applied to the data. 
                Use transf_gen component to generate this list correctly.
            device ('cpu' or 'cuda'): The device to which the data should be moved.
            sox_effects (list): List of lists, where inner lists contain sox effects with their parameters.
                                Use sox_parser component to generate this list correctly.
            sample_rate (int): Sampling rate on which the input data are sampled.
            verbose (bool): If True, print the current transformation being applied.

        Returns:
            waveform: Waveform which has applied all the transformations from the user input.
        '''

        if transformations is None and sox_effects is None:
            raise ValueError(
                'At least one of transformations or sox_effects must be specified!'
            )
        if transformations is not None and sox_effects is not None:
            raise ValueError(
                'Only one of transformations or sox_effects can be specified!'
            )
        if transformations is not None:
            # Remove Nones from transformations list
            self.transformations = [
                trans for trans in transformations if trans is not None]
        else:
            self.transformations = transformations

        self.device = device
        self.sox_effects = sox_effects
        self.sample_rate = sample_rate
        self.verbose = verbose

    def __call__(self, waveform):
        '''
        Args:
            waveform (numpy.ndarray): Waveform to which the user wants to apply the transformations.
                Waveform should be a 1D numpy array.
        '''

        if self.transformations is not None:
            for transform in self.transformations:
                print(
                    f'CURRENT TRANSFORM: {transform}') if self.verbose else None
                if transform.__class__.__module__.split('.')[0] == AA.__name__:
                    waveform = transform(
                        waveform, sample_rate=self.sample_rate)

                elif transform.__class__.__name__ == 'ApplyRIR':
                    x = transform.audio_sample_rate
                    if isinstance(x, dict):
                        transform = rir_setup.ApplyRIR(**x)
                    else:
                        transform = rir_setup.ApplyRIR(**x())
                    waveform = transform(waveform)

                elif transform.__class__.__module__.split('.')[-1] == '_effector':
                    transposed_waveform_tensor = torch.from_numpy(
                        np.expand_dims(waveform, axis=1).transpose(0, 1)
                    )
                    waveform = transform.apply(
                        transposed_waveform_tensor, self.sample_rate
                    ).transpose(0, 1).squeeze(0).numpy()

                elif isinstance(transform, tuple):
                    data = torch.from_numpy(waveform).float()
                    suffix = None
                    if transform[0] == 'g726':
                        suffix = '.wav'
                    elif transform[0] == 'gsm':
                        suffix = '.gsm'
                    elif transform[0] == 'amr':
                        suffix = '.amr'

                    fd, tmp_output_path = tempfile.mkstemp(suffix=suffix)
                    loaded_tensor_with_codec = None
                    tmp_input_path = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
                            torchaudio.save(
                                tmp_input.name, data.to('cpu').unsqueeze(0), self.sample_rate)
                            tmp_input_path = tmp_input.name
                            with os.fdopen(fd, 'w') as tmp:
                                if transform[0] == 'g726':
                                    g726_audio_bitrate = transform[1]['audio_bitrate']
                                    apply_g726(
                                        tmp_input_path,
                                        tmp_output_path,
                                        g726_audio_bitrate
                                    )
                                elif transform[0] == 'gsm':
                                    apply_gsm(
                                        tmp_input_path,
                                        tmp_output_path
                                    )
                                elif transform[0] == 'amr':
                                    amr_audio_bitrate = transform[1]['audio_bitrate']
                                    apply_amr(
                                        tmp_input_path,
                                        tmp_output_path,
                                        amr_audio_bitrate
                                    )

                                # Delete the temporary input file
                                os.remove(tmp_input_path)
                                loaded_tensor_with_codec, fs = torchaudio.load(
                                    tmp_output_path)
                                loaded_tensor_with_codec = T.Resample(
                                    orig_freq=fs, new_freq=self.sample_rate)(loaded_tensor_with_codec)
                    finally:
                        # Delete the temporary output file
                        os.remove(tmp_output_path)

                    waveform = loaded_tensor_with_codec.unsqueeze(
                        0).cpu().numpy().squeeze(0).squeeze(0)
                elif transform.__class__.__module__.split('.')[0] == 'torchaudio' or isinstance(transform, dict):
                    if isinstance(transform, dict):
                        tmp = next(iter(transform))
                        p = transform['p']
                        currently_used_trasnf = get_transf(tmp, TTW_TRANSF)
                        transform = torch_randomizer(
                            getattr(TTW, currently_used_trasnf)(**transform[tmp]), p=p)
                        if transform is None:
                            continue

                    transform = transform.to(self.device)
                    waveform = torch.from_numpy(waveform).float().to(self.device)
                    waveform = waveform.unsqueeze(0)
                    waveform = transform(waveform)
                    if isinstance(waveform, tuple):
                        waveform = waveform[0]
                    waveform = waveform.detach().cpu().numpy()
                    waveform = waveform.squeeze(0)

                elif transform.__class__.__module__.split('.')[0] == 'torch_audiomentations':
                    transform = transform.to(self.device)
                    waveform = torch.from_numpy(waveform).float().to(self.device)
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                    waveform = transform(waveform).detach().cpu().numpy()
                    waveform = waveform.squeeze(0).squeeze(0)

        # @@@@@@@@@@@@@@@@@@@@@@@@ SOX EFFECTS PART OF THE CODE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        if self.sox_effects is not None:
            if self.device == 'cuda':
                print('WARNING: sox_effects are not supported on cuda!\n\t Using cpu.')
            # check if waveform is torch.tensor, if not convert it to torch.tensor
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform).float().unsqueeze(0)

            if isinstance(self.sox_effects, str):
                sox_effects_to_apply = parse(self.sox_effects)
                # Check if sox_effects is a list of lists --> single effect, input already parsed by sox_parer
                if (isinstance(sox_effects_to_apply[0], list) and
                        all(isinstance(sublist, list) for sublist in sox_effects_to_apply[0])):
                    waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                        waveform,
                        self.sample_rate,
                        sox_effects_to_apply[0]
                    )
                # Apply codec if it is present
                if sox_effects_to_apply[1]:
                    waveform = sox_codec_handler(
                        sox_effects_to_apply, waveform, self.sample_rate)

            # Check if sox_effects is a list of strings (file was read and entered) --> multiple effects -->
            # --> needs to be parsed by sox_parser and some effect needs to be randomly chosen and applied
            elif (isinstance(self.sox_effects, list) and
                  all(isinstance(sublist, str) for sublist in self.sox_effects)):
                random_sox = select_random_sox(self.sox_effects)
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform,
                    self.sample_rate,
                    random_sox[0]
                )
                # Apply codec if it is present
                if random_sox[1]:
                    waveform = sox_codec_handler(
                        random_sox, waveform, self.sample_rate)

            # Convert waveform back from tensor
            waveform = waveform.squeeze(0).detach().numpy()

        return waveform


class Collator:
    def __init__(
        self,
        transformations=None,
        device=None,
        sox_effects=None,
        sample_rate=None,
        verbose=False,
    ):
        '''
        This class is used to apply transformations to the batch of audio data.
        It's intended use is as a collate function for the PyTorch DataLoader.

        It is supposed to be initiated with the list of
            transformations - list of transformations generated using transf_gen function
            or
            sox_effects - sox effects list generated using sox_parser function.

        When this class is called it handles differences between multiple tools for audio augmentation 
        and depending on the transformation list input it applies the correct transformation to the input data.
        It solves the hassle that the user has to go through while wanting to 
        apply multiple transformations from different libraries and tools together.

        See the examples folder for examples of how to use this class.

        Args:
            transformations (list): List of transformations to be applied to the data.
            device ('cpu' or 'cuda'): The device to which the data should be moved.
            sox_effects (list): List of lists, where inner lists contain sox effects with their parameters.
                                Use sox_parser component to generate this list.
            sample_rate (int): Sampling rate on which the input data are sampled.
            verbose (bool): If True, print the current transformation being applied.

        Returns:
            Collator: Collate function which can be used within PyTorch's DataLoader.
        '''

        if transformations is None and sox_effects is None:
            raise ValueError(
                'At least one of transformations or sox_effects must be specified!')
        if transformations is not None and sox_effects is not None:
            raise ValueError(
                'Only one of transformations or sox_effects can be specified!')

        self.transformations = transformations
        self.device = device
        self.sox_effects = sox_effects
        self.sample_rate = sample_rate
        self.verbose = verbose

    def __call__(self, batch):
        '''
        Args:
            batch (list): Batch of audio data to which the user wants to apply the transformations. (Dataloader calls this class)
                Batch can look like this:
                [
                (tensor([[-0.0065, ...,  0.0033]]), 16000, 'CHAPTER ONE', 103, 1240), 
                (tensor([[-0.0059, ...,  0.0007]]), 16000, 'THAT HAD ITS', 103, 1240)
                ]

        Important thing is, that the audio data, needs to come first in each item of the batch.
        '''

        transformed_batch = []
        max_length = max(x[0].size(0) if x[0].dim() ==
                         1 else x[0].size(1) for x in batch)

        SPECTR_USED = False
        spectr_max_length = 0
        for item in batch:
            data, *rest = item
            # Add dimension
            data = data.unsqueeze(0)
            # Shape of the data at this point is [1, 1, 1605] (1605 is randomly chosen)
            # Apply all transformations from the transfomations chain
            if self.transformations is not None:
                # Move the data to the selected device
                data = data.to(self.device)

                # Remove Nones from transformations list
                self.transformations = [
                    trans for trans in self.transformations if trans is not None]
                p = None
                for transform in self.transformations:
                    print(
                        f'CURRENT TRANSFORM: {transform}') if self.verbose else None
                    if isinstance(transform, torch.nn.Module) or isinstance(transform, dict):
                        if isinstance(transform, dict):
                            tmp = next(iter(transform))
                            p = transform['p']
                            currently_used_trasnf = get_transf(tmp, TTW_TRANSF)
                            transform = torch_randomizer(
                                getattr(TTW, currently_used_trasnf)(**transform[tmp]), p=p
                            )
                            if transform is None:
                                continue
                        # Move the transform to the correct device
                        transform = transform.to(self.device)
                    # Check if current transform comes from audiomentations library
                    # because it needs to be called with sample_rate argument
                    if transform.__class__.__module__.split('.')[0] == AA.__name__:
                        data = data.squeeze(0).squeeze(
                            0).cpu().detach().numpy()
                        data = transform(data, sample_rate=self.sample_rate)
                        data = torch.from_numpy(data).unsqueeze(
                            0).unsqueeze(0).to(self.device)

                    # Handle torchaudio effector which applies codec to the data
                    elif transform.__class__.__module__.split('.')[-1] == '_effector':
                        # Data enters in this shape: [1, 1, 1605] (1605 is randomly chosen)
                        # Effect on data is [1, 1, 1605] -> [1, 1605] -> [1605, 1] -> [1, 1605] -> [1, 1, 1605]
                        data = transform.apply(
                            data.cpu().squeeze(0).transpose(0, 1), self.sample_rate
                        ).transpose(0, 1).unsqueeze(0).to(self.device)

                    # This branch handles applying codecs using ffmpeg
                    elif isinstance(transform, tuple):
                        suffix = None
                        if transform[0] == 'g726':
                            suffix = '.wav'
                        elif transform[0] == 'gsm':
                            suffix = '.gsm'
                        elif transform[0] == 'amr':
                            suffix = '.amr'

                        # Create temporary file for the output
                        fd, tmp_output_path = tempfile.mkstemp(suffix=suffix)
                        loaded_tensor_with_codec = None
                        tmp_input_path = None
                        try:
                            # Create temporary file for the input
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_input:
                                torchaudio.save(
                                    tmp_input.name, data.to('cpu')[0], self.sample_rate)
                                tmp_input_path = tmp_input.name
                                with os.fdopen(fd, 'w') as tmp:
                                    if transform[0] == 'g726':
                                        g726_audio_bitrate = transform[1]['audio_bitrate']
                                        apply_g726(
                                            tmp_input_path,
                                            tmp_output_path,
                                            g726_audio_bitrate
                                        )
                                    elif transform[0] == 'gsm':
                                        apply_gsm(
                                            tmp_input_path,
                                            tmp_output_path
                                        )
                                    elif transform[0] == 'amr':
                                        amr_audio_bitrate = transform[1]['audio_bitrate']
                                        apply_amr(
                                            tmp_input_path,
                                            tmp_output_path,
                                            amr_audio_bitrate
                                        )

                                    # Delete the temporary input file
                                    os.remove(tmp_input_path)
                                    loaded_tensor_with_codec, fs = torchaudio.load(
                                        tmp_output_path)
                                    loaded_tensor_with_codec = T.Resample(
                                        orig_freq=fs, new_freq=self.sample_rate)(loaded_tensor_with_codec)
                        finally:
                            # Delete the temporary output file
                            os.remove(tmp_output_path)

                        data = loaded_tensor_with_codec.unsqueeze(
                            0).to(self.device)

                    elif transform.__class__.__name__ == 'ApplyRIR':
                        x = transform.audio_sample_rate
                        if isinstance(x, dict):
                            transform = rir_setup.ApplyRIR(**x)
                        else:
                            transform = rir_setup.ApplyRIR(**x())

                        data = torch.from_numpy(
                            transform(data)).unsqueeze(0).unsqueeze(0).to(self.device)
                    else:
                        # Check if transform is Speed from torchaudio - needs specific handling,
                        # for some reason it returns tuple, we want to get only augmented data
                        if transform.__class__.__name__ == 'Speed':
                            data = transform(data)[0]
                        elif (
                            transform.__class__.__name__ == 'Spectrogram' or
                            transform.__class__.__name__ == 'MelSpectrogram' or
                            transform.__class__.__name__ == 'TimeMasking' or
                            transform.__class__.__name__ == 'FrequencyMasking'
                        ):
                            data = transform(data[0])
                            if (
                                transform.__class__.__name__ == 'TimeMasking' or
                                transform.__class__.__name__ == 'FrequencyMasking'
                            ):
                                data = data.unsqueeze(0)
                            SPECTR_USED = True

                            # Check if spectr_max_length should be changed
                            if data.size(2) > spectr_max_length:
                                spectr_max_length = data.size(2)

                        else:
                            data = transform(data)
                    # Check that only tensor was returned, if not make it a tensor
                    # This is done because some transforms return a tuple of (tensor, something)
                    if not isinstance(data, torch.Tensor):
                        data = data[0]
                    # Check if max_length has been changed
                    if data.size(2) > max_length:
                        max_length = data.size(2)

            # @@@@@@@@@@@@@@@@@@@@@@@@ SOX EFFECTS PART OF THE CODE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            if self.sox_effects is not None:
                if self.device == 'cuda' and data.is_cuda:
                    print('WARNING: sox_effects are not supported on cuda!\n\t Using cpu.')
                    data = data.cpu()

                if isinstance(self.sox_effects, str):
                    sox_effects_to_apply = parse(self.sox_effects)
                    # Check if sox_effects is a list of lists --> single effect, input already parsed by sox_parer
                    if (isinstance(sox_effects_to_apply[0], list) and
                            all(isinstance(sublist, list)
                                for sublist in sox_effects_to_apply[0])
                        ):
                        data, _ = torchaudio.sox_effects.apply_effects_tensor(
                            data.squeeze(0),
                            self.sample_rate,
                            sox_effects_to_apply[0]
                        )
                        if sox_effects_to_apply[1]:
                            data = sox_codec_handler(
                                sox_effects_to_apply, data, self.sample_rate)
                        data = data.unsqueeze(0)

                # Check if sox_effects is a list of strings --> multiple effects -->
                # --> needs to be parsed by sox_parser and some effect needs to be randomly chosen and applied
                elif (isinstance(self.sox_effects, list) and
                      all(isinstance(sublist, str)
                          for sublist in self.sox_effects)
                      ):
                    random_sox = select_random_sox(self.sox_effects)
                    print(
                        f'CURRENT SOX THE IS BEIGN APPLIED: {random_sox}') if self.verbose else None
                    data, _ = torchaudio.sox_effects.apply_effects_tensor(
                        data.squeeze(0),
                        self.sample_rate,
                        random_sox[0]
                    )
                    if random_sox[1]:
                        data = sox_codec_handler(
                            random_sox, data, self.sample_rate)
                    data = data.unsqueeze(0)

            # DATA AFTER SOX EFFECTS: torch.Size([1, 1, 225600])
            data = data.detach()
            transformed_batch.append((data, *rest))

        # Pad all samples to new max_length
        if SPECTR_USED:
            for i, (data, *rest) in enumerate(transformed_batch):
                if data.size(2) < spectr_max_length:
                    data = torch.nn.functional.pad(
                        data, (0, spectr_max_length - data.size(2)))
                    transformed_batch[i] = (data, *rest)
        else:
            for i, (data, *rest) in enumerate(transformed_batch):
                if data.size(2) < max_length:
                    data = torch.nn.functional.pad(
                        data, (0, max_length - data.size(2)))
                    transformed_batch[i] = (data, *rest)

        return torch.utils.data.dataloader.default_collate(transformed_batch)


class AugmentLocalAudioDataset:
    def __init__(
        self,
        transformations=None,
        device=None,
        sox_effects=None,
        sample_rate=None,
        verbose=False,
    ):
        self.transformations = transformations
        self.device = device
        self.sox_effects = sox_effects
        self.sample_rate = sample_rate
        self.verbose = verbose
        
        self.augment_waveform = AugmentWaveform(
            transformations=self.transformations,
            device=self.device,
            sox_effects=self.sox_effects,
            sample_rate=self.sample_rate,
            verbose=self.verbose,
        )
    """
    This is a wrapper class for AugmentWaveform class and it enables to augment local audio datasets.
    """

    def __call__(self, input_dir: str, output_dir: str):
        # Ensure output directory exists
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        # Iterate over all files in the input directory
        for file in Path(input_dir).glob('*'):
            if file.is_file():
                # Load the audio file
                loaded_file_extension = str(file).split(".")[-1]
                if loaded_file_extension not in ('wav', 'flac', 'ogg', 'mp3', 'aac', 'opus', 'm4a', 'wma', 'ac3'):
                    raise ValueError(
                        f'File {file} has unsupported extension {loaded_file_extension}. Only .wav and .flac files are supported.'
                    )
                tensor, sample_rate = torchaudio.load(str(file))
                waveform = tensor[0].numpy()

                # Apply transformations
                transformed_waveform = self.augment_waveform(waveform)
                # Save the transformed audio to the output directory
                output_file = output_dir_path / file.name
                write(str(output_file), sample_rate, transformed_waveform)

                if self.verbose:
                    print(f"Processed file: {file.name}")
