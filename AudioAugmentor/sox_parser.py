"""
File containing functions that parse pseudo-sox commands into lists of arguments.

@author: Ladislav Vašina, github: LadislavVasina1
"""

import torch
import torchaudio
import torchaudio.transforms as T
from random import choice

# Build pip package with this uncommented
# from .transf_gen import CODECS

# Test sources with this uncommented
from transf_gen import CODECS

EFFECTS_NAMES = torchaudio.sox_effects.effect_names()

def load_sox_file(file_path):
    """
    This function reads file containing pseudo-sox commands
    on each line and returns a list of lines from the file.
    This function also ignores lines starting with #

    Args:
        file_path: path to the file containing pseudo-sox commands
    Returns:
        sox_file_content: list of pseudo-sox commands
    """

    with open(file_path, 'r') as file:
        sox_file_content = file.readlines()
        sox_file_content = [line for line in sox_file_content if not line.strip().startswith('#')]
    
    return sox_file_content


def split_list(input_list):
    '''
    Args:
        input_list: list of arguments for sox effects
    Returns:
        list of lists of arguments for sox effects


    This function splits a list of arguments into a list of lists of arguments.
    Each sublist contains the arguments for a single effect.
    Arguments are split by looking for effect names in the list.
    If there are no effect names in the list, then the whole list is returned.
    If there are effect names in the list, then the list is split into sublists
    where each sublist starts with an effect name.
    '''

    final_args_list = []
    current_args_list = []

    for item in input_list:
        if item in EFFECTS_NAMES:
            if current_args_list:
                final_args_list.append(current_args_list)
            current_args_list = [item]
        else:
            current_args_list.append(item)

    if current_args_list:
        final_args_list.append(current_args_list)
    
    return final_args_list


def parse(command_string, verbose=False):
    '''
    Args:
        command_string (str): string of "sox effects command"
    Returns:
        list containing arguments of the "sox effects command"

        
    This function takes pseudo-sox commands in the following form:
    --sox="norm gain 0 highpass 1000 phaser 0.5 0.6 1 0.45 0.6 -s"

    and returns a list of arguments between the quotes.
    '''

    # Get sox command from the string
    command_list = command_string.split('"')[1].split()
    # Get codec part from the string
    codec_part = command_string.split('"')[2].split()
    # Check that there is some codec entered by user
    if codec_part:
        # Lowercase the codec part
        codec_part = [i.lower() for i in codec_part]
        # Check if the codec is supported
        if codec_part[0] not in CODECS and codec_part[0] != 'mp3':
            raise ValueError(f'Unsupported codec: {codec_part[0]}')
        
    if verbose:
        print(f'{command_string}\n{command_list}\n\n')

    return split_list(command_list), codec_part


def select_random_sox(read_sox_file, verbose=False):
    """
    This function accepts a list of pseudo-sox commands read from file line by line 
    (each element in the list is read line from the file) 
    and returns parsed and randomly selected command from the provided list.

    Args:
        read_sox_file(list): list of pseudo-sox commands
        verbose(bool): if True, prints the loaded lines and the chosen sox command
    Returns:
        parsed_sox(list): a list of arguments for the chosen sox command
    """

    chosen_sox_cmd = choice(read_sox_file)

    if verbose:
        print('Loaded lines:')
        for line in read_sox_file:
            print(line.replace('\n', ''))
        print(f'\nChosen sox command: \n{chosen_sox_cmd}')
        print(f'Parsed sox command: \n{parse(chosen_sox_cmd)}')

    parsed_sox = parse(chosen_sox_cmd)
    return parsed_sox


def select_random_sox_from_file(file_path, verbose=False):
    """
    This function reads file containing pseudo-sox commands
    on each line and returns a random line from the file.

    Args:
        file_path: path to the file containing pseudo-sox commands
        verbose: if True, prints the loaded lines and the chosen sox command
    Returns:
        chosen_sox_cmd: a list of arguments for the chosen sox command
    """

    with open(file_path, 'r') as file:
        sox_file_content = file.readlines()
        sox_file_content = [line for line in sox_file_content if not line.strip().startswith('#')]


    chosen_sox_cmd = choice(sox_file_content)

    if verbose:
        print('Loaded lines:')
        for line in sox_file_content:
            print(line.replace('\n', ''))
        print(f'\nChosen sox command: \n{chosen_sox_cmd}')
    
    return chosen_sox_cmd
