"""
File containg class that handles generating room impulse responses (RIRs) and applying them to the data.

@author: Ladislav Vašina, github: LadislavVasina1
"""

import pyroomacoustics as pra
import torch
import audiomentations as AA
import torchaudio.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
from matplotlib.path import Path
from shapely.geometry import Point, Polygon

cos = np.cos
sin = np.sin
arctan2 = np.arctan2
uniform = np.random.uniform
mean = np.mean
pi = np.pi


def plot_room(room, speaker, microphone):
    '''
    This function plots the room, speaker (red cross) and microphone (blue cross).

    Args:
        room: List of lists, containing the coordinates of the room corners.
        speaker: List, containing the coordinates of the speaker.
        microphone: List, containing the coordinates of the microphone.
    
    Returns:
        None
    '''

    # Create a new figure
    plt.figure()

    # Plot the room
    room = np.array(room)
    plt.plot(np.append(room[:, 0], room[0, 0]), np.append(room[:, 1], room[0, 1]), 'k-')

    # Plot the speaker
    plt.plot(speaker[0], speaker[1], 'rx')

    # Plot the microphone
    plt.plot(microphone[0], microphone[1], 'bx')

    # Set the aspect ratio of the plot to equal to ensure the room is displayed correctly
    plt.gca().set_aspect('equal', adjustable='box')

    # Show the plot
    plt.show()


def is_collinear(p1, p2, p3):
    '''
    Calculate the area of the triangle formed by p1, p2 and p3
    If the area is 0, the points are collinear (lying in the same straight line).
    https://www.quora.com/How-do-I-prove-that-three-points-p-q-and-r-are-collinear-using-area-concepts

    Args:
        p1: List, containing the coordinates of the first point.
        p2: List, containing the coordinates of the second point.
        p3: List, containing the coordinates of the third point.
    
    Returns:
        area == 0: Boolean, True if the points are collinear, False otherwise.
    '''

    area = (
        p1[0]*(p2[1] - p3[1]) + 
        p2[0]*(p3[1] - p1[1]) + 
        p3[0]*(p1[1] - p2[1])
    )
    return area == 0


def generate_random_polygon(num_vertices, x_range=(0, 10), y_range=(0, 10)):
    '''
    This function generates a random polygon with a given number of vertices.

    Args:
        num_vertices: Integer, number of vertices of the polygon.
        x_range: Tuple of two integers, range of x coordinates for the polygon.
        y_range: Tuple of two integers, range of y coordinates for the polygon.
    
    Returns:
        points: List of lists, containing the coordinates of the polygon vertices.
    '''

    # Initialize the set of points
    points_set = set()

    # Generate unique points
    while len(points_set) < num_vertices:
        # Generate a random angle
        angle = np.random.rand() * 2 * pi

        # Generate a random radius
        radius = np.random.rand() * min(x_range[1] - x_range[0], y_range[1] - y_range[0]) / 2

        # Generate a point
        point = (radius * cos(angle) + (x_range[0] + x_range[1]) / 2,
                 radius * sin(angle) + (y_range[0] + y_range[1]) / 2)

        # Round the coordinates
        point = (round(point[0]), round(point[1]))

        # Check for collinearity with any two existing points
        for p1, p2 in itertools.combinations(points_set, 2):
            if is_collinear(p1, p2, point):
                break
        else:
            # If the point is not collinear with any two existing points, add it to the set
            points_set.add(point)

    # Convert the set of points to a list of lists
    points = [list(point) for point in points_set]

    # Convert points to a numpy array for the following operations
    points = np.array(points)

    # Order the points clockwise
    center = mean(points, axis=0)
    angles = arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    points = points[np.argsort(-angles)]

    # Convert points back to a list of lists
    points = points.tolist()

    return points


def generate_random_point_in_polygon(polygon):
    '''
    This function generates a random point in a given polygon.

    Args:
        polygon: List of lists, containing the coordinates of the polygon vertices.

    Returns:
        point: List, containing the coordinates of the generated point.
    '''

    min_x = min(point[0] for point in polygon)
    max_x = max(point[0] for point in polygon)
    min_y = min(point[1] for point in polygon)
    max_y = max(point[1] for point in polygon)

    poly = Polygon(polygon)

    fail_counter = 0
    while True:
        x = round(uniform(min_x, max_x), 1)
        y = round(uniform(min_y, max_y), 1)
        point = Point(x, y)
        if poly.contains(point):
            return [x, y]
        
        fail_counter += 1
        if fail_counter > 100:
            raise ValueError('Failed to generate a point in the polygon after 100 attempts!')


def generate_speaker_and_microphone_coordinates(polygon):
    '''
    This function generates random speaker and microphone coordinates in a given polygon.

    Args:
        polygon: List of lists, containing the coordinates of the polygon vertices.

    Returns:
        speaker_coordinates: List, containing the coordinates of the generated speaker.
        microphone_coordinates: List, containing the coordinates of the generated microphone.
    '''

    speaker_coordinates = generate_random_point_in_polygon(polygon)
    microphone_coordinates = generate_random_point_in_polygon(polygon)
    
    while speaker_coordinates == microphone_coordinates:
        microphone_coordinates = generate_random_point_in_polygon(polygon)
        
    return speaker_coordinates, microphone_coordinates


def generate_number_of_rooms_with_speaker_and_microphone_coordinates(num_rooms, x_range, y_range, num_vertices_range):
    '''
    This function generates a given number of rooms with random speaker and microphone coordinates.

    Args:
        num_rooms: Integer, number of rooms to generate.
        x_range: Tuple of two integers, range of x coordinates for the rooms.
        y_range: Tuple of two integers, range of y coordinates for the rooms.
        num_vertices_range: Tuple of two integers, range of number of vertices for the rooms.

    Returns:
        rooms: List of tuples, containing the coordinates of the room vertices, speaker coordinates and microphone coordinates.
    '''
    
    rooms = []
    for _ in range(num_rooms):
        num_vertices = random.randint(*num_vertices_range)
        polygon = generate_random_polygon(num_vertices, x_range, y_range)
        speaker_coordinates, microphone_coordinates = generate_speaker_and_microphone_coordinates(polygon)
        rooms.append((polygon, speaker_coordinates, microphone_coordinates))
    return rooms


def create_random_rir_kwargs(
        x_range,
        y_range,
        num_vertices_range,
        mic_height,
        source_height,
        audio_sample_rate,
        walls_mat,
        room_height,
        max_order,
        floor_mat,
        ceiling_mat,
        ray_tracing,
        air_absorption,
        ):
    '''
    This function generates random rir_kwargs for ApplyRIR class.

    Args:
        x_range: Tuple of two integers, range of x coordinates for the room.
        y_range: Tuple of two integers, range of y coordinates for the room.
        num_vertices_range: Tuple of two integers, range of number of vertices for the room.
        mic_height: Float, height of the microphone in meters.
        source_height: Float, height of the source in meters.
        audio_sample_rate: Integer, sample rate of the audio.
        walls_mat: String, material of the walls.
        room_height: Float, height of the room in meters.
        max_order: Integer, maximum order of the RIR.
        floor_mat: String, material of the floor.
        ceiling_mat: String, material of the ceiling.
        ray_tracing: Boolean, whether to use ray tracing.
        air_absorption: Boolean, whether to use air absorption.

    Returns:
        rir_kwargs: Dictionary, containing the parameters for the ApplyRIR class.
    '''

    generated_rooms_sources_mics = generate_number_of_rooms_with_speaker_and_microphone_coordinates(1,x_range, y_range, num_vertices_range)
    rir_kwargs_list = []
    for room in generated_rooms_sources_mics:
        selected_corners, selected_source, selected_mics = room
        selected_source.append(source_height)
        selected_mics.append(mic_height)
        rir_kwargs = {
            'audio_sample_rate': audio_sample_rate,
            'corners_coord': selected_corners,
            'walls_mat': walls_mat,
            'room_height': room_height,
            'max_order': max_order,
            'floor_mat': floor_mat,
            'ceiling_mat': ceiling_mat,
            'ray_tracing': ray_tracing,
            'air_absorption': air_absorption,
            'source_coord': selected_source,
            'microphones_coord': selected_mics,
        }
        rir_kwargs_list.append(rir_kwargs)

    return rir_kwargs_list[0]


class RandomRIRKwargs:
    '''
    Wrapper class around create_random_rir_kwargs function.
    '''

    def __init__(
        self,
        x_range,
        y_range,
        num_vertices_range,
        mic_height,
        source_height,
        audio_sample_rate,
        walls_mat,
        room_height,
        max_order,
        floor_mat,
        ceiling_mat,
        ray_tracing,
        air_absorption,
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.num_vertices_range = num_vertices_range
        self.mic_height = mic_height
        self.source_height = source_height
        self.audio_sample_rate = audio_sample_rate
        self.walls_mat = walls_mat
        self.room_height = room_height
        self.max_order = max_order
        self.floor_mat = floor_mat
        self.ceiling_mat = ceiling_mat
        self.ray_tracing = ray_tracing
        self.air_absorption = air_absorption
    
    def __call__(self):
        return create_random_rir_kwargs(
            self.x_range,
            self.y_range,
            self.num_vertices_range,
            self.mic_height,
            self.source_height,
            self.audio_sample_rate,
            self.walls_mat,
            self.room_height,
            self.max_order,
            self.floor_mat,
            self.ceiling_mat,
            self.ray_tracing,
            self.air_absorption,
        )


def get_all_materials_info():
    '''
    This function prints available absorption and scattering materials
    that are available through the pyroomacoustics library.

    For more details see:
    https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.materials.database.html
    '''
    
    x = pra.materials_data
    center_freqs = x['center_freqs']
    absorption = x['absorption']
    scattering = x['scattering']

    print('@'*79)
    print('CENTER FREQUENCIES:')
    print(f'{center_freqs}')
    print('@'*79)
    print('ABSORPTION:')
    for item in absorption.items():
        key, value = item
        print(f'\t{key}:')

        for key, value in value.items():
            print(f'\t\t{key}: {value}')

    print('@'*79)
    print('SCATTERING:')
    for item in scattering.items():
        key, value = item
        print(f'\t{key}:')

        for key, value in value.items():
            print(f'\t\t{key}: {value}')
    print('@'*79)


class ApplyRIR:
    """This class applies room impulse response (RIR) to the input audio signal."""

    def __init__(
        self,
        audio_sample_rate=None,
        corners_coord=None,
        walls_mat=None,
        room_height=None,
        floor_mat=None,
        ceiling_mat=None,
        max_order=None,
        ray_tracing=None,
        air_absorption=None,
        source_coord=None,
        microphones_coord=None,
    ):
        self.audio_sample_rate = audio_sample_rate
        self.core_sample_rate = 16000
        # coordinates going clockwise around the room [[x1, y2], [x2, y2], ...]
        self.corners_coord = np.array(corners_coord).T
        self.walls_mat = walls_mat
        # Height in meters
        self.room_height = room_height
        self.floor_mat = floor_mat
        self.ceiling_mat = ceiling_mat
        # using lower max_order value will result in a quick (but less accurate) RIR
        self.max_order = max_order
        self.ray_tracing = ray_tracing
        self.air_absorption = air_absorption
        # All coordintes must be in form [[x], [y], [z]] or [[x1, x2, ...], [y1, y2, ...], [z1, z2, ...]]
        self.source_coord = source_coord
        self.microphones_coord = microphones_coord

    def __call__(self, waveform):
        # Handle device and if waveform is in the form of torch.Tensor convert it to NumPy format
        SAMPLE_RATE_HAS_CHANGED = False

        if isinstance(waveform, torch.Tensor):
            initial_wav_type = waveform.dtype
            initial_device = waveform.device
            if waveform.is_cuda:
                waveform = waveform.squeeze(0).squeeze(0).cpu()
            elif waveform.is_cpu:
                waveform = waveform.squeeze(0).squeeze(0)

        elif isinstance(waveform, np.ndarray):
            initial_wav_type = waveform.dtype

        if not (self.audio_sample_rate == self.core_sample_rate):
            SAMPLE_RATE_HAS_CHANGED = True
            if isinstance(waveform, torch.Tensor):
                waveform = T.Resample(self.audio_sample_rate,
                                    self.core_sample_rate)(waveform)
            else:
                waveform = AA.Resample(min_sample_rate=self.core_sample_rate, max_sample_rate=self.core_sample_rate, p=1.0)(waveform, self.audio_sample_rate)

        room = pra.Room.from_corners(
            self.corners_coord,
            fs=self.core_sample_rate,
            max_order=self.max_order,
            materials=pra.Material(self.walls_mat),
            ray_tracing=self.ray_tracing,
            air_absorption=self.air_absorption,
        )

        floor_and_ceiling = pra.make_materials(
            ceiling=self.ceiling_mat,
            floor=self.floor_mat
        )
        room.extrude(
            self.room_height,
            materials=floor_and_ceiling
        )

        if self.ray_tracing:
            # set the ray tracing parameters
            room.set_ray_tracing(
                n_rays=10000,
                receiver_radius=0.5,
                energy_thres=1e-5
            )

        # add source and set the signal to WAV file content
        room.add_source(
            self.source_coord,
            signal=waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform,
        )

        # add microphone/s
        R = np.array(self.microphones_coord)  # [[x], [y], [z]]
        room.add_microphone(R)

        # compute image sources
        room.image_source_model()

        # visualize 3D polyhedron room and image sources
        # fig, ax = room.plot(img_order=1)
        # fig.set_size_inches(5, 3)

        room.simulate()
        waveform = room.mic_array.signals[0, :]
        del(room)
        waveform = waveform.astype(np.float32)

        if initial_wav_type == torch.Tensor:
            waveform = torch.from_numpy(waveform)
            waveform = waveform.unsqueeze(0).unsqueeze(0).to(initial_device)

        if SAMPLE_RATE_HAS_CHANGED:
            if isinstance(waveform, torch.Tensor):
                waveform = T.Resample(self.core_sample_rate,
                                      self.audio_sample_rate)(waveform)
            else:
                waveform = AA.Resample(min_sample_rate=self.audio_sample_rate, max_sample_rate=self.audio_sample_rate, p=1.0)(waveform, self.core_sample_rate)

        return waveform
