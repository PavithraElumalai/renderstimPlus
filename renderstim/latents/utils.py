import numpy as np
import pyquaternion as pyquat
from kubric.core import objects
import png
from PIL import Image
import os
import pathlib
import json


def default_rng():
    return np.random.RandomState()

def get_gso_ids():
    with open(os.path.join(pathlib.Path().resolve().parent,"GSO.json")) as f:
        gso_assets = json.load(f)['assets']
    GSO_IDS = list(gso_assets.keys())
    
    return GSO_IDS
  

def position_sampler(region):
    region = np.array(region, dtype=np.float32)

    def _sampler(obj: objects.PhysicalObject, rng):
        obj.position = (0, 0, 0)  # reset position to origin
        effective_region = np.array(region) - obj.aabbox
        obj.position = rng.uniform(*effective_region)

    return _sampler


def resample_while(
    asset, 
    samplers, 
    condition, 
    max_trials=1000, 
    rng=default_rng()
):
    for _ in range(max_trials):
        for sampler in samplers:
            sampler(asset, rng)
        if not condition(asset):
            return
    else:
        raise RuntimeError("Failed to place", asset)
        # print("Failed to place ", asset)


def figure_out_overlap(
    asset,
    simulator,
    spawn_region=[[-1., -1., -1.], [1., 1., 1.]],
    max_trials=1000,
    rng=default_rng()
):

    return resample_while(
        asset,
        samplers=[position_sampler(spawn_region)],
        condition=simulator.check_overlap,
        max_trials=max_trials,
        rng=rng
    )


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1141])


def get_quaternion(axis, angle):
    """
    axis: X, Y, or Z
    angle: in radians from 0 to 2pi
    """
    ax = {
        "X": (1., 0., 0.), 
        "Y": (0., 1., 0.), 
        "Z": (0., 0., 1.)
    }
    axis = ax[axis.upper()]
    quat = pyquat.Quaternion(axis=axis, angle=angle)
    return tuple(quat)


def write_png(data: np.array, filename: str) -> None:
    if data.dtype in [np.uint32, np.uint64]:
        max_value = np.amax(data)
        if max_value > 65535:
            raise ValueError(f"max value of {max_value} exceeds uint16 bounds")
        data = data.astype(np.uint16)
        
    elif data.dtype in [np.float32, np.float64]:
        min_value = np.amin(data)
        max_value = np.amax(data)
        if min_value < 0.0 or max_value > 1.0:
            raise ValueError(f"Need values in [0, 1] but got [{min_value}, {max_value}]")
        data = (data * 65535).astype(np.uint16)
        
    elif data.dtype in [np.uint8, np.uint16]:
        pass
    else:
        raise NotImplementedError(f"Cannot handle {data.dtype}.")

    bitdepth = 8 if data.dtype == np.uint8 else 16

    assert data.ndim == 3, data.shape
    height, width, channels = data.shape
    
    greyscale = (channels == 1)
    alpha = (channels == 4)
    
    w = png.Writer(
        width=width, 
        height=height, 
        greyscale=greyscale, 
        bitdepth=bitdepth, 
        alpha=alpha
    )

    if channels == 2:
        data = np.concatenate(
            [data, np.zeros_like(data[:, :, :1])], 
            axis=-1
        )

    data = data.reshape(height, -1)
    with open(filename, "wb") as fp:
        w.write(fp, data)
        

def get_array_from_png(frame, path):
    write_png(frame, path)
    return np.asarray(Image.open(path))