import random

import numpy as np
import torch
import torch.nn.functional as F


def jitter(d: int):
    def inner(t_image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = t_image.shape
        i = random.randint(0, d - 1)
        j = random.randint(0, d - 1)
        return t_image[:, :, i: h - d + i, j: w - d + j]
    return inner


def pad(w: int, mode="reflect", constant_value=0.5):
    def inner(t_image: torch.Tensor) -> torch.Tensor:
        padding_mode = mode.lower()
        if constant_value == "uniform":
            constant_value = random.uniform(0, 1)
        return F.pad(t_image, (w, w, w, w), mode=padding_mode, value=constant_value)
    return inner


def random_scale(scales, seed=None):
    def inner(t: torch.Tensor) -> torch.Tensor:
        scale = random.choice(scales)
        _, _, h, w = t.shape
        return F.interpolate(t, size=(int(h * scale), int(w * scale)), mode="bilinear")
    return inner


def random_rotate(angles, units="degrees", seed=None):
    def inner(t: torch.Tensor) -> torch.Tensor:
        angle = random.choice(angles)
        if units.lower() == "degrees":
            angle = np.pi * angle / 180.
        return torch.rot90(t, k=angle, dims=[2, 3])  # TODO: Make continuous
    return inner


def normalize_gradient(grad_scales=None):
    def inner(x: torch.Tensor) -> torch.Tensor:
        grad_norm = torch.sqrt((x ** 2).sum(dim=[1, 2, 3], keepdim=True))
        if grad_scales is not None:
            grad_scales = torch.tensor(grad_scales, dtype=torch.float32)
            x *= grad_scales[:, None, None, None]
        return x / grad_norm
    return inner


def compose(transforms):
    def inner(x: torch.Tensor) -> torch.Tensor:
        for transform in transforms:
            x = transform(x)
        return x
    return inner

def collapse_alpha_random(sd=0.5):
    def inner(t_image: torch.Tensor) -> torch.Tensor:
        rgb, a = t_image[..., :3], t_image[..., 3:4]
        rand_img = torch.rand_like(rgb) * sd
        return a * rgb + (1 - a) * rand_img
    return inner


def crop_or_pad_to(height: int, width: int):
    def inner(t_image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t_image, size=(height, width), mode="bilinear")
    return inner

# Example usage
standard_transforms = [
    pad(12, mode="constant", constant_value=.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50. for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]
