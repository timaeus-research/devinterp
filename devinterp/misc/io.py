import functools
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Container, Dict, List, Optional, Tuple, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import wandb
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchvision import datasets, transforms
from tqdm.notebook import tqdm


def gen_images(*images: torch.Tensor, nrow=None) -> np.ndarray:
    # Process the optimized input
    images = [img.detach().cpu().squeeze(0) for img in images]
    images = [img - img.min() for img in images]
    images = [img / img.max() for img in images]

    # Create grid
    grid_image = vutils.make_grid(images, nrow=nrow or len(images))

    # Convert to numpy and transpose for plotting
    grid_image_np = grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

    return grid_image_np


def show_images(*images: torch.Tensor, nrow=None, **kwargs):
    grid_image_np = gen_images(*images, nrow=nrow)

    # Display using matplotlib
    plt.figure(**kwargs)  # You can change the size as you prefer
    plt.imshow(grid_image_np)
    plt.axis('off') # to remove the axis
    plt.show()