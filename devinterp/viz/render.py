import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Optional
import numpy as np
import logging
from lucid.optvis import objectives, param, transform
from lucid.misc.io import show
from lucid.misc.redirected_relu_grad import redirected_relu_grad, redirected_relu6_grad

# Create logger with module name, e.g., lucid.misc.io.reading
log = logging.getLogger(__name__)


def make_t_image(param_f: Callable = None) -> torch.Tensor:
    if param_f is None:
        t_image = param.image(128)
    elif callable(param_f):
        t_image = param_f()
    else:
        raise TypeError(f"Incompatible type for param_f, {type(param_f)}")

    if not isinstance(t_image, torch.Tensor):
        raise TypeError("param_f should produce a Tensor, but instead created a "
                        + str(type(t_image)))
    else:
        return t_image


def make_transform_f(transforms: list = Optional[None]) -> Callable:
    if type(transforms) is not list:
        transforms = transform.standard_transforms
    transform_f = transform.compose(transforms)
    return transform_f


def make_optimizer(optimizer: Optional[Callable] = None) -> optim.Optimizer:
    if optimizer is None:
        return optim.Adam(lr=0.05)
    elif isinstance(optimizer, optim.Optimizer):
        return optimizer
    else:
        raise ValueError("Could not convert optimizer argument to usable optimizer. "
                         "Needs to be one of None or torch.optim.Optimizer instance.")


def render_vis(model: nn.Module, objective_f: Callable, param_f: Callable = None, optimizer: Callable = None,
               transforms: list = None, thresholds: tuple = (512,), print_objectives: list = None,
               verbose: bool = True, seed: int = 0):
    """Flexible optimization-based feature vis.

    There's a lot of ways one might wish to customize optimization-based
    feature visualization. It's hard to create an abstraction that stands up
    to all the things one might wish to try.

    This function probably can't do *everything* you want, but it's much more
    flexible than a naive attempt. The basic abstraction is to split the problem
    into several parts. Consider the arguments:

    Args:
        model: The model to be visualized, from Alex' modelzoo.
        objective_f: The objective our visualization maximizes.
            See the objectives module for more details.
        param_f: Paramaterization of the image we're optimizing.
            See the paramaterization module for more details.
            Defaults to a naively paramaterized [1, 128, 128, 3] image.
        optimizer: Optimizer to optimize with. Either tf.train.Optimizer instance,
            or a function from (graph, sess) to such an instance.
            Defaults to Adam with lr .05.
        transforms: A list of stochastic transformations that get composed,
            which our visualization should robustly activate the network against.
            See the transform module for more details.
            Defaults to [transform.jitter(8)].
        thresholds: A list of numbers of optimization steps, at which we should
            save (and display if verbose=True) the visualization.
        print_objectives: A list of objectives separate from those being optimized,
            whose values get logged during the optimization.
        verbose: Should we display the visualization when we hit a threshold?
            This should only be used in IPython.

    Returns:
        2D array of optimization results containing of evaluations of supplied
            param_f snapshotted at specified thresholds. Usually that will mean one or
            multiple channel visualizations stacked on top of each other.
    """


    if use_fixed_seed:
        torch.manual_seed(0)

    t_image = make_t_image(param_f)
    transform_f = make_transform_f(transforms)
    optimizer = make_optimizer(optimizer)

    global_step = 0

    images = []
    try:
        for i in range(max(thresholds) + 1):
            # Placeholder for optimization logic
            loss_, _ = 0, 0  # Placeholder for loss computation

            if i in thresholds:
                vis = t_image.numpy()
                images.append(vis)
                if verbose:
                    print(i, loss_)
                    # Placeholder for print_objective_func
                    show(np.hstack(vis))
    except KeyboardInterrupt:
        log.warning(f"Interrupted optimization at step {i + 1}.")
        vis = t_image.numpy()
        show(np.hstack(vis))

    return images
