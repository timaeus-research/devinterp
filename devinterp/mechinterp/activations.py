from contextlib import contextmanager
from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

Transform = Callable[[torch.Tensor], torch.Tensor]

class ActivationProbe:
    """
    A utility class to extract the activation value of a specific neuron within a neural network.
    
    The location of the target neuron is defined using a string that can specify the layer, channel, and spatial coordinates (y, x). The format allows flexibility in defining the location:
    
    - 'layer1.0.conv1.weight.3': Only channel is specified; y and x default to the center (or one-off from center if the width and height are even).
    - 'layer1.0.conv1.weight.3.2': Channel and y are specified; x defaults to center.
    - 'layer1.0.conv1.weight.3..2': Channel and x are specified; y defaults to center.
    - 'layer1.0.conv1.weight.3.2.2': Channel, y, and x are all specified.
    
    The class provides methods to register a forward hook into a PyTorch model to capture the activation of the specified neuron during model inference.
    
    Attributes:
        model: The PyTorch model from which to extract the activation.
        location (str): The dot-separated path specifying the layer, channel, y, and x coordinates.
        activation: The value of the activation at the specified location.
        
    Example:
        model = ResNet18()
        extractor = ActivationProbe(model, 'layer1.0.conv1.weight.3')
        handle = extractor.register_hook()
        output = model(input_tensor)
        print(extractor.activation)  # Prints the activation value

    # TODO: Allow wildcards in the location string to extract composite activations (for an entire layer at a time).
    # TODO: Implement basic arithmetic operators so that you can combine activations (e.g. `0.4 * ActivationProbe(model, 'layer1.0.conv1.weight.3') + 0.6 * ActivationProbe(model, 'layer1.0.conv1.weight.4')`).
    """
    
    def __init__(self, model, location):
        self.activation = None
        self.model = model
        self.location = location.split('.')
        self.layer_path = []
        self.channel = None
        self.y = None
        self.x = None

        # Split the location into layer path and neuron indices
        state_dict_keys = list(model.state_dict().keys())
        for part in self.location:
            self.layer_path.append(part)
            path = '.'.join(self.layer_path)
            
            if any(key.startswith(path) for key in state_dict_keys):
                continue
            else:
                self.layer_path.pop()
                self.channel, *yx = map(int, self.location[len(self.layer_path):])
                if yx:
                    self.y = yx[0]
                    if len(yx) > 1:
                        self.x = yx[1]
                break

        # Get the target layer
        self.layer = model
        for part in self.layer_path[:-1]:
            self.layer = getattr(self.layer, part)

    def hook_fn(self, module, input, output):
        self.activation = output

    def register_hook(self):
        handle = self.layer.register_forward_hook(self.hook_fn)
        return handle
    
    @contextmanager
    def watch(self):
        handle = self.register_hook()
        yield
        handle.remove()

