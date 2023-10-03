from contextlib import contextmanager
from typing import Callable

import torch

Transform = Callable[[torch.Tensor], torch.Tensor]


class ActivationProbe:
    """
    A utility class to extract the activation value of a specific layer or neuron within a neural network.

    The location of the target is defined using a string that can specify the layer, channel, and spatial coordinates (y, x). The format allows flexibility in defining the location:

    - 'layer1.0.conv1': Targets the entire layer.
    - 'layer1.0.conv1.3': Targets channel 3 in the specified layer.
    - 'layer1.0.conv1.3.2.2': Targets channel 3, y-coordinate 2, and x-coordinate 2 in the specified layer.
    - 'layer1.0.conv1.*': Targets all neurons in the specified layer.

    The class provides methods to register a forward hook into a PyTorch model to capture the activation of the specified target during model inference.

    Attributes:
        model: The PyTorch model from which to extract the activation.
        layer_location (List[str]): List of strings specifying the layer hierarchy.
        neuron_location (List[int]): List of integers specifying the channel, y, and x coordinates.
        activation: The value of the activation at the specified location.

    Example:
        model = ResNet18()
        extractor = ActivationProbe(model, 'layer1.0.conv1.3')
        handle = extractor.register_hook()
        output = model(input_tensor)
        print(extractor.activation)  # Prints the activation value

    The wildcard '*' in neuron_location means that all neurons in the specified layer will be targeted.
    For example, 'layer1.0.conv1.*' will capture activations for all neurons in the 'layer1.0.conv1' layer.
    """

    def __init__(self, model, location):
        self.activation = None
        self.model = model
        location = location.split(".")

        self.layer_location = []
        self.neuron_location = []

        # Get the target layer
        self.layer = model
        for part in location:
            if hasattr(self.layer, part):
                self.layer_location.append(part)
                self.layer = getattr(self.layer, part)
            else:
                if part == "*":
                    self.neuron_location.append(...)
                else:
                    self.neuron_location.append(int(part))

    def hook_fn(self, module, input, output):
        self.activation = output

        if self.neuron_location:
            # Assumes first index is over batch
            self.activation = self.activation[(..., *self.neuron_location)]

    def register_hook(self):
        self.handle = self.layer.register_forward_hook(self.hook_fn)
        return self.handle

    def unregister_hook(self):
        self.handle.remove()

    @contextmanager
    def watch(self):
        handle = self.register_hook()
        yield
        handle.remove()
