from contextlib import contextmanager
from typing import Callable, Optional, Tuple, Union

import torch
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm

from devinterp.misc.io import show_images
from devinterp.storage import VisualizationManager

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        y = self.y if self.y is not None else output.size(2) // 2
        x = self.x if self.x is not None else output.size(3) // 2

        self.activation = output[0, self.channel, y, x]

    def register_hook(self):
        handle = self.layer.register_forward_hook(self.hook_fn)
        return handle
    
    @contextmanager
    def watch(self):
        handle = self.register_hook()
        yield
        handle.remove()


class FeatureVisualizer:
    def __init__(self, model: torch.nn.Module, locations: Optional[list[str]]=None, storage: Optional[VisualizationManager] = None):
        self.model = model
        self.locations = locations or self.gen_locations(model)  # Defaults to all neurons in the model
        self.activations = [ActivationProbe(model, location) for location in self.locations]
        self.storage = storage

    def save_visualization(self, location: str, seed: int, step: int, data):
        if self.storage:
            file_id = (location, seed, step)
            self.storage.save_file(file_id, data)

    def load_visualization(self, location: str, seed: int, step: int):
        if self.storage:
            file_id = (location, seed, step)
            return self.storage.load_file(file_id)
        return None
    
    @staticmethod
    def gen_locations(model: torch.nn.Module, layer_type: Optional[Union[type, Tuple[type, ...]]] = None) -> list[str]:
        """Generate neurons of a particular kind of layer from a PyTorch model."""
        channel_locations = []

        def recursive_search(module, prefix):
            for name, submodule in module.named_children():
                path = prefix + '.' + name if prefix else name

                if not layer_type or isinstance(submodule, layer_type):

                    # TODO: Check that the submodule has a 'weight' attribute. 
                    # TODO: Do something about biases
                    for channel in range(submodule.out_channels):
                        location = f"{path}.weight.{channel}"
                        channel_locations.append(location)

                recursive_search(submodule, path)

        recursive_search(model, '')

        return channel_locations

    def render(self, probe: ActivationProbe, transform: Optional[Transform] = None, thresholds: list[int]=[512], verbose: bool = True, seed: int = 0, device: torch.device = DEVICE) -> list[torch.Tensor]:
        """Renders an image that maximizes the activation of the specified neuron.
        
        Args:
            transform (transforms.Compose, optional): Image transform to apply during optimization.
            thresholds (list[int], optional): List of iterations at which to save the optimized image.
            verbose (bool, optional): Whether to print progress information.
            seed (int, optional): Random seed for initialization of the input image.
            device (str, optional): Device on which to perform the computation.
            
        Returns:
            tuple[list[torch.Tensor], float]: A tuple containing the final images and the activation value.
        """

        # Assuming 'model' is your pre-trained ResNet model and 'location' is the string specifying the neuron's location
        self.model.to(device)
        self.model.eval()

        with probe.watch():
            # Create a random image (1x3x224x224) to start optimization, with same size as typical ResNet input
            torch.manual_seed(seed)
            input_image = torch.rand((1, 3, 32, 32), requires_grad=True, device=device)

            # Optimizer
            optimizer = torch.optim.Adam([input_image], lr=0.01, weight_decay=1e-3)

            final_images = []

            # Optimization loop
            pbar = range(max(thresholds) + 1)

            if verbose:
                pbar = tqdm(pbar, desc=f"Visualizing {probe.location} (activation: ???)")

            for iteration in pbar:
                optimizer.zero_grad()
                self.model(input_image)  # Forward pass through the model to trigger the hook
                activation = probe.activation
                loss = -activation  # Maximizing activation
                loss.backward()
                optimizer.step()

                if transform:
                    input_image.data = transform(input_image.data.detach().clone())

                if verbose:
                    pbar.set_description(f"Visualizing {probe.location} (activation: {activation.item():.2f})")

                if iteration in thresholds:
                    image = input_image.detach().clone()
                    image = torch.reshape(image, (1, 3, 32, 32))            
                    final_images.append(image)
        
        return final_images

    # def render(self, probe: ActivationProbe, num_visualizations: int, transform: Optional[Transform] = None, thresholds: list[int]=[512], verbose: bool = True, seed: int = 0, device: torch.device = DEVICE, diversity_weight: float = 0.1) -> list[list[torch.Tensor]]:
    #     """Renders multiple images that maximize the activation of the specified neuron, with a penalty to increase diversity.
        
    #     Args:
    #         num_visualizations (int): Number of visualizations to generate.
    #         ... other args same as render ...
    #         diversity_weight (float, optional): Weight of the diversity penalty.
        
    #     Returns:
    #         list[list[torch.Tensor]]: A list containing lists of final images for each visualization.
    #     """

    #     self.model.to(device)
    #     self.model.eval()
        
    #     with probe.watch():

    #         # Create random images (num_visualizations x 3 x 32 x 32) to start optimization
    #         torch.manual_seed(seed)
    #         input_images = torch.rand((num_visualizations, 3, 32, 32), requires_grad=True, device=device)

    #         # Optimizer
    #         optimizer = torch.optim.Adam([input_images], lr=0.01, weight_decay=1e-3)

    #         all_final_images = [[] for _ in range(num_visualizations)]

    #         pbar = range(max(thresholds) + 1)
    #         if verbose:
    #             pbar = tqdm(pbar, desc=f"Visualizing {probe.location} (activation: ???)")

    #         for iteration in pbar:
    #             optimizer.zero_grad()
    #             total_loss = torch.tensor(0.0, device=device)

    #             for i in range(num_visualizations):
    #                 self.model(input_images[i].unsqueeze(0))  # Forward pass through the model to trigger the hook
    #                 total_loss += -probe.activation  # Maximizing activation

    #                 # Calculate diversity penalty for the current image
    #                 for j in range(num_visualizations):
    #                     if i != j:
    #                         diversity_loss = F.cosine_similarity(input_images[i].view(1, -1), input_images[j].view(1, -1))
    #                         total_loss += diversity_weight * diversity_loss

    #             total_loss.backward()
    #             optimizer.step()

    #             if transform:
    #                 input_images.data = transform(input_images.data.detach().clone())

    #             if verbose:
    #                 pbar.set_description(f"Visualizing {probe.location} (activation: {-total_loss.item():.2f})")

    #             if iteration in thresholds:
    #                 for i in range(num_visualizations):
    #                     image = input_images[i].detach().clone()
    #                     image = torch.reshape(image, (3, 32, 32))
    #                     all_final_images[i].append(image)

    #     return all_final_images

    def render_all(self, thresholds: list[int]=[512], verbose: bool = True, init_seed: int = 0, device: str = "cuda", **kwargs) -> list[tuple[list[torch.Tensor], float]]:
        results = []

        for i, probe in enumerate(self.activations):
            images = self.render(
                probe,
                thresholds = thresholds,
                verbose = verbose,
                seed=init_seed + i,
                device=device
            )

            if verbose: 
                show_images(*images, **kwargs)

            results.append((images, probe.activation))

        return results
    
    def __len__(self) -> int:
        return len(self.activations)

    def __getitem__(self, idx: Union[int, str]) -> ActivationProbe:
        if isinstance(idx, int):
            return self.activations[idx]
        elif isinstance(idx, str):
            return next(filter(lambda probe: probe.location == idx, self.activations))
        else:
            raise TypeError(f"Invalid type for index: {type(idx)}")