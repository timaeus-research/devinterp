import textwrap
from typing import Iterator, List, Tuple, Union

import torch
from torch import nn
from torch.nn.modules.module import Module


def prepend_dict(d: dict, prefix: str, delimiter="."):
    """Utility function to prepend a prefix to the keys of a dictionary."""

    def get_key(key):
        if key == "":
            return prefix

        return f"{prefix}{delimiter}{key}"

    return {get_key(k): v for k, v in d.items()}


def hook(module: nn.Module, *paths: str):
    """Recursively wraps PyTorch modules in Hooked, HookedList, or HookedSequential classes.

    This exposes a `run_with_cache()` method inspired by TransformerLens's HookedTransformer.
    Unlike TransformerLens, this works with any PyTorch module right out of the box.

    Args:
        module (nn.Module): The module to be wrapped.
        paths (str): Paths to the children modules to be wrapped and whose activations will be returned.
            If empty, all children will be wrapped and all internal activations will be returned.

    Examples:

    Returns:
        Hooked, HookedList, or HookedSequential: Wrapped module.
    """
    if len(paths) == 0:  # Default to hooking all children
        return _hook(module)

    module = Hooked(module)
    for path in paths:
        components = path.split(".")
        _hook_recursive(module, components)

    return module


def _hook(module: nn.Module):
    """Recursively wraps PyTorch modules in Hooked, HookedList, or HookedSequential classes."""
    if isinstance(module, (Hooked, HookedList, HookedSequential)):
        for n, c in module.named_children():
            setattr(module, n, _hook(c))
        return module
    if isinstance(module, nn.ModuleList):
        return HookedList([_hook(c) for c in module])
    elif isinstance(module, nn.Sequential):
        return HookedSequential(*[_hook(c) for c in module.children()])
    else:
        module = Hooked(module)

        for n, c in module.named_children():
            setattr(module, n, _hook(c))

        return module


def _hook_recursive(module: nn.Module, components: List[str]):
    # Base case
    if len(components) == 0:
        if isinstance(module, (Hooked, HookedList, HookedSequential)):
            return module
        elif isinstance(module, nn.ModuleList):
            return HookedList(module)
        elif isinstance(module, nn.Sequential):
            return HookedSequential(*module.children())
        else:
            return Hooked(module)

    # Recursive case
    head, *tail = components

    if not isinstance(module, (Hooked, HookedList, HookedSequential)):
        if isinstance(module, (nn.ModuleList, nn.Sequential)):
            children = list(module.children())

            if head == "*":  # Allow wildcards to hook all children
                children = [_hook_recursive(c, tail) for c in children]
            else:
                head = int(head)
                children[head] = _hook_recursive(children[head], tail)

            if isinstance(module, nn.ModuleList):
                module = HookedList(children)
            elif isinstance(module, nn.Sequential):
                module = HookedSequential(*children)

            return module
        else:
            module = Hooked(module)

    # If the current module is already Hooked, we may still need to further hook its children.
    next_module = _hook_recursive(getattr(module, head), tail)
    setattr(module, head, next_module)

    return module


class Hooked(nn.Module):
    def __init__(self, module: nn.Module):
        """Wraps a PyTorch module to cache its output during forward pass.

        Attributes:
            _module (nn.Module): Original module.
            _forward (Callable): Original forward method.
            output: Cached output of the last forward pass.
        """
        super().__init__()
        self._module = module
        self._forward = type(module).forward
        self.output = None

    def collect_cache(self):
        """Collects cached outputs from the current module and its immediate children.

        Returns:
            dict: Nested dictionary of cached outputs.
        """
        cache = {"": self.output}

        for n, c in self.named_children():
            if isinstance(c, (Hooked, HookedList, HookedSequential)):
                cache.update(prepend_dict(c.collect_cache(), n))

        return cache

    def forward(self, *args, **kwargs):
        """Runs the forward pass and caches the output."""
        self.output = self._forward(self, *args, **kwargs)
        return self.output

    def run_with_cache(self, *args, **kwargs):
        """Runs the forward pass and collects cached outputs.

        Inspired by TransformerLens's HookedTransformer.

        Returns:
            tuple: Output of forward pass and the cache.
        """
        output = self.forward(*args, **kwargs)
        cache = self.collect_cache()
        del cache[""]

        return output, cache

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._module, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        """Returns the state dictionary of the original module."""
        return self._module.state_dict(destination, prefix, keep_vars)

    def load_state_dict(self, state_dict, strict=True):
        """Loads the state dictionary into the original module."""
        return self._module.load_state_dict(state_dict, strict)

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        """Returns an iterator over immediate children modules, yielding both the name of the module as well as the module itself."""
        for n, c_original in self._module.named_children():
            if hasattr(self, n):
                yield n, getattr(self, n)
            else:
                yield n, c_original

    def __repr__(self):
        """Returns a string representation of the module."""
        child_str_list = []
        children = list(self.named_children())

        for n, c in children:
            if n != "_module":
                child_repr = repr(c)
                child_str_list.append(f"{n}: {child_repr}")

        child_str = ",\n".join(child_str_list)

        indented_child_str = textwrap.indent(child_str, "  ")
        mod_str = self._module.__class__.__name__

        if len(children) == 0:
            return f"Hooked({repr(self._module)})"

        return f"Hooked({mod_str}(\n{indented_child_str}\n))"


class HookedList(nn.ModuleList):
    """Extension of PyTorch's ModuleList to support caching of outputs."""

    def collect_cache(self):
        """Collects cached outputs from the list of modules.

        Returns:
            dict: Nested dictionary of cached outputs.
        """
        cache = {}

        for i, c in enumerate(self):
            if isinstance(c, (Hooked, HookedList, HookedSequential)):
                cache.update(prepend_dict(c.collect_cache(), str(i)))

        return cache


class HookedSequential(nn.Sequential):
    """Extension of PyTorch's Sequential to support caching of outputs."""

    def collect_cache(self):
        """Collects cached outputs from the sequence of modules.

        Returns:
            dict: Nested dictionary of cached outputs.
        """
        cache = {}

        for i, c in enumerate(self):
            if isinstance(c, (Hooked, HookedList, HookedSequential)):
                cache.update(prepend_dict(c.collect_cache(), str(i)))

        return cache

    def run_with_cache(self, *args, **kwargs):
        """Runs the forward pass and collects cached outputs.

        Inspired by TransformerLens's HookedTransformer.

        Returns:
            tuple: Output of forward pass and the cache.
        """
        output = self.forward(*args, **kwargs)
        return output, self.collect_cache()
