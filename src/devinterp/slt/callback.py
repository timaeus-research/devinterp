import warnings
from typing import Callable, List


class SamplerCallback:
    def __init__(self, device="cpu"):
        self.device = device
    
    def share_memory_(self):
        if self.device == "mps":
            warnings.warn("Cannot share memory with MPS device.")
            return self

        for attr in dir(self):
            if attr.startswith("_"):
                continue
            
            attr = getattr(self, attr)
            if hasattr(attr, "share_memory_"):
                attr.share_memory_()

        return self
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError
    

def validate_callbacks(callbacks: List[Callable]):
    for i, callback in enumerate(callbacks):
        if isinstance(callback, SamplerCallback) and hasattr(callback, 'base_callback'):
            base_callback = callback.base_callback
            base_callback_exists = False
            for j, other_callback in enumerate(callbacks):
                if other_callback is base_callback:
                    if j > i:
                        raise ValueError(f"Derivative callback {callback} must be called after base callback {base_callback}.")
                    base_callback_exists = True
            if not base_callback_exists:
                raise ValueError(f"Base callback {base_callback} of derivative callback {callback} was not passed.")
    
    return True

