import warnings

import numpy as np
from typing import Union


class ChainCallback:
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
    
    def _validate_attribute_is_trace(self, attribute_name: str):
        if not self.finalized:
            raise RuntimeError("Cannot compute trace statistics before the callback is finalized.")
        if not hasattr(self, 'num_chains'):
            raise ValueError("Cannot compute trace statistics without num_chains.")
        if not hasattr(self, 'num_draws'):
            raise ValueError("Cannot compute trace statistics without num_draws.")
        if not hasattr(self.base_callback, self.attribute):
            raise ValueError(f"Base callback does not have attribute {self.attribute}")
        
        attribute = getattr(self, attribute_name)
        if not attribute.shape == (self.num_chains, self.num_draws):
            raise ValueError(f"Attribute {self.attribute} does not have shape {(self.num_chains, self.num_draws)}: {self.attribute} is not a trace.")

    def compute_trace_statistics(self, attribute_name: str):
        self._validate_attribute_is_trace(attribute_name)
        attribute = getattr(self, attribute_name)
        return {
            f'{attribute_name}/chain/mean': attribute.mean(axis=1),
            f'{attribute_name}/chain/std': attribute.std(axis=1),
            f'{attribute_name}/draw/mean': attribute.mean(axis=0),
            f'{attribute_name}/draw/std': attribute.std(axis=0),
        }
    
    def compute_trace_histograms(self, attribute_name: str, bins: int = None, chain_idx: Union[int, list] = None):
        self._validate_attribute_is_trace(attribute_name)
        attribute = getattr(self, attribute_name)
        if chain_idx is None:
            chain_idx = list(range(self.num_chains))
        elif type(chain_idx) == int:
            chain_idx = [chain_idx]
        
        return {
            f'{attribute_name}/chain/histogram/{i}': np.histogram(attribute[i], bins=bins) for i in chain_idx
        }

    def __call__(self, *args, **kwargs):
        raise NotImplementedError