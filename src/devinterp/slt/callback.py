import warnings


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