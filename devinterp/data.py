import torch


class CustomDataloader(torch.utils.data.DataLoader):
    def __init__(self, data: torch._utils.data.DataSet, *args, **kwargs):
        self.generator = torch.Generator(device="cpu")
        sampler = torch.utils.data.RandomSampler(data, generator=self.generator)
        kwargs.update({"sampler": self.sampler})
        super().__init__(data, *args, **kwargs)

    def set_seed(self, seed: int):
        self.generator.manual_seed(seed)
