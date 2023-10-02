from typing import Union

import torch
from torch.utils.data import Dataset

from devinterp.zoo.dlns.model import DLN


class DLNDataset(Dataset):
    teacher: DLN

    def __init__(
        self,
        teacher: Union[DLN, torch.Tensor],
        num_samples: int = 100,
        noise_level: float = 0.0,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        torch.manual_seed(seed)

        if isinstance(teacher, torch.Tensor):
            teacher = DLN.from_matrix(teacher)

        self.teacher = teacher.to(device=device)
        self.num_features = teacher.to_matrix().shape[0]
        self.num_samples = num_samples
        self.noise_level = noise_level

        inputs = torch.rand(self.num_samples, self.num_features, device=device).detach()

        num_outputs = self.teacher.to_matrix().shape[1]
        labels = (teacher(inputs) + noise_level * torch.rand(num_outputs)).detach()
        self.data = torch.utils.data.TensorDataset(inputs, labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f"DLNDataset(teacher={self.teacher}, num_samples={self.num_samples}, noise_level={self.noise_level})"

    @classmethod
    def generate_split(
        cls,
        teacher: Union[DLN, torch.Tensor],
        num_samples: int = 100,
        noise_level: float = 0.0,
        seed: int = 0,
        device: torch.device = torch.device("cpu"),
    ):
        if isinstance(teacher, torch.Tensor):
            teacher = DLN.from_matrix(teacher)

        train_data = cls(
            teacher,
            num_samples=num_samples,
            noise_level=noise_level,
            seed=seed,
            device=device,
        )
        test_data = cls(
            teacher,
            num_samples=num_samples,
            noise_level=noise_level,
            seed=seed + 1,
            device=device,
        )

        return train_data, test_data
