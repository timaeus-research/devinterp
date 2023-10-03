# Copied from [openai's implementation](https://github.com/openai/grok/blob/main/grok/data.py)

from typing import Literal, Optional, Tuple

import torch
from pydantic import BaseModel
from torch import Tensor
from torch.utils.data import Dataset

from devinterp.zoo.arithmetic.utils import is_prime, modular_division, modular_exponentiation

DEFAULT_MODULUS = 97
DEFAULT_DATA_DIR = "data"

OperatorLiteral = Literal["/", "*", "+", "-", "^", "**"]


class ModularArithmeticConfig(BaseModel):
    operator: OperatorLiteral
    modulus: int
    seed: Optional[int] = None
    split: Optional[float] = None
    train: Optional[bool] = None

    class Config:
        frozen = True

    def factory(self):
        return ModularArithmetic.generate(self)

    def factory_split(self):
        return ModularArithmetic.generate_split(self)


class ModularArithmetic(Dataset):
    """A Dataset of modular arithmetic equations.

    Each example is a tuple of the form (i, j, k) where i, j, and k are
    integers representing the two operands and result.

    """

    def __init__(
        self, data: torch.Tensor, labels: torch.Tensor, config: ModularArithmeticConfig
    ) -> None:
        """
        :param train: if true, creates a training dataset, otherwise creates a validation dataset.

        """
        self.data = data
        self.targets = labels
        self.config = config

    @classmethod
    def generate(
        cls,
        config: ModularArithmeticConfig,
    ):
        """
        Generates a dataset of modular arithmetic equations.

        :param operator: the operator to use in the equations
        :param modulus: the modulus to use in the equations
        :param seed: the random seed to use
        :param shuffle: if true, shuffles the data
        :returns: a dataset of modular arithmetic equations
        """
        assert is_prime(config.modulus), f"{config.modulus} is not prime"
        assert (
            config.split is None and config.seed is None and config.train is None
        ), "Not implemented"

        def apply(i: int, j: int, operator: OperatorLiteral, modulus: int):
            if operator == "+":
                return (i + j) % modulus
            elif operator == "-":
                return (i - j) % modulus
            elif operator == "*":
                return (i * j) % modulus
            elif operator == "/":
                return modular_division(i, j, modulus)
            elif operator == "^" or operator == "**":
                return modular_exponentiation(i, j, modulus)
            else:
                raise ValueError(f"Unknown operator {operator}")

        data = torch.tensor(
            [(i, j, config.modulus) for i in range(config.modulus) for j in range(config.modulus)],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [apply(i, j, config.operator, config.modulus) for i, j in data[:, :2]],
            dtype=torch.long,
        )

        return cls(data=data, labels=labels, config=config)

    def split(self, split=0.8, seed: Optional[int] = None):
        """
        Splits the dataset into a training and validation dataset.

        This does not shuffle the data, so the first `frac_train` of the data
        will be used for training and the rest for validation.

        :param frac_train: fraction of data to use for training
        """
        train_len = int(len(self.data) * split)

        config_dict = self.config.model_dump()
        config_dict["split"] = split
        config_dict["seed"] = seed
        del config_dict["train"]

        train_config = ModularArithmeticConfig(**config_dict, train=True)
        test_config = ModularArithmeticConfig(**config_dict, train=False)

        if config_dict["seed"] is not None:
            torch.manual_seed(config_dict["seed"])

        train_indices, test_indices = torch.randperm(len(self.data)).split(
            [train_len, len(self.data) - train_len]
        )

        train_set, train_labels = self.data[train_indices], self.targets[train_indices]
        test_set, test_labels = self.data[test_indices], self.targets[test_indices]

        return (
            ModularArithmetic(train_set, train_labels, config=train_config),
            ModularArithmetic(test_set, test_labels, config=test_config),
        )

    @classmethod
    def generate_split(
        cls,
        config: ModularArithmeticConfig,
    ):
        """
        A convenience method to generate a modular arithmetic datset and
        split it into a training and validation dataset.

        See `generate` and `split` for more details.
        """
        assert config.split is not None and config.seed is not None, "Not implemented"
        config_dict = config.model_dump()

        del config_dict["split"]
        del config_dict["train"]
        del config_dict["seed"]

        raw_config = ModularArithmeticConfig(**config_dict)

        return ModularArithmetic.generate(raw_config).split(split=config.split, seed=config.seed)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        :param index: the index of the equation
        :returns: the equation at that index
        """
        return self.data[index], self.targets[index]

    def __iter__(self):
        return zip(self.data, self.targets)

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"ModularArithmetic({len(self)}, {self.config})"
