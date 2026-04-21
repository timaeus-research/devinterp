"""Configuration for SGLD sampling."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)

from devinterp.optim import SGMCMC

EpochMode = Literal["once", "cycle"]

SamplingMethodLiteral = Literal[
    "sgmcmc_sgld",
    "rmsprop_sgld",
]

SAMPLING_METHODS = {
    "sgmcmc_sgld": SGMCMC.sgld,
    "rmsprop_sgld": SGMCMC.rmsprop_sgld,
}


class SamplerConfig(BaseModel):
    """Configuration for SGLD sampling.

    Validates types and value ranges for all sampler parameters.
    """

    model_config = ConfigDict(extra="forbid")

    lr: NonNegativeFloat
    n_beta: NonNegativeFloat

    batch_size: PositiveInt = 32
    num_chains: PositiveInt = 4
    num_draws: PositiveInt = 200
    num_burnin_steps: NonNegativeInt = 0
    num_steps_bw_draws: PositiveInt = 1
    gradient_accumulation_steps: PositiveInt = 1
    num_init_loss_batches: PositiveInt = 32

    init_seed: int = 100

    # Optimizer parameters
    localization: NonNegativeFloat = 0.0
    noise_level: NonNegativeFloat = 1.0
    llc_weight_decay: NonNegativeFloat = 0.0
    bounding_box_size: NonNegativeFloat | None = None

    sampling_method: SamplingMethodLiteral = "sgmcmc_sgld"
    sampling_method_kwargs: dict[str, Any] = Field(default_factory=dict)

    init_noise: NonNegativeFloat | None = None
    save_metrics: bool = False
    shuffle: bool = True
    epoch_mode: EpochMode = "cycle"
    match_sampling_input_ids_across_chains: bool = True

    @property
    def num_sgld_steps(self) -> int:
        return self.num_draws * self.num_steps_bw_draws + self.num_burnin_steps
