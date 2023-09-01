from typing import Dict, List, Literal, Set, Tuple, Union

from devinterp.utils import int_linspace, int_logspace

StepsConfig = Dict[
    Literal["log_space", "linear_space"], Tuple[int, int, int]
]

StepsConfigShortened = Dict[
    Literal["log_space", "linear_space"], int
]

def process_steps(config: Union[List[int], Tuple[int], Set[int], StepsConfig]):
    if isinstance(config, dict):
        result = set()
        log_args = config.get("log_space")
        lin_args = config.get("linear_space")

        if log_args is not None:
            result |= int_logspace(*log_args, return_type="set")

        if lin_args is not None:
            result |= int_linspace(*lin_args, return_type="set")

        return result
    elif isinstance(config, (list, tuple, set)):
        return set(config)
    else:
        raise ValueError(f"Invalid steps config: {config}")


def expand_steps_config_(config: Union[StepsConfigShortened, StepsConfig], num_steps: int) -> StepsConfig:
    if isinstance(config.get("log_space", None), int):
        config["log_space"] = [1, num_steps, config["log_space"]]
    if isinstance(config.get("linear_space", None), int):
        config["linear_space"] = [0, num_steps, config["linear_space"]]
