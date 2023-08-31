from typing import Dict, List, Literal, Set, Tuple, TypeAlias, Union

from devinterp.utils import int_linspace, int_logspace

StepsConfig: TypeAlias = Dict[
    Literal["log_space", "linear_space"], Tuple[int, int, int]
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
