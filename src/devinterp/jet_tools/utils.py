from typing import Any, Mapping, Union

import torch


def prepare_input(
    data: Union[torch.Tensor, Any],
    device,
    is_deepspeed_enabled=False,
    accelerator=None,
) -> Union[torch.Tensor, Any]:
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.

    Adapted from HuggingFace's transformers's Trainer._prepare_input().
    """
    if isinstance(data, Mapping):
        return type(data)(
            {
                k: prepare_input(
                    v,
                    device=device,
                    is_deepspeed_enabled=is_deepspeed_enabled,
                    accelerator=accelerator,
                )
                for k, v in data.items()
            }
        )
    elif isinstance(data, (tuple, list)):
        return type(data)(
            prepare_input(
                v,
                device=device,
                is_deepspeed_enabled=is_deepspeed_enabled,
                accelerator=accelerator,
            )
            for v in data
        )
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        if is_deepspeed_enabled and (
            torch.is_floating_point(data) or torch.is_complex(data)
        ):
            # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            # embedding. Other models such as wav2vec2's inputs are already float and thus
            # may need special handling to match the dtypes of the model
            kwargs.update(
                {"dtype": accelerator.state.deepspeed_plugin.hf_ds_config.dtype()}
            )
        return data.to(**kwargs)
    return data
