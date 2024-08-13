from pprint import pp

import numpy as np
import pytest
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens.utils import lm_cross_entropy_loss, tokenize_and_concatenate
from transformers import AutoModelForCausalLM, AutoTokenizer

from devinterp.optim.sgld import SGLD
from devinterp.slt.llc import LLCEstimator
from devinterp.utils import USE_TPU_BACKEND, prepare_input, set_seed


def _test_hf(model, dataset, device: str):
    assert (
        USE_TPU_BACKEND
    ), "This test is intended to run using TPU, feel free to ignore failure if unavailable"

    set_seed(1)

    if device == "tpu":
        from devinterp.backends.tpu.slt.sampler import sample

        device = xm.xla_device()

    else:
        from devinterp.backends.default.slt.sampler import sample

    print(f"Testing on {device}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.to(device)
    model.eval()
    init_loss = torch.zeros(1).to(device)

    def evaluate(model, batch):
        logits = model(batch["tokens"]).logits
        return lm_cross_entropy_loss(logits, batch["tokens"]), {"logits": logits}

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=4):
            batch = prepare_input(
                batch, device, is_deepspeed_enabled=False, accelerator=None
            )

            init_loss += evaluate(model, batch)[0]

            if i >= 4:
                break

    init_loss /= 4
    init_loss = init_loss.detach()

    print("\n\nInit loss", init_loss)

    # model = torch.compile(model)

    nbeta = 20.0
    num_chains = 1
    num_draws = 50
    batch_size = 16

    llc_estimator = LLCEstimator(
        num_chains=num_chains,
        num_draws=num_draws,
        nbeta=nbeta,
        device=device,
        init_loss=init_loss,
    )

    # Run the LLC estimation
    metrics = sample(
        model,
        loader,
        callbacks=[llc_estimator],
        evaluate=evaluate,
        sampling_method=SGLD,
        optimizer_kwargs=dict(
            lr=0.001,
            noise_level=10.0,
            weight_decay=0.0,
            localization=0.0,
            nbeta=nbeta,
            save_noise=False,
            save_mala_vars=False,
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=0,
        num_steps_bw_draws=1,
        seed=42,
        device=device,
        verbose=True,
        batch_size=batch_size,
        init_loss=init_loss,
    )

    return metrics


@pytest.mark.tpu
@pytest.mark.slow
def test_hf():
    import torch_xla.core.xla_model as xm

    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

    # count_parameters(model)
    print(tokenizer)

    # Load the dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = tokenize_and_concatenate(dataset, tokenizer)

    # Set up the LLC estimator

    metrics_tpu = _test_hf(model, dataset, "tpu")
    pp(metrics_tpu)
    metrics_cpu = _test_hf(model, dataset, "cpu")
    pp(metrics_cpu)
    metrics_cpu.pop("llc/std")  # 1 chain only
    metrics_cpu.pop("loss/trace")  # 1 chain only
    for k, v in metrics_cpu.items():
        if isinstance(v, torch.Tensor):
            assert torch.allclose(
                v, metrics_tpu[k], rtol=5e-3
            ), f"Evaluation failed for {k}"
        elif isinstance(v, np.ndarray):
            assert np.isclose(
                v, metrics_tpu[k], rtol=5e-3
            ).all(), f"Evaluation failed for {k}"
        else:
            assert np.isclose(
                v, metrics_tpu[k], rtol=5e-3
            ), f"Evaluation failed for {k}"
