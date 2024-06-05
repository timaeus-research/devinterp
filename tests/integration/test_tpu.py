from pprint import pp
from typing import Dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_dataset
from transformer_lens import HookedTransformerConfig
from transformer_lens.utils import tokenize_and_concatenate, lm_cross_entropy_loss

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from datasets import load_dataset

from devinterp.utils import optimal_temperature, set_seed, prepare_input
from devinterp.optim.sgld import SGLD


def _test_hf(model, dataset, device: str):
    set_seed(1)

    if device == "tpu":
        from devinterp.backends.tpu.slt.llc import LLCEstimator
        from devinterp.backends.tpu.slt.sampler import sample

        device = xm.xla_device()

    else:
        from devinterp.backends.default.slt.llc import LLCEstimator
        from devinterp.backends.default.slt.sampler import sample

    print(f"Testing on {device}")

    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    model.to(device)
    model.eval()
    init_loss = torch.zeros(1).to(device)

    evaluate = lambda model, batch: lm_cross_entropy_loss(
        model(batch["tokens"]).logits, batch["tokens"]
    )

    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader), total=4):
            batch = prepare_input(
                batch, device, is_deepspeed_enabled=False, accelerator=None
            )

            init_loss += evaluate(model, batch)

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
    )

    # Run the LLC estimation
    metrics = sample(
        model,
        dataset,
        callbacks=[llc_estimator],
        evaluate=evaluate,
        sampling_method=SGLD,
        optimizer_kwargs=dict(
            lr=0.001,
            noise_level=10.0,
            weight_decay=0.0,
            localization=0.0,
            temperature=nbeta,
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


def test_hf():
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-1M")
    tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-1M")

    # count_parameters(model)
    print(tokenizer)

    # Load the dataset
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    dataset = tokenize_and_concatenate(dataset, tokenizer)

    # Set up the LLC estimator

    metrics_cpu = _test_hf(model, dataset, "cpu")
    pp(metrics_cpu)

    metrics_tpu = _test_hf(model, dataset, "tpu")
    pp(metrics_tpu)

    for k, v in metrics_cpu.items():
        if isinstance(v, torch.Tensor):
            assert torch.allclose(
                v, metrics_tpu[k], atol=1e-4
            ), f"Evaluation failed for {k}"
        else:
            assert v == metrics_tpu[k], f"Evaluation failed for {k}"
