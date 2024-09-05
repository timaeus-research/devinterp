import torch
import warnings

import torch
import torchvision
from torch.nn import functional as F
from transformers import AutoModelForImageClassification
import numpy as np

from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import plot_trace, USE_TPU_BACKEND
import pytest

warnings.filterwarnings("ignore")

def evaluate(model, data):
    inputs, outputs = data

    return F.cross_entropy(model(inputs).logits, outputs), {
        "logits": model(inputs).logits
    }  # transformers doesn't output a vector


def get_stats(device, gpu_idxs = None, cores = 1, chains = 4, seed = None, num_workers = 1):
    # Load a pretrained MNIST classifier
    model = AutoModelForImageClassification.from_pretrained("fxmarty/resnet-tiny-mnist").to(
        device
    )
    data = torchvision.datasets.MNIST(
        root="../data",
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        ),
    )
    loader = torch.utils.data.DataLoader(data, batch_size=256, shuffle=True, num_workers=num_workers)

    return estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=4e-4, localization=100.0),
        num_chains=chains,  # How many independent chains to run
        num_draws=200,  # How many samples to draw per chain
        num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        device=device,
        online=True,
        cores=1,  # How many cores to use for parallelization
        gpu_idxs=None,  # Which GPUs to use ([0, 1] for using GPU 0 and 1)
        seed=seed,
    )

def distance(s1, s2):
    assert s1.keys() == s2.keys(), f"Expected the same keys in both stats, got {s1.keys()} and {s2.keys()}."
    assert s1["llc/trace"].shape == s2["llc/trace"].shape, f"Expected the same shape for llc/trace, got {s1['llc/trace'].shape} and {s2['llc/trace'].shape}."
    return np.mean((s1["llc/trace"] - s2["llc/trace"])**2)

@pytest.fixture(scope="module")
def cpu_default():
    return get_stats("cpu", seed = 100)

@pytest.fixture(scope="module")
def gpu_default():
    return get_stats("cuda", seed = 100)

def test_cpu_consistent(cpu_default):
    repeat_stats = get_stats("cpu", seed=100)
    assert distance(cpu_default, repeat_stats) < 2e-3, f"Repeat trials with the same seed on the CPU were not consistent. MSE: {distance(cpu_default, repeat_stats)}."

def test_cpu_consistent_seeds(cpu_default):
    diff_seed_stats = get_stats("cpu", seed=101)
    assert distance(cpu_default, diff_seed_stats) > 1, f"Trials with different seeds on the CPU were consistent. MSE: {distance(cpu_default, diff_seed_stats)}."

def test_cpu_multicore(cpu_default):
    multicore_stats = get_stats("cpu", seed=100, cores = 4)
    assert distance(cpu_default, multicore_stats) < 2e-3, f"Multicore trials with the same seed on the CPU were not consistent. MSE: {distance(cpu_default, multicore_stats)}."

def test_cpu_multiworker(cpu_default):
    multiworker_stats = get_stats("cpu", seed=100, num_workers = 4)
    assert distance(cpu_default, multiworker_stats) < 2e-3, f"Multiworker trials with the same seed on the CPU were not consistent. MSE: {distance(cpu_default, multiworker_stats)}."

def test_gpu_consistent(gpu_default):
    repeat_stats = get_stats("cuda", seed=100)
    assert distance(gpu_default, repeat_stats) < 2e-3, f"Repeat trials with the same seed on the GPU were not consistent. MSE: {distance(gpu_default, repeat_stats)}."

def test_gpu_consistent_seeds(gpu_default):
    diff_seed_stats = get_stats("cuda", seed=101)
    assert distance(gpu_default, diff_seed_stats) > 1, f"Trials with different seeds on the GPU were consistent. MSE: {distance(gpu_default, diff_seed_stats)}."

def test_gpu_multicore(gpu_default):
    multicore_stats = get_stats("cuda", seed=100, cores = 4)
    assert distance(gpu_default, multicore_stats) < 2e-3, f"Multicore trials with the same seed on the GPU were not consistent. MSE: {distance(gpu_default, multicore_stats)}."

def test_gpu_multiworker(gpu_default):
    multiworker_stats = get_stats("cuda", seed=100, num_workers = 4)
    assert distance(gpu_default, multiworker_stats) < 2e-3, f"Multiworker trials with the same seed on the GPU were not consistent. MSE: {distance(gpu_default, multiworker_stats)}."

def test_multigpu(gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_stats = get_stats("cuda", seed=100, gpu_idxs = [0, 1], cores = 2)
        assert distance(gpu_default, multigpu_stats) < 2e-3, f"Multigpu trials with the same seed on the GPU were not consistent. MSE: {distance(gpu_default, multigpu_stats)}."
        
def test_multigpu_multicore(gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_multicore_stats = get_stats("cuda", seed=100, gpu_idxs = [0, 1], cores = 4)
        assert distance(gpu_default, multigpu_multicore_stats) < 2e-3, f"Multigpu multicore trials with the same seed on the GPU were not consistent. MSE: {distance(gpu_default, multigpu_multicore_stats)}."