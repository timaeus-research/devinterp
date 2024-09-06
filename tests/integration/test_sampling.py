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
        num_draws=10,  # How many samples to draw per chain
        num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        device=device,
        online=True,
        cores=1,  # How many cores to use for parallelization
        gpu_idxs=None,  # Which GPUs to use ([0, 1] for using GPU 0 and 1)
        seed=seed,
    )

def check(s1, s2, rtol=1e-3, reverse = False):
    """
    Check if two stats are close to each other.
    
    """
    assert s1.keys() == s2.keys(), f"Expected the same keys in both stats, got {s1.keys()} and {s2.keys()}."
    assert s1["llc/trace"].shape == s2["llc/trace"].shape, f"Expected the same shape for llc/trace, got {s1['llc/trace'].shape} and {s2['llc/trace'].shape}."
    valid = np.allclose(s1["llc/trace"], s2["llc/trace"], rtol=rtol)
    if reverse:
        valid = not valid
    assert valid, f"Expected {'different' if reverse else 'close'} llc/trace in both stats, got {s1['llc/trace']} and {s2['llc/trace']}."

@pytest.fixture(scope="module")
def cpu_default():
    return get_stats("cpu", seed = 100)

@pytest.mark.gpu
@pytest.fixture(scope="module")
def gpu_default():
    return get_stats("cuda", seed = 100)

def test_cpu_consistent(cpu_default):
    repeat_stats = get_stats("cpu", seed=100)
    check(cpu_default, repeat_stats, 1e-3)

def test_cpu_consistent_seeds(cpu_default):
    diff_seed_stats = get_stats("cpu", seed=101)
    check(cpu_default, diff_seed_stats, 0.1, reverse = True)

def test_cpu_multicore(cpu_default):
    multicore_stats = get_stats("cpu", seed=100, cores = 4)
    check(cpu_default, multicore_stats, 1e-4)

def test_cpu_multiworker(cpu_default):
    multiworker_stats = get_stats("cpu", seed=100, num_workers = 4)
    check(cpu_default, multiworker_stats, 1e-4)

@pytest.mark.gpu
def test_gpu_consistent(gpu_default):
    repeat_stats = get_stats("cuda", seed=100)
    check(gpu_default, repeat_stats, 5e-3)

@pytest.mark.gpu
def test_gpu_consistent_seeds(gpu_default):
    diff_seed_stats = get_stats("cuda", seed=101)
    check(gpu_default, diff_seed_stats, 0.1, reverse = True)

@pytest.mark.gpu
def test_gpu_multicore(gpu_default):
    multicore_stats = get_stats("cuda", seed=100, cores = 4)
    check(gpu_default, multicore_stats, 5e-3)

@pytest.mark.gpu
def test_gpu_multiworker(gpu_default):
    multiworker_stats = get_stats("cuda", seed=100, num_workers = 4)
    check(gpu_default, multiworker_stats, 5e-3)

@pytest.mark.gpu
def test_multigpu(gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_stats = get_stats("cuda", seed=100, gpu_idxs = [0, 1], cores = 2)
        check(gpu_default, multigpu_stats, 5e-3)
    else:
        pytest.skip("Multiple GPUs unavailable.")

@pytest.mark.gpu
def test_multigpu_multicore(gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_multicore_stats = get_stats("cuda", seed=100, gpu_idxs = [0, 1], cores = 4)
        check(gpu_default, multigpu_multicore_stats, 5e-3)
    else:
        pytest.skip("Multiple GPUs unavailable.")