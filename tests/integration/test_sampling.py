import time
import warnings

import numpy as np
import pytest
import torch
from datasets import load_dataset
from devinterp.optim import SGLD
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.utils import USE_TPU_BACKEND, plot_trace
from torch.nn import functional as F
from transformers import AutoModelForImageClassification

warnings.filterwarnings("ignore")


def evaluate(model, data):
    inputs, outputs = data

    return F.cross_entropy(model(inputs).logits, outputs), {
        "logits": model(inputs).logits
    }  # transformers doesn't output a vector


class torchvisionWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return item["pixel_values"], item["label"]


@pytest.fixture(scope="module")
def data():

    mnist_dataset = load_dataset("mnist")

    def preprocess(examples):
        # Convert images to tensors and normalize
        examples["pixel_values"] = [
            torch.tensor(np.array(img)).float().unsqueeze(0) / 255.0
            for img in examples["image"]
        ]
        return examples

    mnist_dataset = mnist_dataset.map(
        preprocess, batched=True, remove_columns=["image"]
    )
    mnist_dataset.set_format(type="torch", columns=["pixel_values", "label"])

    return torchvisionWrapper(mnist_dataset["train"])


def get_stats(
    data,
    device,
    gpu_idxs=None,
    cores=1,
    chains=2,
    seed=None,
    num_workers=0,
    batch_size=64,
    grad_accum_steps=1,
    use_amp=False,
):
    # Load a pretrained MNIST classifier
    model = AutoModelForImageClassification.from_pretrained(
        "fxmarty/resnet-tiny-mnist"
    ).to(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    # Set the format of the dataset to PyTorch tensors

    torch.manual_seed(seed)
    np.random.seed(seed)
    return estimate_learning_coeff_with_summary(
        model,
        loader=loader,
        evaluate=evaluate,
        sampling_method=SGLD,
        optimizer_kwargs=dict(lr=4e-4, localization=100.0, nbeta=2.0),
        num_chains=chains,  # How many independent chains to run
        num_draws=10,  # How many samples to draw per chain
        num_burnin_steps=0,  # How many samples to discard at the beginning of each chain
        num_steps_bw_draws=1,  # How many steps to take between each sample
        device=device,
        online=True,
        cores=cores,  # How many cores to use for parallelization
        gpu_idxs=gpu_idxs,  # Which GPUs to use ([0, 1] for using GPU 0 and 1)
        seed=seed,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        init_loss=0.1,
    )


def check(s1, s2, atol=1e-3, reverse=False):
    """
    Check if two stats are close to each other.

    """
    assert (
        s1.keys() == s2.keys()
    ), f"Expected the same keys in both stats, got {s1.keys()} and {s2.keys()}."
    assert (
        s1["llc/trace"].shape == s2["llc/trace"].shape
    ), f"Expected the same shape for llc/trace, got {s1['llc/trace'].shape} and {s2['llc/trace'].shape}."
    valid = np.allclose(s1["llc/trace"], s2["llc/trace"], atol=atol)
    if reverse:
        valid = not valid
    assert (
        valid
    ), f"Expected {'different' if reverse else 'close'} llc/trace in both stats, got {s1['llc/trace']} and {s2['llc/trace']}, {np.isclose(s1['llc/trace'], s2['llc/trace'], atol=atol)}."


@pytest.fixture(scope="module")
def cpu_default(data):
    return get_stats(data, "cpu", seed=100)


@pytest.mark.gpu
@pytest.fixture(scope="module")
def gpu_default(data):
    return get_stats(data, "cuda", seed=100)


def test_cpu_consistent(data, cpu_default):
    repeat_stats = get_stats(data, "cpu", seed=100)
    check(cpu_default, repeat_stats, 1e-3)


def test_cpu_consistent_seeds(data, cpu_default):
    diff_seed_stats = get_stats(data, "cpu", seed=101)
    check(cpu_default, diff_seed_stats, 0.000001, reverse=True)


@pytest.mark.slow
def test_cpu_multicore(data, cpu_default):
    multicore_stats = get_stats(data, "cpu", seed=100, cores=2)
    check(cpu_default, multicore_stats, 1e-4)


def test_grad_accum(data, cpu_default: dict):
    grad_accum_stats = get_stats(
        data, "cpu", seed=100, grad_accum_steps=4, batch_size=16
    )
    check(cpu_default, grad_accum_stats, 1)


@pytest.mark.gpu
def test_gpu_consistent(data, gpu_default):
    repeat_stats = get_stats(data, "cuda", seed=100)
    check(gpu_default, repeat_stats, 0.2)


@pytest.mark.gpu
def test_gpu_consistent_seeds(data, gpu_default):
    diff_seed_stats = get_stats(data, "cuda", seed=101)
    check(gpu_default, diff_seed_stats, 5, reverse=True)


@pytest.mark.gpu
def test_gpu_multicore(data, gpu_default):
    multicore_stats = get_stats(data, "cuda", seed=100, cores=4)
    check(gpu_default, multicore_stats, 0.2)


@pytest.mark.gpu
def test_gpu_multiworker(data, gpu_default):
    multiworker_stats = get_stats(data, "cuda", seed=100, num_workers=4)
    check(gpu_default, multiworker_stats, 0.2)


@pytest.mark.gpu
def test_multigpu(data, gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_stats = get_stats(data, "cuda", seed=100, gpu_idxs=[0, 1], cores=2)
        check(gpu_default, multigpu_stats, 0.2)
    else:
        pytest.skip("Multiple GPUs unavailable.")


@pytest.mark.gpu
def test_multigpu_multicore(data, gpu_default):
    if torch.cuda.device_count() > 1:
        multigpu_multicore_stats = get_stats(
            data, "cuda", seed=100, gpu_idxs=[0, 1], cores=4
        )
        check(gpu_default, multigpu_multicore_stats, 0.4)
    else:
        pytest.skip("Multiple GPUs unavailable.")


@pytest.mark.gpu
def test_gpu_grad_accum(data, gpu_default: dict):
    grad_accum_stats = get_stats(
        data, "cuda", seed=100, cores=4, grad_accum_steps=2, batch_size=128
    )
    check(gpu_default, grad_accum_stats, 1)


@pytest.mark.gpu
def test_gpu_amp(data, gpu_default: dict):
    amp_stats = get_stats(data, "cuda", seed=100, cores=4, use_amp=True)
    check(gpu_default, amp_stats, 0.2)
