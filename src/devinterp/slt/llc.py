import os
from typing import Optional, Union

import torch

from devinterp.slt.callback import SamplerCallback


class LLCEstimator(SamplerCallback):
    r"""
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:

    $$LLC = \textrm{n \beta} * (\textrm{avg_loss} - \textrm{init_loss})$$

    For use with :func:`devinterp.slt.sampler.sample`.


    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Effective Inverse Temperature, float (default: 1., set by sample() to utils.default_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Supports GPUs and TPUs.
    :type device: str | torch.device, optional
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        init_loss: Union[float, torch.Tensor],
        device: Union[torch.device, str] = "cpu",
        eval_field: str = "loss",
        nbeta: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        self.device = device
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.losses: torch.Tensor = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)
        self.set_init_loss(init_loss)

        assert nbeta is not None, "Please provide a value for nbeta."
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        self.temperature = temperature

        self.eval_field = eval_field

    def set_init_loss(self, init_loss: Union[torch.Tensor, float]):
        self.init_loss = torch.as_tensor(
            init_loss, dtype=torch.float32, device=self.device
        )

    def update(self, chain: int, draw: int, loss: torch.Tensor):
        if torch.isnan(loss).any():
            raise RuntimeError(f"NaN detected in loss at chain {chain}, draw {draw}")
        self.losses[chain, draw] = loss.to(self.device)

    def finalize(self):
        if os.environ.get("USE_SPMD", "0") == "1" and not str(self.device).startswith(
            "cpu:"
        ):
            if str(self.device).startswith("cuda") and torch.cuda.device_count() > 1:
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(
                        self.losses, op=torch.distributed.ReduceOp.AVG
                    )
            else:
                pass

        elif str(
            self.device
        ).startswith(
            "cuda"
        ):  # if we've ran on multi-GPU, we should do a reduce as well. see above for how this would work
            try:
                torch.distributed.all_reduce(self.losses)
            except ValueError:
                pass
        avg_losses = self.losses.mean(axis=1)
        if (
            str(self.device).startswith("cuda")
            and os.environ.get("USE_SPMD", "0") == "1"
        ):
            self.llc_per_chain = self.nbeta.to(device="cpu", dtype=torch.float32) * (
                avg_losses.to(device="cpu", dtype=torch.float32)
                - self.init_loss.to(device="cpu", dtype=torch.float32)
            )
        else:
            self.llc_per_chain = self.nbeta * (avg_losses - self.init_loss)
        self.llc_mean = self.llc_per_chain.mean(dtype=torch.float32)
        self.llc_std = self.llc_per_chain.std()

    def get_results(self):
        """
        :returns: A dict :python:`{"llc/mean": llc_mean, "llc/std": llc_std, "llc-chain/{i}": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """

        return {
            "init_loss": self.init_loss.cpu().numpy().item(),
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{
                f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item()
                for i in range(self.num_chains)
            },
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        self.update(chain, draw, kwargs[self.eval_field])


class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process.
    It calculates LLCs using the same formula as :func:`devinterp.slt.llc.LLCEstimator`, but continuously and including means and std across draws (as opposed to just across chains).
    For use with :func:`devinterp.slt.sampler.sample`.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Effective Inverse Temperature, float (default: 1., set by sample() to utils.default_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Supports GPUs. Default is 'cpu'
    :type device: str | torch.device, optional
    """

    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        init_loss: Union[float, torch.Tensor],
        device="cpu",
        eval_field="loss",
        nbeta: Optional[float] = None,
        temperature: Optional[float] = None,  # Temperature is deprecated
    ):
        self.device = device
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.set_init_loss(init_loss)

        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(
            device
        )
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)

        self.losses = torch.zeros((num_chains, num_draws)).to(device)
        self.llcs = torch.zeros((num_chains, num_draws)).to(device)
        assert nbeta is not None, "Please provide a value for nbeta."
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        self.temperature = temperature

        self.eval_field = eval_field

    def set_init_loss(self, init_loss: Union[torch.Tensor, float]):
        self.init_loss = torch.as_tensor(
            init_loss, dtype=torch.float32, device=self.device
        )

    def update(self, chain: int, draw: int, loss: torch.Tensor):
        if torch.isnan(loss).any():
            raise RuntimeError(f"NaN detected in loss at chain {chain}, draw {draw}")
        loss = loss.to(self.device)
        self.losses[chain, draw] = loss
        self.llcs[chain, draw] = self.nbeta * (loss - self.init_loss)

    def finalize(self):
        # TODO
        self.llc_means = self.llcs.mean(dim=0, dtype=torch.float32)
        self.llc_stds = self.llcs.std(dim=0)

    def get_results(self):
        """
        :returns: A dict :python:`{"llc/means": llc_means, "llc/stds": llc_stds, "llc/trace": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "init_loss": self.init_loss.cpu().numpy(),
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        self.update(chain, draw, kwargs[self.eval_field])
