from typing import Union, Optional
import torch
from devinterp.slt.callback import SamplerCallback
import warnings

# TPU support
try:
    import torch_xla.core.xla_model as xm
    USING_XLA = True
except ImportError:
    USING_XLA = False

class LLCEstimator(SamplerCallback):
    r"""
    Callback for estimating the Local Learning Coefficient (LLC) in a rolling fashion during a sampling process.
    It calculates the LLC based on the average loss across draws for each chain:
    
    $$LLC = \textrm{T} * (\textrm{avg_loss} - \textrm{init_loss})$$

    For use with :func:`devinterp.slt.sampler.sample`.
    
    Note: 
        - `init_loss` gets set inside :func:`devinterp.slt.get_results()`. It can be passed as an argument to that function, 
        and if not passed will be the average loss of the supplied model over `num_chains` batches.

    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Reparameterized inverse temperature
    :type nbeta: float
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'.
    :type device: str | torch.device, optional
    """
    def __init__(
        self,
        num_chains: int,
        num_draws: int,
        init_loss: torch.Tensor,
        eval_field: str = "loss",
        device: Union[torch.device, str] = "cpu",
        nbeta: float = None,
        temperature: Optional[float] = None, # Temperature is deprecated
    ):
        self.num_chains = num_chains
        self.num_draws = num_draws

        self.loss = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.init_loss = init_loss

        assert nbeta is not None or temperature is not None, "Please provide a value for nbeta."
        if nbeta is None and temperature is not None:
            nbeta = temperature
            warnings.warn("Temperature is deprecated. Please use nbeta instead.")
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        self.eval_field = eval_field
        self.device = device
        self.count = 0.0

    def update(self, chain: int, draw: int, loss: torch.Tensor, **kwargs):
        """
        Called at each step of sampling.

        Order of operations: 
        1. update() is called at each draw.
        2. finalize() is called at the end of sampling.
        3. get_results() is called after finalize().

        :param draw: The current draw index.
        :type draw: int
        :param loss: The loss at the current draw.
        :type loss: float
        :param chain: The chain index. Note that this is not used here at all.
        :type chain: int
        """
        self.loss[chain, draw] = loss
        self.count += 1.0

    def finalize(self):
        """Called once at the end of sampling."""

        # If using TPUs, aggregate the loss across all devices.
        if USING_XLA:
            if self.count == 0:
                self.count = 1
            scale = 1.0 / (self.count * xm.xrt_world_size())
            self.loss = xm.all_reduce(xm.REDUCE_SUM, self.loss, scale=scale)

        avg_loss = self.loss.mean(axis=1) # [num_chains]. Average loss across draws for each chain.

        self.llc_per_chain = self.nbeta * (avg_loss - self.init_loss) # [num_chains]. A single LLC value for each chain.
        self.sq_loss = torch.square(self.loss)
        self.llc_mean = self.llc_per_chain.mean() # Mean LLC across chains.
        self.llc_std = self.llc_per_chain.std() # Standard deviation of LLC across chains.

    def get_results(self):
        """
        Method to return results. Called after finalize().

        :returns: A dict :python:`{"llc/mean": llc_mean, "llc/std": llc_std, "llc-chain/{i}": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "llc/mean": self.llc_mean.cpu().numpy().item(),
            "llc/std": self.llc_std.cpu().numpy().item(),
            **{
                f"llc-chain/{i}": self.llc_per_chain[i].cpu().numpy().item()
                for i in range(self.num_chains)
            },
            "loss/trace": self.loss.cpu().numpy(),
            "loss/init_loss": self.init_loss,
            "loss/sq_loss": self.sq_loss.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        """Calls self.update."""
        self.update(chain, draw, kwargs[self.eval_field])


class OnlineLLCEstimator(SamplerCallback):
    """
    Callback for estimating the Local Learning Coefficient (LLC) in an online fashion during a sampling process.
    It calculates LLCs using the same formula as :func:`devinterp.slt.llc.LLCEstimator`, but continuously and including means and std across draws (as opposed to just across chains).
    For use with :func:`devinterp.slt.sampler.sample`.
    
    Note: 
        - `init_loss` gets set inside :func:`devinterp.slt.get_results()`. It should be passed as an argument to that function.
    :param num_draws: Number of samples to draw (should be identical to :python:`num_draws` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_draws: int
    :param num_chains: Number of chains to run (should be identical to :python:`num_chains` passed to :python:`devinterp.slt.sampler.sample`)
    :type num_chains: int
    :param nbeta: Inverse reparameterized temperature
    :type nbeta: float
    :param device: Device to perform computations on, e.g., 'cpu' or 'cuda'. Default is 'cpu'
    :type device: str | torch.device, optional
    """

    def __init__(
        self, 
        num_chains: int, 
        num_draws: int, 
        init_loss: torch.Tensor,
        device="cpu",
        eval_field="loss",
        nbeta: float = None,
        temperature: Optional[float] = None, # Temperature is deprecated
    ):
        self.num_draws = num_draws
        self.init_loss = init_loss

        self.losses = torch.zeros((num_chains, num_draws)).to(device)
        self.llcs = torch.zeros((num_chains, num_draws)).to(device)

        assert nbeta is not None or temperature is not None, "Please provide a value for nbeta."
        if nbeta is None and temperature is not None:
            nbeta = temperature
            warnings.warn("Temperature is deprecated. Please use nbeta instead.")
        self.nbeta = torch.tensor(nbeta, dtype=torch.float32).to(device)
        
        self.device = device
        self.eval_field = eval_field

        self.llc_means = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_draws, dtype=torch.float32).to(device)
        self.moving_avg_llcs = torch.zeros(
            (num_chains, num_draws), dtype=torch.float32
        ).to(device)
        
    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss
        self.llcs[chain, draw] = self.nbeta * (loss - self.init_loss)
        if draw == 0:
            self.moving_avg_llcs[chain, draw] = self.nbeta * (
                loss - self.init_loss
            )
        else:
            t = draw + 1
            prev_moving_avg = self.moving_avg_llcs[chain, draw - 1]
            self.moving_avg_llcs[chain, draw] = (1 / t) * (
                (t - 1) * prev_moving_avg + self.nbeta * (loss - self.init_loss)
            )

    def finalize(self):
        self.llc_means = self.llcs.mean(dim=0)
        self.llc_stds = self.llcs.std(dim=0)

    def get_results(self):
        """    
        :returns: A dict :python:`{"llc/means": llc_means, "llc/stds": llc_stds, "llc/trace": llc_trace_per_chain, "loss/trace": loss_trace_per_chain}`. (Only after running :python:`devinterp.slt.sampler.sample(..., [llc_estimator_instance], ...)`).
        """
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/moving_avg": self.moving_avg_llcs.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }

    def __call__(self, chain: int, draw: int, **kwargs):
        self.update(chain, draw, kwargs[self.eval_field])
