import numpy as np

from devinterp.slt.callback import ChainCallback
from devinterp.slt.llc import OnlineLLCEstimator

class TraceStatistics(ChainCallback):
    """Derivative callback of OnlineLLCEstimator that computes mean/std statistics of the loss and llc traces.

    Parameters:
        online_llc_estimator (OnlineLLCEstimator): Base callback that computes the llc and loss traces.
    """
    def __init__(self, online_llc_estimator: OnlineLLCEstimator):
        self.estimator = online_llc_estimator

        self.num_chains = online_llc_estimator.num_chains
        self.num_draws = online_llc_estimator.num_draws

        self.llc_mean_by_chain = np.zeros(self.num_chains, dtype=np.float32)
        self.llc_std_by_chain = np.zeros(self.num_chains, dtype=np.float32)

        self.llc_mean_by_draw = np.zeros(self.num_draws, dtype=np.float32)
        self.llc_std_by_draw = np.zeros(self.num_draws, dtype=np.float32)

        self.loss_mean_by_chain = np.zeros(self.num_chains, dtype=np.float32)
        self.loss_std_by_chain = np.zeros(self.num_chains, dtype=np.float32)

        self.loss_mean_by_draw = np.zeros(self.num_draws, dtype=np.float32)
        self.loss_std_by_draw = np.zeros(self.num_draws, dtype=np.float32)

    def finalize(self):
        if not self.esimator.finalized:
            raise RuntimeError("Cannot finalize TraceStatistics before OnlineLLCEstimator, " +
                               "ensure TraceStatistics is passed later in the list of callbacks.")
        
        llcs = self.estimator.llcs.cpu().numpy()
        losses = self.estimator.losses.cpu().numpy()

        self.llc_mean_by_chain = llcs.mean(axis=1)
        self.llc_std_by_chain = llcs.std(axis=1)

        self.llc_mean_by_draw = llcs.mean(axis=0)
        self.llc_std_by_draw = llcs.std(axis=0)

        self.loss_mean_by_chain = losses.mean(axis=1)
        self.loss_std_by_chain = losses.std(axis=1)

        self.loss_mean_by_draw = losses.mean(axis=0)
        self.loss_std_by_draw = losses.std(axis=0)

    def sample(self):
        return {
            'llc/chain/mean': self.llc_mean_by_chain,
            'llc/chain/std': self.llc_std_by_chain,
            'llc/draw/mean': self.llc_mean_by_draw,
            'llc/draw/std': self.llc_std_by_draw,
            'loss/chain/mean': self.loss_mean_by_chain,
            'loss/chain/std': self.loss_std_by_chain,
            'loss/draw/mean': self.loss_mean_by_draw,
            'loss/draw/std': self.loss_std_by_draw,
        }

    def __call__(self):
        pass