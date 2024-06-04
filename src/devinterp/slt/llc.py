from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.llc import sample
    raise NotImplementedError("TPU backend not supported for LLC")
else:
    from devinterp.backends.default.slt.llc import LLCEstimator, OnlineLLCEstimator
