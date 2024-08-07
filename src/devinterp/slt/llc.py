from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.llc import sample
    from devinterp.backends.tpu.slt.llc import LLCEstimator, OnlineLLCEstimator
else:
    from devinterp.backends.default.slt.llc import LLCEstimator, OnlineLLCEstimator
