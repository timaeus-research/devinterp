from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.llc import sample
    raise NotImplementedError("TPU backend not supported for WBIC")
else:
    from devinterp.backends.default.slt.wbic import OnlineWBICEstimator
