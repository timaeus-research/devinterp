from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.loss import OnlineLossStatistics
    raise NotImplementedError("TPU backend not supported for Loss")
else:
    from devinterp.backends.default.slt.loss import OnlineLossStatistics
