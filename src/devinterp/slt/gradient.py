from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.gradient import sample
    raise NotImplementedError("TPU backend not supported for Gradients")
else:
    from devinterp.backends.default.slt.gradient import GradientDistribution
