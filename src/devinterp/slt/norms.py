from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.norms import sample
    raise NotImplementedError("TPU backend not supported for Norms")
else:
    from devinterp.backends.default.slt.norms import GradientNorm, NoiseNorm, WeightNorm
