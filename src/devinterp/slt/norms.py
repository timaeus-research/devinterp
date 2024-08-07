from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    from devinterp.backends.tpu.slt.norms import GradientNorm, NoiseNorm, WeightNorm
else:
    from devinterp.backends.default.slt.norms import GradientNorm, NoiseNorm, WeightNorm
