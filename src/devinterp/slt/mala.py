from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.mala import sample
    raise NotImplementedError("TPU backend not supported for Mala")
else:
    from devinterp.backends.default.slt.mala import (
        MalaAcceptanceRate,
        mala_acceptance_probability,
    )
