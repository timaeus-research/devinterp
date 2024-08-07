from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    from devinterp.backends.default.slt.mala import (
        MalaAcceptanceRate,
        mala_acceptance_probability,
    )
else:
    from devinterp.backends.default.slt.mala import (
        MalaAcceptanceRate,
        mala_acceptance_probability,
    )

