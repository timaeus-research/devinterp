from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.cov import sample
    raise NotImplementedError("TPU backend not supported for Covariance")
else:
    from devinterp.backends.default.slt.cov import (
        AttentionHeadWeightsAccessor,
        BetweenLayerCovarianceAccumulator,
        CovarianceAccumulator,
        CovarianceEstimator,
        LayerWeightsAccessor,
        WithinHeadCovarianceAccumulator,
    )
