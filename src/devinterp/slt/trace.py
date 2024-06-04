from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.trace import OnlineTraceStatistics
    raise NotImplementedError("TPU backend not supported for Trace")
else:
    from devinterp.backends.default.slt.trace import OnlineTraceStatistics
