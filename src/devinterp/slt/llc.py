from devinterp.slt.callback import *
from devinterp.utils import USE_TPU_BACKEND

if USE_TPU_BACKEND:
    # from devinterp.backends.tpu.slt.llc import sample
    from devinterp.backends.tpu.slt import *
else:
    from devinterp.backends.default.slt import *
