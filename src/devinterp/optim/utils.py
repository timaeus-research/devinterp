from typing import Literal

OptimizerMetric = Literal[
    "distance",  # Distance (L2 norm) between prior center and current weights (not post-update weights)
    "weight_norm",  # L2 norm of current weights
    "grad_norm",  # L2 norm of currentgradients
    "localization_loss",  # Localization loss
    "noise_norm",  # L2 norm of noise
    "dw",  # Vector change in weights (not post-update weights)
    "noise",  # Vector noise (not post-update weights)
    # SGNHT-specific:
    "momentum_norm",  # L2 norm of momentum
    "thermostat",  # Thermostat variable
]
