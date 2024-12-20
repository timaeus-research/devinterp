import warnings
from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, Iterator, List, Literal, Optional, Union

import torch
from devinterp.optim.preconditioner import (
    CompositePreconditioner,
    IdentityPreconditioner,
    MaskPreconditioner,
    NHTPreconditioning,
    Preconditioner,
    RMSpropPreconditioner,
)
from devinterp.optim.prior import CompositePrior, GaussianPrior, Prior
from devinterp.optim.utils import OptimizerMetric
from torch.optim import Optimizer

SamplingMethodLiteral = Literal["sgld", "rmsprop_sgld", "sgnht"]


class SGMCMC(Optimizer):
    """Unified Stochastic Gradient Markov Chain Monte Carlo (SGMCMC) optimizer.

    This optimizer implements a general SGMCMC framework that unifies several common variants
    like SGLD, SGHMC, and SGNHT. It supports custom priors and preconditioners for flexible
    posterior sampling.

    The general update rule follows:

    .. math::

        Δθ_t = (ε/2)G(θ_t)(nβ/m ∑∇logp(y|x,θ_t) + ∇logp(θ_t)) + √(εG(θ_t))N(0,σ²)

    where:

    * ε is learning rate
    * nβ is inverse temperature
    * m is batch size
    * G(θ) is the preconditioner
    * p(θ) is the prior distribution
    * σ is noise level

    :param params: Iterable of parameters to optimize or dicts defining parameter groups
    :param lr: Learning rate ε (default: 0.01)
    :param noise_level: Standard deviation σ of the noise (default: 1.0)
    :param nbeta: Inverse temperature nβ (default: 1.0)
    :param prior: Prior distribution specification. Can be:
        - GaussianPrior instance
        - string specifying center type
        - iterable of tensor centers
        - float specifying precision
        (default: None)
    :param prior_kwargs: Additional keyword arguments for prior initialization
    :param preconditioner: Preconditioner specification. Can be:
        - "identity" for no preconditioning
        - "rmsprop" for RMSprop-style preconditioning
        - Preconditioner instance
        (default: None, equivalent to "identity")
    :param preconditioner_kwargs: Additional keyword arguments for preconditioner
    :param bounding_box_size: Size of bounding box around initial parameters
    :param optimize_over: Boolean mask for restricting updatable parameters
    :param metrics: List of metrics to track during training
    :param weight_decay: Weight decay factor applied separately from other updates (like AdamW).
        For preconditioned weight decay (like Adam), use a GaussianPrior centered at zero instead.
        (default: 0.0)
    :type params: Iterable
    :type lr: float
    :type noise_level: float
    :type nbeta: float
    :type prior: Optional[Union[Prior, Literal["initial"], Iterable[torch.Tensor], float]]
    :type prior_kwargs: Optional[Dict]
    :type preconditioner: Optional[Union[Preconditioner, str]]
    :type preconditioner_kwargs: Optional[Dict]
    :type bounding_box_size: Optional[float]
    :type optimize_over: Optional[torch.Tensor]
    :type metrics: Optional[List[OptimizerMetric]]
    :type weight_decay: float

    Valid metrics options:
        * noise_norm
        * grad_norm
        * weight_norm
        * distance
        * noise
        * dws
        * localization_loss

    Example::

        from devinterp.utils import default_nbeta

        # Basic SGLD-style usage
        optimizer = SGMCMC.sgld(
            model.parameters(),
            lr=0.1,
            nbeta=default_nbeta(dataloader)
        )

        # RMSprop-preconditioned with prior
        optimizer = SGMCMC.rmsprop_sgld(
            model.parameters(),
            lr=0.01,
            localization=0.1,
            nbeta=default_nbeta(dataloader)
        )

        # SGNHT-style with thermostat
        optimizer = SGMCMC.sgnht(
            model.parameters(),
            lr=0.01,
            diffusion_factor=0.01,
            nbeta=default_nbeta(dataloader)
        )

        # Training loop
        for data, target in dataloader:
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()

    Notes:
        * Use the factory methods (sgld, rmsprop_sgld, sgnht) for easier initialization
        * nbeta should typically be set using devinterp.utils.default_nbeta() rather than manually
        * The prior helps explore the local posterior by pulling toward initialization
        * Tracked metrics can be accessed as attributes, e.g. optimizer.grad_norm

    References:
        * Welling & Teh (2011) - Original SGLD paper
        * Li et al. (2015) - RMSprop-SGLD
        * Ding et al. (2014) - SGNHT
        * Lau et al. (2023) - Implementation with localization term
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        noise_level: float = 1.0,
        nbeta: float = 1.0,
        prior: Optional[
            Union[Prior, Literal["initial"], Iterable[torch.Tensor], float]
        ] = None,
        prior_kwargs: Optional[Dict] = None,
        localization: Optional[float] = None,
        preconditioner: Optional[
            Union[Preconditioner, Literal["identity", "rmsprop"]]
        ] = "identity",
        preconditioner_kwargs: Optional[Dict] = None,
        bounding_box_size: Optional[float] = None,
        optimize_over: Optional[torch.Tensor] = None,
        metrics: Optional[List[OptimizerMetric]] = None,
        weight_decay: float = 0.0,
    ):
        """
        The localization parameter controls how strongly parameters are pulled back toward their
        initialization point (or other specified center). If provided directly, it will override
        prior_kwargs["localization"].
        """
        # Handle single parameter case
        if isinstance(params, (torch.Tensor, dict)):
            params = [params]

        # Define per-group parameters
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            nbeta=nbeta,
            weight_decay=weight_decay,
            prior=prior,
            prior_kwargs=prior_kwargs or {},
            localization=localization,
            preconditioner=preconditioner,
            preconditioner_kwargs=preconditioner_kwargs or {},
            bounding_box_size=bounding_box_size,
            optimize_over=optimize_over,
            metrics=metrics or [],
        )
        super().__init__(params, defaults)

        # Initialize each parameter group
        for group in self.param_groups:
            self._init_group(group)

    def _init_group(self, group: dict) -> None:
        """Initialize all group-specific settings.

        Prior initialization supports several formats:
        1. Prior object: Use any existing Prior instance directly
        2. String ("initial"): Creates GaussianPrior centered at parameter initialization
        3. Tensor centers: Creates GaussianPrior centered at provided tensor values
        4. Number (float/int): Creates GaussianPrior with specified localization strength

        The localization parameter controls how strongly parameters are pulled toward their center:
        - If prior is None but localization > 0, defaults to "initial" center
        - Default localization is 0.0 (no pull)
        - Additional prior parameters can be passed via prior_kwargs

        Args:
            group: Parameter group dictionary containing optimizer settings
        """

        # Validate metrics
        valid_metrics = {
            "noise_norm",
            "grad_norm",
            "weight_norm",
            "distance",
            "noise",
            "dws",
            "localization_loss",
        }
        if not set(group["metrics"]).issubset(valid_metrics):
            raise ValueError(
                f"Invalid metrics {group['metrics']}. Choose from: {valid_metrics}"
            )

        # Initialize metrics directly in the group
        group["metrics"] = {
            metric: (
                []
                if metric in {"noise", "dws"}
                else torch.zeros((), device=self.device)
            )
            for metric in group["metrics"]
        }

        # Initialize prior
        prior = group["prior"]
        prior_kwargs = group.pop("prior_kwargs")
        localization = group.get("localization", prior_kwargs.get("localization", 0.0))

        if prior is not None or localization:
            prior = prior if prior is not None else "initial"
            localization = localization or 1.0

            if isinstance(prior, Prior):
                group["prior"] = prior
            elif isinstance(prior, str):
                group["prior"] = GaussianPrior(
                    localization=localization, center=prior, **prior_kwargs
                )
            elif isinstance(prior, Iterable) and not isinstance(prior, str):
                group["prior"] = GaussianPrior(
                    localization=localization, center=prior, **prior_kwargs
                )
            elif isinstance(prior, (int, float)):
                group["prior"] = GaussianPrior(
                    localization=float(prior), **prior_kwargs
                )
            else:
                raise ValueError(f"Unsupported prior type: {type(prior)}")

        # Initialize preconditioner
        preconditioner = group["preconditioner"]
        preconditioner_kwargs = group.pop("preconditioner_kwargs")

        if preconditioner is None or preconditioner == "identity":
            group["preconditioner"] = IdentityPreconditioner(**preconditioner_kwargs)
        elif preconditioner == "rmsprop":
            group["preconditioner"] = RMSpropPreconditioner(**preconditioner_kwargs)
        elif isinstance(preconditioner, Preconditioner):
            group["preconditioner"] = preconditioner
        else:
            raise ValueError(f"Unsupported preconditioner type: {preconditioner}")

        optimize_over = group.pop("optimize_over", None)
        if optimize_over is not None:
            # Convert optimize_over to masks (1.0 where True, 0.0 where False)
            if isinstance(optimize_over, torch.Tensor):
                optimize_over = [optimize_over]

            def _process_mask(m):
                if not isinstance(m, torch.Tensor):
                    m = torch.tensor(m).to(self.device)

                return m.float()

            masks = [_process_mask(mask) for mask in optimize_over]
            mask_preconditioner = MaskPreconditioner(masks=masks)

            if group["preconditioner"] is not None:
                group["preconditioner"] = CompositePreconditioner(
                    [group["preconditioner"], mask_preconditioner]
                )
            else:
                group["preconditioner"] = mask_preconditioner

        pstates = {}

        # Initialize prior state if needed
        if group["prior"] is not None:
            pstates = group["prior"].initialize(iter(group["params"]))

        # Initialize states for each parameter in the group
        for i, p in enumerate(group["params"]):
            pstate = pstates.get(p, {})

            self.state[p] = {
                **self.state[p],
                **pstate,
                "param_idx": i,
                "initial_param": (
                    p.data.clone().detach()
                    if group["bounding_box_size"] is not None
                    else None
                ),
            }

    def reset_metrics(self):
        """Reset all tracked metrics at the start of each step."""
        for group in self.param_groups:
            for metric, value in group["metrics"].items():
                if isinstance(value, torch.Tensor):
                    value.zero_()
                else:  # List metrics (noise, dws)
                    value.clear()

    def zero_grad(self):
        """Zero out gradients"""
        self.reset_metrics()
        super().zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            Optional[float]: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            prior = group["prior"]
            preconditioner = group["preconditioner"]
            params = group["params"]

            for i, p in enumerate(params):
                if p.grad is None:
                    continue

                state = self.state[p]

                # Get preconditioner coefficients
                preconditioning = preconditioner.get_coefficients(p, p.grad, state)

                # Gradient computation
                d_p = preconditioning.grad_coef * p.grad.mul(group["nbeta"])

                # Prior gradient contribution
                if prior is not None:
                    prior_grad = prior.grad(p.data, state)
                    d_p.add_(preconditioning.prior_coef * prior_grad)
                    if (
                        "distance" in self.tracked_metrics
                        or "localization_loss" in self.tracked_metrics
                    ):
                        group["metrics"]["distance"] += prior.distance_sq(p.data, state)

                # if (
                #     isinstance(preconditioning.overall_coef, torch.Tensor)
                #     and not preconditioning.overall_coef.any()
                # ) or (
                #     not isinstance(preconditioning.overall_coef, torch.Tensor)
                #     and not preconditioning.overall_coef
                # ):
                #     continue

                # Noise addition
                noise = torch.normal(
                    mean=0.0,
                    std=group["noise_level"],
                    size=d_p.size(),
                    device=d_p.device,
                )

                # Apply weight decay separately from other updates
                if group["weight_decay"] != 0:
                    d_p.add_(group["weight_decay"] * p.data)

                d_p = preconditioning.overall_coef * d_p
                noise = preconditioning.overall_coef * noise

                # Parameter updates
                p.data.add_(d_p, alpha=-0.5 * group["lr"])
                p.data.add_(
                    preconditioning.noise_coef * noise,
                    alpha=group["lr"] ** 0.5,
                )

                # Bounding box enforcement
                if group["bounding_box_size"] is not None:
                    initial_param = state["initial_param"]
                    torch.clamp_(
                        p.data,
                        min=initial_param - group["bounding_box_size"],
                        max=initial_param + group["bounding_box_size"],
                    )

                # Track metrics
                metrics = group["metrics"]
                if "dws" in metrics:
                    metrics["dws"].append(d_p.clone())

                if "grad_norm" in metrics and p.grad is not None:
                    metrics["grad_norm"] += (
                        (p.grad.data * group["nbeta"] * 0.5 * group["lr"]) ** 2
                    ).sum()

                if "weight_norm" in metrics:
                    metrics["weight_norm"] += (p.data**2).sum()

                if "noise" in metrics:
                    metrics["noise"].append(noise)

                if "noise_norm" in metrics:
                    metrics["noise_norm"] += (noise**2).sum()

        # Finalize scalar metrics
        for group in self.param_groups:
            metrics = group["metrics"]

            for metric in {
                "grad_norm",
                "weight_norm",
                "noise_norm",
                "distance",
            } & metrics.keys():
                metrics[metric] = torch.sqrt(metrics[metric])

            if "localization_loss" in metrics and group["prior"] is not None:
                metrics["localization_loss"] = (
                    metrics["distance"] ** 2 * (group["prior"].localization) / 2
                )

        return loss

    def get_params(self) -> Iterator[torch.Tensor]:
        """Helper to get all parameters"""
        for group in self.param_groups:
            for p in group["params"]:
                yield p

    @classmethod
    def sgld(
        cls,
        params,
        lr=0.01,
        noise_level=1.0,
        weight_decay=0.0,
        localization=0.0,
        nbeta=1.0,
        bounding_box_size=None,
        optimize_over=None,
        metrics: Optional[List[OptimizerMetric]] = None,
    ):
        """Factory method to create an SGMCMC instance that implements Stochastic Gradient Langevin Dynamics (SGLD)
        with a localization term (Lau et al. 2023).

        This optimizer combines Stochastic Gradient Descent (SGD) with Langevin Dynamics,
        introducing Gaussian noise to the gradient updates. This makes it sample weights from
        the posterior distribution, instead of finding point estimates through optimization (Welling and Teh 2011).

        The update rule follows::

            Δθ_t = (ε/2)(nβ/m ∑∇logp(y|x,θ_t) + γ(θ_0-θ_t) - λθ_t) + N(0,εσ²)

        where:

        * ε is learning rate
        * nβ is inverse temperature
        * m is batch size
        * γ is localization strength
        * λ is weight decay
        * σ is noise level

        This follows Lau et al.'s (2023) implementation, which modifies Welling and Teh (2011) by:

        * Omitting the learning rate schedule (this functionality could be recoverd by using a separate learning rate scheduler).
        * Adding a localization term that pulls weights toward initialization
        * Using tempered Bayes paradigm with inverse temperature nβ


        This allows SGMCMC to be used as a drop-in replacement for SGLD.
        :param params: Iterable of parameters to optimize
        :param lr: Learning rate (default: 0.01)
        :param noise_level: Standard deviation of noise (default: 1.0)
        :param weight_decay: Weight decay factor. Applied with preconditioning (Adam-style).
            Creates a GaussianPrior centered at zero with localization=weight_decay.
            (default: 0.0)
        :param localization: Strength of pull toward initial parameters.
            Creates a GaussianPrior centered at initialization with localization=localization.
            (default: 0.0)
        :param nbeta: Inverse temperature (default: 1.0)
        :param bounding_box_size: Size of bounding box around initial parameters (default: None)
        :param optimize_over: Boolean mask for restricting updatable parameters (default: None)
        :param metrics: List of metrics to track during training (default: None)
        :return: SGMCMC optimizer instance
        """
        if noise_level != 1.0:
            warnings.warn(
                "noise_level in SGLD is unequal to one, this removes SGLD posterior sampling guarantees."
            )
        if nbeta == 1.0:
            warnings.warn(
                "nbeta set to 1, LLC estimates will be off unless you know what you're doing. Use utils.default_nbeta(dataloader) instead"
            )

        # if isinstance(params, list) and all(isinstance(p, dict) for p in params):
        #     raise ValueError(
        #         "params should be an iterator of parameters, not param_groups"
        #     )

        # Updated prior initialization to handle both weight decay and localization
        priors = []
        if weight_decay > 0:
            priors.append(GaussianPrior(localization=weight_decay, center=None))

        priors.append(GaussianPrior(localization=localization, center="initial"))

        prior = CompositePrior(priors)
        prior_kwargs = {}

        instance = cls(
            params,
            lr=lr,
            noise_level=noise_level,
            nbeta=nbeta,
            prior=prior,
            prior_kwargs=prior_kwargs,
            bounding_box_size=bounding_box_size,
            optimize_over=optimize_over,
            metrics=metrics,
        )

        return instance

    @classmethod
    def sgnht(
        cls,
        params,
        lr=0.01,
        diffusion_factor=0.01,
        nbeta=1.0,
        bounding_box_size=None,
        metrics: Optional[List[OptimizerMetric]] = None,
    ):
        """Factory method to create an SGMCMC instance that matches SGNHT's interface.

        This allows SGMCMC to be used as a drop-in replacement for SGNHT.

        :param params: Iterable of parameters to optimize
        :param lr: Learning rate (default: 0.01)
        :param diffusion_factor: Diffusion factor (default: 0.01)
        :param nbeta: Inverse temperature (default: 1.0)
        :param bounding_box_size: Size of bounding box around initial parameters (default: None)
        :param metrics: List of metrics to track during training (default: None)
        :return: SGMCMC optimizer instance
        """
        if nbeta == 1.0:
            warnings.warn(
                "nbeta set to 1, LLC estimates will be off unless you know what you're doing. Use utils.default_nbeta(dataloader) instead"
            )

        # Create NHT preconditioner
        preconditioner = NHTPreconditioning(diffusion_factor=diffusion_factor)

        instance = cls(
            params,
            lr=lr,
            noise_level=1.0,  # Noise scaling handled by preconditioner
            nbeta=nbeta,
            preconditioner=preconditioner,
            bounding_box_size=bounding_box_size,
            metrics=metrics,  # Pass metrics here
        )

        return instance

    @classmethod
    def rmsprop_sgld(
        cls,
        params,
        lr=0.01,
        noise_level=1.0,
        weight_decay=0.0,
        localization=0.0,
        nbeta=1.0,
        alpha=0.99,
        eps=0.1,
        add_grad_correction=False,
        bounding_box_size=None,
        optimize_over=None,
        metrics: Optional[List[OptimizerMetric]] = None,
    ):
        """Factory method to create an SGMCMC instance that wraps RMSprop's adaptive preconditioning with SGLD to perform Bayesian
        sampling of neural network weights.

        The update rule with preconditioning follows::

            V(θ_t) = αV(θ_{t-1}) + (1-α)g̅(θ_t)g̅(θ_t)
            G(θ_t) = diag(1/(λ1 + √V(θ_t)))
            Δθ_t = (ε/2)G(θ_t)(nβ/m ∑∇logp(y|x,θ_t) + γ(θ_0-θ_t) - λθ_t) + √(εG(θ_t))N(0,σ²)

        where:

        * ε is learning rate
        * nβ is effective dataset size (=dataset size * inverse temperature)
        * m is batch size
        * γ is localization strength
        * λ is weight decay
        * σ is noise level
        * G(θ) is the RMSprop preconditioner
        * V(θ) tracks squared gradient moving average
        * α is the exponential decay rate

        Key differences from standard SGLD:

        * Uses RMSprop preconditioner to adapt to local geometry and curvature
        * Scales both the gradients and noise by the preconditioner
        * Handles pathological curvature through adaptive step sizes

        :param params: Iterable of parameters to optimize
        :param lr: Learning rate (default: 0.01)
        :param noise_level: Standard deviation of noise (default: 1.0)
        :param weight_decay: Weight decay factor. Applied with preconditioning (Adam-style).
            Creates a GaussianPrior centered at zero with localization=weight_decay.
            (default: 0.0)
        :param localization: Strength of pull toward initial parameters.
            Creates a GaussianPrior centered at initialization with localization=localization.
            (default: 0.0)
        :param nbeta: Inverse temperature (default: 1.0)
        :param alpha: RMSprop moving average coefficient (default: 0.99)
        :param eps: RMSprop stability constant (default: 0.1)
        :param add_grad_correction: Whether to add gradient correction term (default: False)
        :param bounding_box_size: Size of bounding box around initial parameters (default: None)
        :param optimize_over: Boolean mask for restricting updatable parameters (default: None)
        :param metrics: List of metrics to track during training (default: None)
        :return: SGMCMC optimizer instance
        """
        if noise_level != 1.0:
            warnings.warn(
                "noise_level in RMSProp-SGLD is unequal to one, this removes SGLD posterior sampling guarantees."
            )
        if nbeta == 1.0:
            warnings.warn(
                "nbeta set to 1, LLC estimates will be off unless you know what you're doing. Use utils.default_nbeta(dataloader) instead"
            )

        # if isinstance(params, list) and all(isinstance(p, dict) for p in params):
        #     raise ValueError(
        #         "params should be an iterator of parameters, not param_groups"
        #     )

        # Updated prior initialization to handle both weight decay and localization
        priors = []
        prior = None
        if weight_decay > 0:
            priors.append(GaussianPrior(localization=weight_decay, center=None))

        priors.append(GaussianPrior(localization=localization, center="initial"))

        prior = CompositePrior(priors)
        prior_kwargs = {}

        # Configure RMSprop preconditioner
        preconditioner_kwargs = {
            "alpha": alpha,
            "eps": eps,
            "add_grad_correction": add_grad_correction,
        }

        instance = cls(
            params,
            lr=lr,
            noise_level=noise_level,
            nbeta=nbeta,
            prior=prior,
            prior_kwargs=prior_kwargs,
            preconditioner="rmsprop",
            preconditioner_kwargs=preconditioner_kwargs,
            bounding_box_size=bounding_box_size,
            optimize_over=optimize_over,
            metrics=metrics,
        )

        return instance

    @classmethod
    def get_method(cls, method: SamplingMethodLiteral):
        if method == "sgld":
            return cls.sgld
        elif method == "rmsprop_sgld":
            return cls.rmsprop_sgld
        elif method == "sgnht":
            return cls.sgnht
        else:
            raise ValueError(
                f"`method` should be one of 'sgld', 'rmsprop_sgld', or 'sgnht'. Got {method}"
            )

    @property
    def device(self):
        return next(self.get_params()).device

    @property
    def metrics(self):
        if len(self.param_groups) > 1:
            warnings.warn(
                "metrics is only available for single-parameter groups. Returning first group's metrics."
            )

        return self.param_groups[0]["metrics"]

    @property
    def tracked_metrics(self):
        if len(self.param_groups) > 1:
            warnings.warn(
                "tracked_metrics is only available for single-parameter groups. Returning first group's tracked metrics."
            )

        return set(self.param_groups[0]["metrics"].keys())

    def __getattr__(self, name):
        """Return metrics if there's only one param group"""

        # For other attributes, check if they exist in param_groups
        if len(self.param_groups) == 1:
            if name in self.param_groups[0]["metrics"]:
                return self.param_groups[0]["metrics"][name]

        raise AttributeError(f"'SGMCMC' object has no attribute '{name}'")
