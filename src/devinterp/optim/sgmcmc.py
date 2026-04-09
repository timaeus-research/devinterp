import warnings
from typing import Iterable, Iterator, Literal, NamedTuple, Optional, Union

import torch
from torch.optim import Optimizer

from devinterp.optim.metrics import Metrics
from devinterp.optim.preconditioner import (
    CompositePreconditioner,
    IdentityPreconditioner,
    MaskPreconditioner,
    NHTPreconditioning,
    Preconditioner,
    PreconditionerCoefs,
    RMSpropPreconditioner,
)
from devinterp.optim.prior import CompositePrior, GaussianPrior, Prior
from devinterp.optim.sketch import CountSketch, SketchBuffer

SamplingMethodLiteral = Literal["sgld", "rmsprop_sgld", "sgnht"]


class _ComponentVectors(NamedTuple):
    """Post-preconditioned component vectors from a single parameter update."""

    # Structurally identical to the copy in sgld.py. Kept separate because
    # SGLD is deprecated — sharing a private type would couple the old
    # module to this one and complicate its eventual removal.

    scaled_grad: torch.Tensor
    unscaled_grad: torch.Tensor
    localization: torch.Tensor
    weight_decay: torch.Tensor
    noise: torch.Tensor


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
        In SGMCMC, weight_decay and localization should be implemented
        as a prior. See the `rmsprop_sgld()` method for an example.
        - GaussianPrior instance
        - string specifying center type
        - iterable of tensor centers
        - float specifying precision
        (default: None)
    :param preconditioner: Preconditioner specification. Can be:
        - "identity" for no preconditioning
        - "rmsprop" for RMSprop-style preconditioning
        - Preconditioner instance
        (default: None, equivalent to "identity")
    :param preconditioner_kwargs: Additional keyword arguments for preconditioner
    :param bounding_box_size: Size of bounding box around initial parameters
    :param mask: Boolean mask for restricting updatable parameters
    :param save_metrics: Whether to track metrics during training (default: False)
    :type params: Iterable
    :type lr: float
    :type noise_level: float
    :type nbeta: float
    :type prior: Optional[Union[Prior, Literal["initial"], Iterable[torch.Tensor], float]]
    :type preconditioner: Optional[Union[Preconditioner, str]]
    :type preconditioner_kwargs: Optional[dict]
    :type bounding_box_size: Optional[float]
    :type mask: Optional[torch.Tensor]
    :type save_metrics: bool

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
        * Use save_metrics=True and call get_metrics() to access tracked metrics

    References:
        * Welling & Teh (2011) - Original SGLD paper
        * Li et al. (2015) - RMSprop-SGLD
        * Ding et al. (2014) - SGNHT
        * Lau et al. (2023) - Implementation with localization term
    """

    def __init__(
        self,
        params,
        *,
        lr: float = 0.01,
        noise_level: float = 1.0,
        nbeta: float = 1.0,
        prior: Optional[
            Union[Prior, Literal["initial"], Iterable[torch.Tensor], float]
        ] = None,
        preconditioner: Optional[
            Union[Preconditioner, Literal["identity", "rmsprop"]]
        ] = "identity",
        preconditioner_kwargs: Optional[dict] = None,
        bounding_box_size: Optional[float] = None,
        mask: Optional[torch.Tensor] = None,
        save_metrics: bool = False,
        sketch_dim: Optional[int] = None,
        sketch_seed: int = 0,
    ):
        # Handle single parameter case
        if isinstance(params, (torch.Tensor, dict)):
            params = [params]

        # Define per-group parameters
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            nbeta=nbeta,
            prior=prior,
            preconditioner=preconditioner,
            preconditioner_kwargs=preconditioner_kwargs or {},
            bounding_box_size=bounding_box_size,
            mask=mask,
        )
        super().__init__(params, defaults)

        self.save_metrics = save_metrics

        if sketch_dim is not None:
            self._total_numel = sum(
                p.numel() for group in self.param_groups for p in group["params"]
            )
            device = next(iter(self.param_groups[0]["params"])).device
            self._sketch = CountSketch.create(
                self._total_numel, sketch_dim, sketch_seed
            ).to(device)
            # Single buffer shared across all param groups. Sketch accumulation
            # is purely additive (linear), so per-group buffers are unnecessary.
            # N.B. With FSDP each rank only sees a parameter shard; this buffer
            # would hold only the local contribution and would need an all-reduce
            # across ranks before consumption. See the FSDP guard in sampler.py.
            self._sketch_buf = SketchBuffer.create(sketch_dim, device)
            self.save_sketches = True
        else:
            self._sketch = None
            self._sketch_buf = None
            self.save_sketches = False

        # Initialize each parameter group
        for group in self.param_groups:
            self._init_group(group)

    def _init_group(self, group: dict) -> None:
        """Initialize all group-specific settings.

        Prior initialization supports several formats:
        1. Prior object: Use any existing Prior instance directly
          - localization and weight_decay should be implemented via a Prior. To see how
            to do this, look at e.g. `SGMCMC.sgld()`.
        2. String ("initial"): Creates GaussianPrior centered at parameter initialization
        3. Tensor centers: Creates GaussianPrior centered at provided tensor values
        4. Number (float/int): Creates GaussianPrior with specified localization strength

        Args:
            group: Parameter group dictionary containing optimizer settings
        """
        # Initialize prior
        prior = group["prior"]
        localization = group.get("localization", 0.0)

        if prior is not None or localization:
            prior = prior if prior is not None else "initial"
            localization = localization or 1.0

            if isinstance(prior, Prior):
                group["prior"] = prior
            elif isinstance(prior, str):
                group["prior"] = GaussianPrior(localization=localization, center=prior)
            elif isinstance(prior, Iterable) and not isinstance(prior, str):
                group["prior"] = GaussianPrior(localization=localization, center=prior)
            elif isinstance(prior, (int, float)):
                group["prior"] = GaussianPrior(localization=float(prior))
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

        mask = group.pop("mask", None)
        if mask is not None:
            # Convert mask to masks (1.0 where True, 0.0 where False)
            if isinstance(mask, torch.Tensor):
                mask = [mask]

            def _process_mask(m, p):
                if not isinstance(m, torch.Tensor):
                    m = torch.tensor(m).to(self.device)

                # Validate mask shape matches parameter shape
                if m.shape != p.shape:
                    raise ValueError(
                        f"Mask shape {m.shape} does not match parameter shape {p.shape}. "
                    )

                return m.float()

            params = list(group["params"])
            masks = [_process_mask(_mask, p) for _mask, p in zip(mask, params)]
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
            pstates = group["prior"].initialize(list(group["params"]))

        # Initialize states for each parameter in the group
        store_initial = group["bounding_box_size"] is not None or self.save_metrics
        for i, p in enumerate(group["params"]):
            pstate = pstates.get(p, {})

            self.state[p] = {
                **self.state[p],
                **pstate,
                "param_idx": i,
                "initial_param": (p.data.clone().detach() if store_initial else None),
            }

        # Initialize per-group metrics
        if self.save_metrics:
            device = next(iter(group["params"])).device
            group["metrics"] = Metrics().to(device)

    def _decompose_update(
        self,
        group: dict,
        p: torch.Tensor,
        loc_grad: Optional[torch.Tensor],
        wd_grad: Optional[torch.Tensor],
        noise: torch.Tensor,
        preconditioning: PreconditionerCoefs,
        d_p: torch.Tensor,
    ) -> _ComponentVectors:
        """Decompose a parameter update into its post-preconditioned component vectors.

        Shared by both metrics accumulation and sketch scattering. Debug
        assertions verify the decomposition matches the actual update.

        Args:
            loc_grad: Pre-computed localization prior gradient, or None.
            wd_grad: Pre-computed weight_decay prior gradient, or None.
            d_p: The actual deterministic update vector (for debug assertion).
        """
        _lr = group["lr"]
        _precond = preconditioning.overall_coef
        _half_lr = 0.5 * _lr

        _grad_pre = preconditioning.grad_coef * p.grad.mul(group["nbeta"])
        _loc_pre = loc_grad if loc_grad is not None else torch.zeros_like(p)
        _wd_pre = wd_grad if wd_grad is not None else torch.zeros_like(p)

        _scaled_grad = _half_lr * (_precond * _grad_pre)
        _unscaled_grad = _half_lr * p.grad.mul(group["nbeta"])

        _prior_scale = _half_lr * preconditioning.prior_coef * _precond
        _loc = _prior_scale * _loc_pre
        _wd = _prior_scale * _wd_pre

        # noise already includes overall_coef from step()
        _noise = (group["lr"] ** 0.5) * preconditioning.noise_coef * noise

        if __debug__:
            _combined_prior = (
                _half_lr * preconditioning.prior_coef * _precond * (_loc_pre + _wd_pre)
            )

            # (1) grad + prior must reconstruct the full step update.
            # step() computes d_p at _grad_pre's precision before
            # overall_coef promotion, so rounding error is at that scale.
            _step_eps = torch.finfo(_grad_pre.dtype).eps
            _decomp_scale = float((_scaled_grad.abs() + _combined_prior.abs()).max())
            _decomp_atol = max(_decomp_scale * _step_eps * 16, 1e-12)
            torch.testing.assert_close(
                (_scaled_grad + _combined_prior).float(),
                (_half_lr * d_p).float(),
                atol=_decomp_atol,
                rtol=0,
                msg=lambda s: f"Decomposition components don't match gradient update:\n{s}",
            )

            # (2) loc + wd must match the combined prior.
            # _loc_pre + _wd_pre sums at input precision regardless
            # of preconditioner promotion, so use p.grad.dtype.
            _input_eps = torch.finfo(p.grad.dtype).eps
            _dist_scale = float((_loc.abs() + _wd.abs()).max())
            _dist_atol = max(_dist_scale * _input_eps * 16, 1e-12)
            torch.testing.assert_close(
                (_loc + _wd).float(),
                _combined_prior.float(),
                atol=_dist_atol,
                rtol=0,
                msg=lambda s: f"Prior distribution error exceeds bound:\n{s}",
            )

        return _ComponentVectors(
            scaled_grad=_scaled_grad,
            unscaled_grad=_unscaled_grad,
            localization=_loc,
            weight_decay=_wd,
            noise=_noise,
        )

    def _accumulate_metrics(
        self,
        group: dict,
        components: _ComponentVectors,
        distance: torch.Tensor,
        preconditioning: PreconditionerCoefs,
        p: torch.Tensor,
    ) -> None:
        """Accumulate decomposed component vectors into per-group metrics."""
        group["metrics"].add_sum_squared_(
            scaled_grad=components.scaled_grad,
            unscaled_grad=components.unscaled_grad,
            localization=components.localization,
            weight_decay=components.weight_decay,
            noise=components.noise,
            distance=distance,
        )
        group["metrics"].add_dot_products_(
            scaled_grad=components.scaled_grad,
            prior=components.localization + components.weight_decay,
            noise=components.noise,
        )
        _precond = preconditioning.overall_coef
        if isinstance(_precond, torch.Tensor):
            group["metrics"].numel += int(_precond.count_nonzero())
        else:
            group["metrics"].numel += p.numel()

    def _scatter_sketches(self, components: _ComponentVectors, offset: int) -> None:
        """Scatter decomposed component vectors into the sketch buffer."""
        buf = self._sketch_buf
        sketch = self._sketch
        assert buf is not None and sketch is not None
        sketch.scatter_into_(buf.scaled_grad, components.scaled_grad, offset)
        sketch.scatter_into_(buf.unscaled_grad, components.unscaled_grad, offset)
        sketch.scatter_into_(buf.localization, components.localization, offset)
        sketch.scatter_into_(buf.weight_decay, components.weight_decay, offset)
        sketch.scatter_into_(buf.noise, components.noise, offset)

    def get_sketches(self) -> SketchBuffer:
        """Return a CPU snapshot of the current sketch buffer."""
        if not self.save_sketches:
            raise RuntimeError("Sketches not enabled.")
        assert self._sketch_buf is not None
        return SketchBuffer(
            **{
                q: getattr(self._sketch_buf, q).detach().cpu().clone()
                for q in SketchBuffer.QUANTITIES
            }
        )

    @torch.no_grad()
    def step(
        self, closure=None, noise_generator: Optional[torch.Generator] = None
    ) -> None:
        """Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            noise_generator (torch.Generator, optional): Generator for reproducible noise.

        Returns:
            Optional[float]: The loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        _need_decomposition = self.save_metrics or self.save_sketches
        param_offset = 0

        if self.save_sketches:
            assert self._sketch_buf is not None
            self._sketch_buf.zero_()

        for group_idx, group in enumerate(self.param_groups):
            # Metrics lifecycle: zero → accumulate per-param → sqrt (see Metrics docstring)
            if self.save_metrics:
                group["metrics"].zero_()

            prior = group["prior"]
            preconditioner = group["preconditioner"]
            params = group["params"]

            for i, p in enumerate(params):
                if p.grad is None:
                    if self.save_sketches:
                        # Advance past this param's region in the sketch's
                        # hash/sign index space so subsequent params stay aligned.
                        param_offset += p.numel()
                    continue

                state = self.state[p]

                # Get preconditioner coefficients
                preconditioning = preconditioner.get_coefficients(p, p.grad, state)

                # Gradient computation
                d_p = preconditioning.grad_coef * p.grad.mul(group["nbeta"])

                # Prior contribution — decompose into sub-priors when
                # tracking metrics or sketches so we can record localization
                # and weight_decay separately. Must happen before p.data
                # is modified below.
                loc_grad = None
                wd_grad = None
                if prior is not None:
                    if _need_decomposition:
                        loc_grad = torch.zeros_like(p)
                        wd_grad = torch.zeros_like(p)
                        sub_priors = (
                            prior.priors
                            if isinstance(prior, CompositePrior)
                            else [prior]
                        )
                        for sub_prior in sub_priors:
                            sub_grad = sub_prior.grad(p.data, state)
                            if (
                                isinstance(sub_prior, GaussianPrior)
                                and sub_prior.center is None
                            ):
                                wd_grad += sub_grad
                            else:
                                loc_grad += sub_grad
                        prior_grad = loc_grad + wd_grad
                    else:
                        prior_grad = prior.grad(p.data, state)
                    d_p.add_(preconditioning.prior_coef * prior_grad)

                d_p = preconditioning.overall_coef * d_p

                if self.save_metrics:
                    _distance = p.data - state["initial_param"]

                p.data.add_(d_p, alpha=-0.5 * group["lr"])

                # Noise addition
                noise = torch.normal(
                    mean=0.0,
                    std=group["noise_level"],
                    size=d_p.size(),
                    device=d_p.device,
                    generator=noise_generator,
                )
                noise = preconditioning.overall_coef * noise

                # Parameter updates
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

                if _need_decomposition:
                    components = self._decompose_update(
                        group, p, loc_grad, wd_grad, noise, preconditioning, d_p
                    )
                    if self.save_metrics:
                        self._accumulate_metrics(
                            group, components, _distance, preconditioning, p
                        )
                    if self.save_sketches:
                        self._scatter_sketches(components, param_offset)

                if self.save_sketches:
                    param_offset += p.numel()

            if self.save_metrics:
                # All params accumulated; convert sum-of-squares to L2 norms
                group["metrics"].sqrt_norms_()

        return loss

    def get_params(self) -> Iterator[torch.Tensor]:
        """Helper to get all parameters"""
        for group in self.param_groups:
            for p in group["params"]:
                yield p

    def iter_group_metrics(self) -> Iterator[Metrics]:
        """Yield metrics for each param group."""
        if not self.save_metrics:
            raise RuntimeError("Metrics not enabled. Set save_metrics=True.")
        for group in self.param_groups:
            yield group["metrics"]

    def get_metrics(self) -> Metrics:
        """Aggregate metrics across all param groups into a single CPU Metrics."""
        return Metrics.aggregate(self.iter_group_metrics())

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
        mask=None,
        save_metrics: bool = False,
        sketch_dim: Optional[int] = None,
        sketch_seed: int = 0,
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
        :param mask: Boolean mask for restricting updatable parameters (default: None)
        :param save_metrics: Whether to track metrics during training (default: False)
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
        priors = []
        if weight_decay > 0:
            priors.append(GaussianPrior(localization=weight_decay, center=None))
        if localization > 0:
            priors.append(GaussianPrior(localization=localization, center="initial"))

        prior = CompositePrior(priors)

        instance = cls(
            params,
            lr=lr,
            noise_level=noise_level,
            nbeta=nbeta,
            prior=prior,
            bounding_box_size=bounding_box_size,
            mask=mask,
            save_metrics=save_metrics,
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
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
        save_metrics: bool = False,
        sketch_dim: Optional[int] = None,
        sketch_seed: int = 0,
    ):
        """Factory method to create an SGMCMC instance that matches SGNHT's interface.

        This allows SGMCMC to be used as a drop-in replacement for SGNHT.

        :param params: Iterable of parameters to optimize
        :param lr: Learning rate (default: 0.01)
        :param diffusion_factor: Diffusion factor (default: 0.01)
        :param nbeta: Inverse temperature (default: 1.0)
        :param bounding_box_size: Size of bounding box around initial parameters (default: None)
        :param save_metrics: Whether to track metrics (default: False)
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
            save_metrics=save_metrics,
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
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
        mask=None,
        save_metrics: bool = False,
        sketch_dim: Optional[int] = None,
        sketch_seed: int = 0,
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
        :param mask: Boolean mask for restricting updatable parameters (default: None)
        :param save_metrics: Whether to track metrics during training (default: False)
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

        priors = []
        if weight_decay > 0:
            priors.append(GaussianPrior(localization=weight_decay, center=None))
        if localization > 0:
            priors.append(GaussianPrior(localization=localization, center="initial"))

        prior = CompositePrior(priors)

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
            preconditioner="rmsprop",
            preconditioner_kwargs=preconditioner_kwargs,
            bounding_box_size=bounding_box_size,
            mask=mask,
            save_metrics=save_metrics,
            sketch_dim=sketch_dim,
            sketch_seed=sketch_seed,
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
