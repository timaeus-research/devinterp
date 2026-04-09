import warnings
from typing import Callable, Iterable, Iterator, NamedTuple, Optional, Union

import torch

from .metrics import Metrics
from .sketch import CountSketch, SketchBuffer


class _ComponentVectors(NamedTuple):
    """Post-masked component vectors from a single SGLD parameter update."""

    # Duplicated from SGMCMC deliberately: SGLD is deprecated and will be
    # removed, so coupling it to the replacement via a shared type would make
    # that cleanup harder. Also the semantics differ slightly in the way masks
    # are applied.

    scaled_grad: torch.Tensor
    unscaled_grad: torch.Tensor
    localization: torch.Tensor
    weight_decay: torch.Tensor
    noise: torch.Tensor


class SGLD(torch.optim.Optimizer):
    r"""
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer blends Stochastic Gradient Descent (SGD) with Langevin Dynamics,
    introducing Gaussian noise to the gradient updates. This makes it sample weights from the posterior distribution, instead of optimizing weights.

    This implementation follows Lau et al.'s (2023) implementation, which is a modification of
    Welling and Teh (2011) that omits the learning rate schedule and introduces
    an localization term that pulls the weights towards their initial values.

    The equation for the update is as follows:

    $$\Delta w_t = \frac{\epsilon}{2}\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right)+\gamma\left(w_0-w_t\right) - \lambda w_t\right) + N(0, \epsilon\sigma^2)$$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of training samples, $m$ is the batch size, $\gamma$ is
    the localization strength, $\lambda$ is the weight decay strength,
    and $\sigma$ is the noise term.

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=0.1, nbeta=utils.default_nbeta(dataloader))

        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. |colab6| image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb

    Note:
        - :python:`localization` is unique to this class and serves to guide the weights towards their original values. This is useful for estimating quantities over the local posterior.
        - :python:`noise_level` is not intended to be changed, except when testing! Doing so will raise a warning.
        - Although this class is a subclass of :python:`torch.optim.Optimizer`, this is a bit of a misnomer in this case. It's not used for optimizing in LLC estimation, but rather for sampling from the posterior distribution around a point.
        - Hyperparameter optimization is more of an art than a science. Check out `the calibration notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb>`_ |colab6| for how to go about it in a simple case.
    :param params: Iterable of parameters to optimize or dicts defining parameter groups. Either :python:`model.parameters()` or something more fancy, just like other :python:`torch.optim.Optimizer` classes.
    :type params: Iterable
    :param lr: Learning rate $\epsilon$. Default is 0.01
    :type lr: float, optional
    :param noise_level: Amount of Gaussian noise $\sigma$ introduced into gradient updates. Don't change this unless you know very well what you're doing! Default is 1
    :type noise_level: float, optional
    :param weight_decay: L2 regularization term $\lambda$, applied as weight decay. Default is 0
    :type weight_decay: float, optional
    :param localization: Strength of the force $\gamma$ pulling weights back to their initial values. Default is 0
    :type localization: float, optional
    :param nbeta: Inverse reparameterized temperature (otherwise known as n*beta or ~beta), float (default: 1., set to utils.default_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: float or Callable, optional
    :param bounding_box_size: the size of the bounding box enclosing our trajectory in parameter space. Default is None, in which case no bounding box is used.
    :type bounding_box_size: float, optional
    :param save_metrics: Whether to track metrics (scaled_grad, localization, weight_decay, noise norms) during optimization. Use :meth:`get_metrics` to retrieve. Default is False
    :type save_metrics: bool, optional

    :raises Warning: if :python:`noise_level` is set to anything other than 1
    :raises Warning: if :python:`nbeta` is set to 1
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 0.01,
        noise_level: float = 1.0,
        weight_decay: float = 0.0,
        localization: float = 0.0,
        nbeta: Union[Callable[[], float], float] = 1.0,
        bounding_box_size: Optional[float] = None,
        save_metrics: bool = False,
        sketch_dim: Optional[int] = None,
        sketch_seed: int = 0,
    ):
        warnings.warn(
            "SGLD has been deprecated. Please use SGMCMC.sgld instead.",
            DeprecationWarning,
        )

        self.save_metrics = save_metrics
        self.save_sketches = sketch_dim is not None

        if noise_level != 1.0:
            warnings.warn(
                "Warning: noise_level in SGLD is unequal to one, this removes SGLD posterior sampling guarantees."
            )
        if nbeta == 1.0:
            warnings.warn(
                "Warning: nbeta set to 1, LLC estimates will be off unless you know what you're doing. Use utils.default_nbeta(dataloader) instead"
            )
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            localization=localization,
            nbeta=nbeta,
            bounding_box_size=bounding_box_size,
        )

        # In torch.optim.Optimizer, the parameters are stored in a list of dictionaries.
        # defaults holds the default values for the optimizer parameters.
        super(SGLD, self).__init__(params, defaults)

        # Save the initial parameters if the localization term is set
        for group in self.param_groups:
            group["num_el"] = 0

            # Validate mask shape if present
            if group.get("mask") is not None:
                for p in group["params"]:
                    mask = group["mask"]
                    if isinstance(mask, torch.Tensor) and mask.shape != p.shape:
                        raise ValueError(
                            f"Mask shape {mask.shape} does not match parameter shape {p.shape}. "
                            "Scalar masks are not supported."
                        )

            store_initial = (
                group["localization"] != 0
                or group["bounding_box_size"] != 0
                or self.save_metrics
                or self.save_sketches
            )
            if store_initial:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["initial_param"] = p.data.clone().detach()
                    group["num_el"] += p.numel()

            if self.save_metrics:
                device = next(iter(group["params"])).device
                group["metrics"] = Metrics().to(device)

        if self.save_sketches:
            assert sketch_dim is not None
            total_numel = sum(
                p.numel() for group in self.param_groups for p in group["params"]
            )
            device = next(iter(self.param_groups[0]["params"])).device
            self._sketch = CountSketch.create(total_numel, sketch_dim, sketch_seed).to(
                device
            )
            self._sketch_buf = SketchBuffer.create(sketch_dim, device)
        else:
            self._sketch = None
            self._sketch_buf = None

    def step(self, noise_generator: Optional[torch.Generator] = None) -> None:
        """
        Perform a single SGLD optimization step.
        """
        with torch.no_grad():
            _need_decomposition = self.save_metrics or self.save_sketches
            param_offset = 0

            if self.save_sketches:
                assert self._sketch_buf is not None
                self._sketch_buf.zero_()

            for group_idx, group in enumerate(self.param_groups):
                # Metrics lifecycle: zero → accumulate per-param → sqrt (see Metrics docstring)
                if self.save_metrics:
                    group["metrics"].zero_()

                for p in group["params"]:
                    param_state = self.state[p]

                    # Gradients are None if the parameter is not trainable
                    # We'll denote the gradient of the loss with respect to this param group (p) as dw
                    if p.grad is None:
                        dw = torch.zeros_like(p.data)
                    else:
                        dw = p.grad.data * group["nbeta"]

                    # Weight decay
                    if group["weight_decay"] != 0:
                        dw.add_(
                            p.data, alpha=group["weight_decay"]
                        )  # inplace addition. Effectively, dw = dw + p.data * group["weight_decay"]

                    # Here, group["localization"] is the localization strength $\gamma$ (a single float). If it's 0, we don't do anything.
                    initial_param = self.state[p]["initial_param"]
                    initial_param_distance = p.data - initial_param
                    if group["localization"] != 0:
                        dw.add_(initial_param_distance, alpha=group["localization"])

                    # Add Gaussian noise
                    noise = torch.normal(
                        mean=0.0,
                        std=group["noise_level"],
                        size=dw.size(),
                        device=dw.device,
                        generator=noise_generator,
                    )

                    if group.get("mask") is not None:
                        # Restrict the noise and gradient to the subset of parameters we're optimizing over.
                        dw = dw * group["mask"]
                        noise = noise * group["mask"]

                    if _need_decomposition:
                        components = self._decompose_update(
                            group, p, dw, initial_param_distance, noise
                        )
                        if self.save_metrics:
                            self._accumulate_metrics(
                                group, p, components, initial_param_distance
                            )
                        if self.save_sketches:
                            self._scatter_sketches(components, param_offset)

                    # Update parameters
                    p.data.add_(dw, alpha=-0.5 * group["lr"])
                    p.data.add_(
                        noise, alpha=group["lr"] ** 0.5
                    )  # Scale noise by sqrt(lr)

                    if self.save_sketches:
                        param_offset += p.numel()

                    # Rebound if exceeded bounding box size
                    if group["bounding_box_size"]:
                        torch.clamp_(
                            p.data,
                            min=param_state["initial_param"]
                            - group["bounding_box_size"],
                            max=param_state["initial_param"]
                            + group["bounding_box_size"],
                        )

                if self.save_metrics:
                    # All params accumulated; convert sum-of-squares to L2 norms
                    group["metrics"].sqrt_norms_()

    def _decompose_update(
        self,
        group: dict,
        p: torch.Tensor,
        dw: torch.Tensor,
        initial_param_distance: torch.Tensor,
        noise: torch.Tensor,
    ) -> _ComponentVectors:
        """Decompose an SGLD parameter update into post-masked component vectors.

        Shared by both metrics accumulation and sketch scattering. SGLD has no
        preconditioner, so unscaled_grad == scaled_grad. Debug assertions verify
        the decomposition matches the actual update.
        """
        _lr = group["lr"]
        _mask = group.get("mask")
        _half_lr = 0.5 * _lr

        # Multiplication order must match step() to avoid bfloat16 associativity errors:
        #   step() builds dw as: (p.grad * nbeta) + (p.data * wd) + (dist * loc)
        raw_grad = p.grad.data if p.grad is not None else torch.zeros_like(p.data)
        _scaled_grad = _half_lr * raw_grad.mul(group["nbeta"])
        _noise_vec = noise * (_lr**0.5)
        _loc = _half_lr * initial_param_distance.mul(group["localization"])
        _wd = _half_lr * p.data.mul(group["weight_decay"])

        if _mask is not None:
            _scaled_grad = _scaled_grad * _mask
            _loc = _loc * _mask
            _wd = _wd * _mask
            _noise_vec = _noise_vec * _mask

        # Sanity-check: reconstructed components must sum to the actual update.
        # step() builds dw via in-place add_ (which may use FMA on GPU),
        # then we multiply by half_lr once. Here we multiply half_lr into
        # each component separately and sum — distributing the scalar
        # breaks IEEE 754 associativity. The rounding gap scales with the
        # component magnitudes and the dtype's epsilon (significant for
        # bfloat16 models where eps ≈ 0.004). Using the sum of absolute
        # values as the scale handles catastrophic cancellation, where
        # large components nearly cancel leaving a small residual whose
        # relative error would otherwise look enormous.
        if __debug__:
            _step_eps = torch.finfo(dw.dtype).eps
            _decomp_scale = float((_scaled_grad.abs() + _loc.abs() + _wd.abs()).max())
            _decomp_atol = max(_decomp_scale * _step_eps * 16, 1e-12)
            torch.testing.assert_close(
                (_scaled_grad + _loc + _wd).float(),
                (_half_lr * dw).float(),
                atol=_decomp_atol,
                rtol=0,
                msg=lambda s: f"Decomposition components don't match gradient update:\n{s}",
            )

        return _ComponentVectors(
            scaled_grad=_scaled_grad,
            unscaled_grad=_scaled_grad,
            localization=_loc,
            weight_decay=_wd,
            noise=_noise_vec,
        )

    def _accumulate_metrics(
        self,
        group: dict,
        p: torch.Tensor,
        components: _ComponentVectors,
        initial_param_distance: torch.Tensor,
    ) -> None:
        """Accumulate decomposed component vectors into per-group metrics."""
        _mask = group.get("mask")
        if _mask is not None:
            initial_param_distance = initial_param_distance * _mask
            numel = p[_mask.bool()].numel()
        else:
            numel = p.numel()

        group["metrics"].add_sum_squared_(
            scaled_grad=components.scaled_grad,
            unscaled_grad=components.unscaled_grad,
            localization=components.localization,
            weight_decay=components.weight_decay,
            noise=components.noise,
            distance=initial_param_distance,
        )
        group["metrics"].add_dot_products_(
            scaled_grad=components.scaled_grad,
            prior=components.localization + components.weight_decay,
            noise=components.noise,
        )
        group["metrics"].numel += numel

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

    def iter_group_metrics(self) -> Iterator[Metrics]:
        """Yield metrics for each param group."""
        if not self.save_metrics:
            raise RuntimeError("Metrics not enabled. Set save_metrics=True.")
        for group in self.param_groups:
            yield group["metrics"]

    def get_metrics(self) -> Metrics:
        """Aggregate metrics across all param groups into a single CPU Metrics."""
        return Metrics.aggregate(self.iter_group_metrics())
