"""Weight restrictions for selecting subsets of model parameters.

Parses restriction strings like "full", "l0", "l0h1", "l0g0", "l0 attn",
"l0 mlp", "embed", "unembed" and returns ParamMasks suitable for the SGLD sampler.

Supports HuggingFace models (via auto-detected model_type) and TransformerLens
HookedTransformer models, using the structure data in _model_structures.py.
"""

from __future__ import annotations

import re

import torch

from devinterp.slt._model_structures import HEAD_STRUCTURES
from devinterp.slt.sampler import ParamMasks, is_transformer_lens_model


def create_param_masks(
    model: torch.nn.Module,
    restriction: str | list[str],
) -> ParamMasks:
    """Create parameter masks from a restriction string.

    Args:
        model: PyTorch model (HuggingFace or TransformerLens).
        restriction: Which parameters to select. Options:
            - "full": all parameters, no masks
            - "l0": all params in layer 0
            - "l0h1": layer 0, head 1 (with per-element masks)
            - "l0g0": layer 0, group 0 (GQA group: Q heads sharing a KV head)
            - "l0 attn": layer 0 attention params
            - "l0 mlp": layer 0 MLP params
            - "embed": embedding params
            - "unembed": unembedding params
            - list of the above: union (masks are ORed for shared params)

    Returns:
        Dict mapping parameter names to mask tensors (or None for unrestricted).
    """
    if not isinstance(restriction, (str, list)):
        raise TypeError(
            f"restriction must be a str or list[str], got {type(restriction).__name__}"
        )

    if isinstance(restriction, list):
        if "full" in restriction:
            raise ValueError("Cannot combine 'full' with other restrictions.")
        masks_list = [create_param_masks(model, r) for r in restriction]
        return _merge_param_masks(masks_list)

    restriction = restriction.strip()
    if restriction == "full":
        return {name: None for name, _ in model.named_parameters()}

    spec = _get_spec(model)
    params = dict(model.named_parameters())

    r = restriction.lower()

    if r == "embed":
        return _select_component(params, spec, "embed")
    if r == "unembed":
        return _select_component(params, spec, "unembed")

    # l0h1, L0H1, l0 h1
    m = re.match(r"l(\d+)\s*h(\d+)$", r)
    if m:
        layer, head = int(m[1]), int(m[2])
        n_layers = _count_layers(params, spec)
        if layer >= n_layers:
            raise ValueError(
                f"Layer {layer} out of range (model has {n_layers} layers)."
            )
        n_heads, _, _ = _get_model_config(model)
        if head >= n_heads:
            raise ValueError(f"Head {head} out of range (model has {n_heads} heads).")
        return _head_masks(model, spec, params, layer, head)

    # l0g0, L0G0, l0 g0 — GQA group (Q heads sharing a KV head)
    m = re.match(r"l(\d+)\s*g(\d+)$", r)
    if m:
        layer, group = int(m[1]), int(m[2])
        n_layers = _count_layers(params, spec)
        if layer >= n_layers:
            raise ValueError(
                f"Layer {layer} out of range (model has {n_layers} layers)."
            )
        _, _, n_kv = _get_model_config(model)
        if group >= n_kv:
            raise ValueError(
                f"Group {group} out of range (model has {n_kv} KV heads/groups)."
            )
        return _group_masks(model, spec, params, layer, group)

    # l0 attn, l0 mlp, l0
    m = re.match(r"l(\d+)(?:\s+(attn|mlp))?$", r)
    if m:
        layer = int(m[1])
        component = m[2]
        n_layers = _count_layers(params, spec)
        if layer >= n_layers:
            raise ValueError(
                f"Layer {layer} out of range (model has {n_layers} layers)."
            )
        return _layer_masks(params, spec, layer, component)

    # Direct state dict key fallback
    if restriction in params:
        return {restriction: None}

    raise ValueError(
        f"Unknown restriction '{restriction}'. "
        f"Valid patterns: 'full', 'l0', 'l0h1', 'l0g0', 'l0 attn', 'l0 mlp', 'embed', 'unembed', "
        f"or a direct parameter name."
    )


def _get_spec(model: torch.nn.Module) -> dict:
    """Look up the model's structure spec from HEAD_STRUCTURES.

    If the entry is a list of variants, pick the one that best matches the
    model's actual parameters (most layer-scoped entries found).
    """
    if is_transformer_lens_model(model):
        model_type = "hooked_transformer"
    else:
        model_type = getattr(getattr(model, "config", None), "model_type", None)

    if model_type is None:
        raise ValueError(
            f"Cannot detect model type for {model.__class__.__name__}. "
            f"Expected model.config.model_type or TransformerLens model."
        )
    if model_type not in HEAD_STRUCTURES:
        raise ValueError(
            f"Model type '{model_type}' not in HEAD_STRUCTURES. "
            f"Supported: {', '.join(sorted(HEAD_STRUCTURES))}"
        )

    entry = HEAD_STRUCTURES[model_type]
    if isinstance(entry, dict):
        return entry

    # List of variants: pick the best match against actual params
    params = dict(model.named_parameters())
    best_spec, best_score = entry[0], -1
    for variant in entry:
        prefixes = variant.get("layer_prefixes", [])
        suffixes = (
            set(variant.get("attn", []))
            | set(variant.get("mlp", []))
            | set(variant.get("head_params", {}).keys())
        )
        score = sum(
            1
            for name in params
            if any(
                name.startswith(f"{p}.") and name.endswith(s)
                for p in prefixes
                for s in suffixes
            )
        )
        if score > best_score:
            best_spec, best_score = variant, score
    return best_spec


def _count_layers(params: dict[str, torch.nn.Parameter], spec: dict) -> int:
    """Count the number of layers by checking which layer prefixes exist."""
    prefix_base = spec["layer_prefixes"][0]
    layer = 0
    while any(name.startswith(f"{prefix_base}.{layer}.") for name in params):
        layer += 1
    return layer


def _get_model_config(model: torch.nn.Module) -> tuple[int, int, int]:
    """Extract n_heads, d_model, n_kv from model config."""
    if is_transformer_lens_model(model):
        cfg = model.cfg  # pyright: ignore[reportAttributeAccessIssue]
        n_heads: int = cfg.n_heads
        d_model: int = cfg.d_model
        n_kv: int = getattr(cfg, "n_key_value_heads", None) or n_heads
    else:
        cfg = model.config  # pyright: ignore[reportAttributeAccessIssue]
        n_heads = cfg.num_attention_heads
        d_model = cfg.hidden_size
        n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    return n_heads, d_model, n_kv


def _select_component(
    params: dict[str, torch.nn.Parameter],
    spec: dict,
    component: str,
) -> ParamMasks:
    """Select all params matching a component list (embed, unembed, attn, mlp)."""
    keys = spec.get(component, [])
    return {name: None for name in params if any(name.endswith(k) for k in keys)}


def _layer_masks(
    params: dict[str, torch.nn.Parameter],
    spec: dict,
    layer: int,
    component: str | None,
) -> ParamMasks:
    """Select all params in a layer, optionally filtered to attn or mlp."""
    prefix = f"{spec['layer_prefixes'][0]}.{layer}."
    if component is None:
        # Bare layer: select all params in the layer
        return {name: None for name in params if name.startswith(prefix)}

    suffixes = spec.get(component, [])
    return {
        name: None
        for name in params
        if name.startswith(prefix) and any(name.endswith(s) for s in suffixes)
    }


def _head_masks(
    model: torch.nn.Module,
    spec: dict,
    params: dict[str, torch.nn.Parameter],
    layer: int,
    head: int,
) -> ParamMasks:
    """Build per-element boolean masks for a single attention head."""
    n_heads, d_model, n_kv = _get_model_config(model)
    hd = d_model // n_heads
    kv_head = head * n_kv // n_heads
    prefix = f"{spec['layer_prefixes'][0]}.{layer}."

    head_params = spec.get("head_params", {})

    masks: ParamMasks = {}
    for full_name, param in params.items():
        if not full_name.startswith(prefix):
            continue
        short_name = full_name[len(prefix) :]
        if short_name not in head_params:
            continue

        head_spec = head_params[short_name]
        dim = head_spec["slice_dim"]
        fused = head_spec.get("fused")
        axis_size = param.shape[0] if param.dim() == 1 else param.shape[dim]

        mask = torch.zeros(axis_size, dtype=torch.bool)

        if fused == "interleaved":
            # GPT-NeoX layout: (n_heads, qkv=3, head_dim) flattened
            mask.view(n_heads, 3, hd)[head] = True
        elif fused == "concat":
            # GPT-2 / Falcon layout: Q then K then V concatenated
            q_size = n_heads * hd
            kv_size = (axis_size - q_size) // 2
            q, kv = mask.split([q_size, 2 * kv_size])
            q.view(n_heads, hd)[head] = True
            kv.view(2, n_kv, hd)[:, kv_head] = True
        elif axis_size in (n_kv * hd, n_kv):
            # Separate K/V projection for GQA models
            mask.view(n_kv, -1)[kv_head] = True
        else:
            # Separate Q projection or O projection
            mask.view(n_heads, -1)[head] = True

        # Expand 1D mask to match param shape
        if param.dim() > 1:
            shape = [1] * param.dim()
            shape[dim] = -1
            mask = mask.view(*shape).expand_as(param)

        masks[full_name] = mask

    return masks


def _group_masks(
    model: torch.nn.Module,
    spec: dict,
    params: dict[str, torch.nn.Parameter],
    layer: int,
    group: int,
) -> ParamMasks:
    """Build per-element boolean masks for a GQA group.

    A group contains n_heads // n_kv consecutive Q heads sharing one KV head.
    group index ranges from 0 to n_kv - 1. For MHA models (n_kv == n_heads),
    this is equivalent to a single head.
    """
    n_heads, d_model, n_kv = _get_model_config(model)
    hd = d_model // n_heads
    prefix = f"{spec['layer_prefixes'][0]}.{layer}."

    head_params = spec.get("head_params", {})

    masks: ParamMasks = {}
    for full_name, param in params.items():
        if not full_name.startswith(prefix):
            continue
        short_name = full_name[len(prefix) :]
        if short_name not in head_params:
            continue

        head_spec = head_params[short_name]
        dim = head_spec["slice_dim"]
        fused = head_spec.get("fused")
        axis_size = param.shape[0] if param.dim() == 1 else param.shape[dim]

        mask = torch.zeros(axis_size, dtype=torch.bool)

        if fused == "interleaved":
            # GPT-NeoX: (n_heads, qkv=3, head_dim) — group = contiguous Q heads + shared KV
            mask.view(n_kv, -1)[group] = True
        elif fused == "concat":
            # Q then K then V concatenated
            q_size = n_heads * hd
            kv_size = (axis_size - q_size) // 2
            q, kv = mask.split([q_size, 2 * kv_size])
            q.view(n_kv, n_heads // n_kv, hd)[group] = True
            kv.view(2, n_kv, hd)[:, group] = True
        else:
            # Separate projections: reshape as (n_kv, ...) and select group
            mask.view(n_kv, -1)[group] = True

        # Expand 1D mask to match param shape
        if param.dim() > 1:
            shape = [1] * param.dim()
            shape[dim] = -1
            mask = mask.view(*shape).expand_as(param)

        masks[full_name] = mask

    return masks


def _merge_param_masks(masks_list: list[ParamMasks]) -> ParamMasks:
    """Merge multiple ParamMasks by ORing masks for shared parameters."""
    if len(masks_list) == 1:
        return masks_list[0]

    # Preserve model parameter order: use dict.fromkeys across all inputs
    all_names: dict[str, None] = {}
    for m in masks_list:
        all_names.update(dict.fromkeys(m))

    merged: ParamMasks = {}
    for name in all_names:
        entries = [m[name] for m in masks_list if name in m]
        if any(e is None for e in entries):
            merged[name] = None
        elif len(entries) == 1:
            merged[name] = entries[0]
        else:
            merged[name] = entries[0]
            for e in entries[1:]:
                merged[name] = torch.logical_or(merged[name], e)

    return merged


def preview_weight_restriction(
    model: torch.nn.Module,
    masks: ParamMasks,
    *,
    plain: bool = False,
) -> None:
    """Print a tree of which parameters are selected by a mask dict.

    Shows how a weight restriction maps to actual state_dict keys, with a
    per-key selected/total count. Useful for debugging ambiguous selection
    patterns (e.g. "does 'unembed' include the layer norm?").

    Args:
        model: PyTorch model. Used to look up parameter sizes for the
            "selected of total" counts.
        masks: Dict mapping param name to mask tensor (or None for unrestricted).
            Typically from `create_param_masks` or user-built.
        plain: If True, print just one param name per line (useful for piping).
    """
    from collections import defaultdict

    if plain:
        for name in sorted(masks.keys()):
            print(name)
        return

    # ANSI colors (no extra dependency).
    CYAN, YELLOW, RESET = "\033[36m", "\033[33m", "\033[0m"

    def _bar(percent: float, width: int = 24) -> str:
        filled = int(percent / 100 * width)
        return "\u25b0" * filled + "\u25b1" * (width - filled)

    stats: list[tuple[str, int, int]] = []
    for name, mask in masks.items():
        total = model.get_parameter(name).numel()
        selected = total if mask is None else int(mask.int().sum().item())
        stats.append((name, selected, total))

    grouped: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    for name, selected, total in stats:
        parts = name.split(".")
        prefix = ".".join(parts[:-1]) if len(parts) > 1 else ""
        suffix = parts[-1]
        grouped[prefix].append((suffix, selected, total))

    for prefix in sorted(grouped):
        items = grouped[prefix]
        if prefix:
            print(f"\n{CYAN}{prefix}{RESET}")
        for i, (suffix, selected, total) in enumerate(items):
            is_last = i == len(items) - 1
            branch = (
                "" if not prefix else ("\u2514\u2500" if is_last else "\u251c\u2500")
            )
            percent = (selected / total) * 100 if total > 0 else 0.0
            indent = "  " if prefix else ""
            print(
                f"{indent}{CYAN}{branch}{RESET} "
                f"{suffix:35} "
                f"{YELLOW}{_bar(percent)}{RESET}  "
                f"{YELLOW}{percent:5.1f}%{RESET}  "
                f"({selected:,} of {total:,})"
            )
