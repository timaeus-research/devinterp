"""Validate _model_structures entries against real (tiny) HuggingFace models.

For each model type that AutoConfig supports:
  1. Check that listed attn/mlp/embed/unembed params exist in the model
  2. Check that head_params entries exist and produce correct mask shapes
  3. Dead-head gradient test: zeroing the O projection for a head/group
     should kill gradients on that head/group's masked elements
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from devinterp.slt._model_structures import HEAD_STRUCTURES

NUM_HEADS = 2
HEAD_DIM = 8
HIDDEN = NUM_HEADS * HEAD_DIM


def _make_tiny(model_type: str, gqa: bool = False) -> AutoConfig:
    """Wrapper around johan.tiny_configs.make_tiny, imported lazily."""
    from devinterp.slt._tiny_configs import make_tiny

    return make_tiny(model_type, gqa=gqa)


def _testable_model_types():
    """Return model types that AutoConfig can create (skip unrecognized ones)."""
    types = []
    for mt in sorted(HEAD_STRUCTURES):
        if mt == "hooked_transformer":
            continue
        try:
            _make_tiny(mt, gqa=True)
            types.append(mt)
        except (ValueError, KeyError):
            pass
    return types


TESTABLE = _testable_model_types()


def _layer0_params(model, spec):
    """Extract layer-0 parameters as {short_name: param}."""
    prefix = f"{spec['layer_prefixes'][0]}.0."
    return {
        name[len(prefix) :]: param
        for name, param in model.named_parameters()
        if name.startswith(prefix)
    }


def _resolve_spec(model_type, all_names):
    """Resolve HEAD_STRUCTURES entry, picking best variant if it's a list."""
    entry = HEAD_STRUCTURES[model_type]
    if isinstance(entry, dict):
        return entry

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
            for name in all_names
            if any(
                name.startswith(f"{p}.") and name.endswith(s)
                for p in prefixes
                for s in suffixes
            )
        )
        if score > best_score:
            best_spec, best_score = variant, score
    return best_spec


# --- Test 1: all listed params exist in the model ---


@pytest.mark.parametrize("model_type", TESTABLE)
def test_listed_params_exist(model_type):
    """Every param name in attn/mlp/embed/unembed/head_params should exist in the model.

    Mirrors johan/test_yaml_entries_exist.py: constructs full param names using
    layer_prefixes and checks against named_parameters | state_dict (catching tied weights).
    Also verifies that layer indices beyond the last don't exist (prefix sanity).
    """
    config = _make_tiny(model_type)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    all_names = {n for n, _ in model.named_parameters()} | set(
        model.state_dict().keys()
    )
    spec = _resolve_spec(model_type, all_names)
    n_layers = config.num_hidden_layers
    prefixes = spec["layer_prefixes"]

    # Collect all layer-scoped short names
    layer_shorts = set(spec.get("attn", []) + spec.get("mlp", []))
    layer_shorts |= set(spec.get("head_params", {}).keys())
    layer_shorts |= set(spec.get("other", []))

    missing = []
    should_miss = []

    for short in layer_shorts:
        found = any(
            f"{prefix}.{i}.{short}" in all_names
            for prefix in prefixes
            for i in range(n_layers)
        )
        if not found:
            missing.append(short)

        # Layer beyond last should NOT exist (verifies prefix is correct)
        for prefix in prefixes:
            bogus = f"{prefix}.{n_layers}.{short}"
            if bogus in all_names:
                should_miss.append(bogus)

    # Check global params (embed, unembed)
    for name in spec.get("embed", []) + spec.get("unembed", []):
        if name not in all_names:
            missing.append(name)

    assert not missing, f"{model_type}: missing params: {missing}"
    assert not should_miss, f"{model_type}: found beyond last layer: {should_miss}"


# --- Test 2: dead-head gradient test ---


O_LEAF_NAMES = {
    "o_proj",
    "out_proj",
    "c_proj",
    "dense",
    "o_net",
    "out_lin",
    "attention_out",
}
STRONG_O_NAMES = {"o_proj", "out_proj", "out_lin", "o_net", "attention_out"}
ATTN_KEYWORDS = ["attn", "attention", "self_att", "rel_attn", "multi_head"]


def _is_o_param(short):
    base = short.removesuffix(".weight").removesuffix(".bias")
    leaf = base.split(".")[-1]
    if leaf in STRONG_O_NAMES:
        return True
    if leaf in O_LEAF_NAMES and any(kw in base.lower() for kw in ATTN_KEYWORDS):
        return True
    if leaf == "o" and any(kw in base.lower() for kw in ATTN_KEYWORDS):
        return True
    return False


def _find_o_projection(head_params_spec, l0_params):
    """Find the O projection: unfused param with opposite slice_dim."""
    for name in l0_params:
        if name in head_params_spec and "fused" not in head_params_spec[name]:
            if _is_o_param(name):
                return name
    return None


def _infer_n_kv(l0, structure, n_heads, hd):
    for name, spec in structure.items():
        if name not in l0:
            continue
        param = l0[name]
        fused = spec.get("fused")
        if fused == "concat":
            dim = spec["slice_dim"]
            total = param.shape[0] if param.dim() == 1 else param.shape[dim]
            kv_size = (total - n_heads * hd) // 2
            return kv_size // hd
        elif fused == "interleaved":
            return n_heads
    for name, spec in structure.items():
        if name not in l0:
            continue
        if "k_proj" in name or "key" in name:
            param = l0[name]
            dim = spec["slice_dim"]
            axis_size = param.shape[0] if param.dim() == 1 else param.shape[dim]
            return axis_size // hd
    return n_heads


def _build_mask(param, spec, index, n_heads, n_kv, hd, is_group):
    """Build a 1D boolean mask for a single head or group."""
    dim = spec["slice_dim"]
    fused = spec.get("fused")
    axis_size = param.shape[0] if param.dim() == 1 else param.shape[dim]

    mask = torch.zeros(axis_size, dtype=torch.bool)

    if is_group:
        if fused == "concat":
            q_size = n_heads * hd
            kv_size = (axis_size - q_size) // 2
            q, kv = mask.split([q_size, 2 * kv_size])
            q.view(n_kv, n_heads // n_kv, hd)[index] = True
            kv.view(2, n_kv, hd)[:, index] = True
        else:
            mask.view(n_kv, -1)[index] = True
    else:
        kv_idx = index * n_kv // n_heads
        if fused == "interleaved":
            mask.view(n_heads, 3, hd)[index] = True
        elif fused == "concat":
            q_size = n_heads * hd
            kv_size = (axis_size - q_size) // 2
            q, kv = mask.split([q_size, 2 * kv_size])
            q.view(n_heads, hd)[index] = True
            kv.view(2, n_kv, hd)[:, kv_idx] = True
        elif axis_size in (n_kv * hd, n_kv):
            mask.view(n_kv, -1)[kv_idx] = True
        else:
            mask.view(n_heads, -1)[index] = True

    if param.dim() > 1:
        shape = [1] * param.dim()
        shape[dim] = -1
        mask = mask.view(*shape).expand_as(param)

    return mask


def _backward(model, input_ids):
    model.zero_grad()
    if model.config.model_type == "xmod":
        model.set_default_language("en_XX")
    model(input_ids).logits.sum().backward()


@pytest.mark.parametrize("model_type", TESTABLE)
def test_dead_head_gradient(model_type):
    """Zeroing a head's O projection should kill gradients on that head's params."""
    config = _make_tiny(model_type, gqa=True)
    model = AutoModelForCausalLM.from_config(config)
    all_names = {n for n, _ in model.named_parameters()}
    spec = _resolve_spec(model_type, all_names)
    head_params_spec = spec.get("head_params", {})

    n_heads = config.num_attention_heads
    hd = config.hidden_size // n_heads
    input_ids = torch.randint(0, config.vocab_size, (8, 8))

    l0 = _layer0_params(model, spec)
    n_kv = _infer_n_kv(l0, head_params_spec, n_heads, hd)

    o_name = _find_o_projection(head_params_spec, l0)
    assert o_name is not None, (
        f"{model_type}: no O projection found in {list(head_params_spec.keys())}"
    )

    orig_state = {k: v.clone() for k, v in model.state_dict().items()}

    def restore():
        model.load_state_dict(orig_state)

    hi = n_heads - 1
    gi = hi * n_kv // n_heads

    def has_grad(param, mask):
        return param.grad[mask].abs().sum() > 0

    # --- Head test ---
    head_masks = {
        name: _build_mask(
            l0[name], head_params_spec[name], hi, n_heads, n_kv, hd, False
        )
        for name in head_params_spec
        if name in l0
    }
    # Also build masks for a second head to find unique elements
    hi2 = n_heads - 2
    head_masks_2 = {
        name: _build_mask(
            l0[name], head_params_spec[name], hi2, n_heads, n_kv, hd, False
        )
        for name in head_params_spec
        if name in l0
    }

    # Head alive: gradients should be nonzero
    restore()
    _backward(model, input_ids)
    for name, mask in head_masks.items():
        if name == o_name:
            continue
        unique = mask & ~head_masks_2[name]
        if not unique.any():
            continue
        assert has_grad(l0[name], unique), (
            f"{model_type}: {name} head alive but no gradient"
        )

    # Head dead: zero O for this head, gradients should vanish
    restore()
    with torch.no_grad():
        l0[o_name].data[head_masks[o_name]] = 0
    _backward(model, input_ids)
    for name, mask in head_masks.items():
        if name == o_name:
            continue
        unique = mask & ~head_masks_2[name]
        if not unique.any():
            continue
        assert not has_grad(l0[name], unique), (
            f"{model_type}: {name} head dead but still has gradient"
        )

    # --- Group test ---
    group_masks = {
        name: _build_mask(l0[name], head_params_spec[name], gi, n_heads, n_kv, hd, True)
        for name in head_params_spec
        if name in l0
    }

    # Group alive: gradients should be nonzero
    restore()
    _backward(model, input_ids)
    for name, gmask in group_masks.items():
        if name == o_name:
            continue
        assert has_grad(l0[name], gmask), (
            f"{model_type}: {name} group alive but no gradient"
        )

    # Group dead: zero O for this group, gradients should vanish
    restore()
    with torch.no_grad():
        l0[o_name].data[group_masks[o_name]] = 0
    _backward(model, input_ids)
    for name, gmask in group_masks.items():
        if name == o_name:
            continue
        assert not has_grad(l0[name], gmask), (
            f"{model_type}: {name} group dead but still has gradient"
        )
