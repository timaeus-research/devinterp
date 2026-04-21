"""Create minimal HuggingFace configs for testing.

Not part of the public API. Intended for parity tests only.

`make_tiny(model_type)` produces an `AutoConfig` sized small enough for
fast test iteration (4 attention heads x 8 dim = 32 hidden, vocab=32,
max_position=64, 1 layer, etc). Used by parity tests that need to
instantiate every supported architecture.

Only supports model types present in HEAD_STRUCTURES (_model_structures.py).

`_ATTR_TABLE` is a list of `(value, [attr_names])` pairs. For each
config, we set whichever of those attrs exist to the given value -- this
handles HuggingFace's inconsistent naming across model families
(e.g. `num_hidden_layers` vs `n_layer` vs `num_layers` all mean the same
thing). Example: `(1, ["num_hidden_layers", "n_layer", ...])` sets
any of those attrs to 1, making a 1-layer model.
"""

from transformers import AutoConfig

from devinterp.slt._model_structures import HEAD_STRUCTURES

NUM_HEADS = 4
HEAD_DIM = 8
HIDDEN = NUM_HEADS * HEAD_DIM  # 32

# {value: [attr_names]} — set each attr to value if it exists on the config.
_ATTR_TABLE: list[tuple[object, list[str]]] = [
    (
        1,
        [
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            "decoder_layers",
            "encoder_layers",
            "num_decoder_layers",
            "num_encoder_layers",
            "num_experts_per_tok",
        ],
    ),
    (
        NUM_HEADS,
        [
            "num_attention_heads",
            "n_head",
            "decoder_attention_heads",
            "encoder_attention_heads",
            "num_decoder_attention_heads",
            "num_encoder_attention_heads",
        ],
    ),
    (
        HIDDEN,
        ["hidden_size", "d_model", "n_embd", "embed_dim", "dim", "word_embed_proj_dim"],
    ),
    (
        HIDDEN * 4,
        [
            "intermediate_size",
            "n_inner",
            "ffn_dim",
            "d_inner",
            "encoder_ffn_dim",
            "decoder_ffn_dim",
        ],
    ),
    (32, ["vocab_size"]),
    (2, ["num_experts", "num_local_experts", "moe_num_experts"]),
    (
        64,
        [
            "max_position_embeddings",
            "n_positions",
            "max_seq_len",
            "max_source_positions",
            "max_target_positions",
            "original_max_position_embeddings",
        ],
    ),
]


def _set_if_exists(
    config: AutoConfig, attr: str, value: object, even_if_none: bool = False
) -> None:
    if even_if_none:
        if not hasattr(config, attr):
            return
    elif getattr(config, attr, None) is None:
        return
    try:
        setattr(config, attr, value)
    except AttributeError:
        pass  # read-only property


def make_tiny(model_type: str, gqa: bool = False) -> AutoConfig:
    if model_type not in HEAD_STRUCTURES or model_type == "hooked_transformer":
        raise ValueError(f"{model_type} is not a supported HuggingFace model type")

    config = AutoConfig.for_model(model_type)

    for value, attrs in _ATTR_TABLE:
        for attr in attrs:
            _set_if_exists(config, attr, value)

    # These attrs may default to None but must be set for model construction
    for attr in ["head_dim", "d_head", "kv_channels", "rotary_dim", "dim_head"]:
        _set_if_exists(config, attr, HEAD_DIM, even_if_none=True)
    _set_if_exists(config, "num_key_value_heads", NUM_HEADS, even_if_none=True)

    # Catch variant vocab/hidden sizes (e.g. vocab_size_per_layer_input)
    for attr in dir(config):
        if attr.startswith("_"):
            continue
        try:
            if "vocab_size" in attr:
                setattr(config, attr, 32)
            elif "hidden_size" in attr:
                setattr(config, attr, HIDDEN)
        except (AttributeError, TypeError):
            pass

    config.is_decoder = True
    config.attn_implementation = "eager"
    config.pad_token_id = 0
    dec_start = getattr(config, "decoder_start_token_id", None)
    if dec_start is not None and dec_start >= config.vocab_size:
        config.decoder_start_token_id = 1

    # GQA: use fewer KV heads than Q heads
    GQA_N_KV = 2
    if gqa and getattr(config, "num_key_value_heads", None) is not None:
        config.num_key_value_heads = GQA_N_KV

    return config
