import warnings
from typing import Union

import numpy as np
from torch.utils.data import DataLoader


def default_nbeta(
    dataloader: Union[DataLoader, int], gradient_accumulation_steps: int = 1
) -> float:
    if isinstance(dataloader, DataLoader):
        default_nbeta = dataloader.batch_size * gradient_accumulation_steps
        if default_nbeta <= 1:
            warnings.warn(
                "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
            )
            return 1
        else:
            return default_nbeta / np.log(default_nbeta)
    elif isinstance(dataloader, int):
        default_nbeta = dataloader * gradient_accumulation_steps
        if default_nbeta <= 1:
            warnings.warn(
                "default nbeta is undefined for batch_size * gradient_accumulation_steps == 1, falling back to default value of 1"
            )
            return 1
        else:
            return default_nbeta / np.log(default_nbeta)
    else:
        raise NotImplementedError(
            f"N*beta for data type {type(dataloader)} not implemented, use DataLoader or int instead."
        )


def tokenize_and_concatenate(
    dataset,
    tokenizer,
    streaming: bool = False,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
    num_proc: int = 10,
):
    """Tokenize and concatenate a text dataset into fixed-length sequences.

    Based on TransformerLens (MIT License, Copyright 2022 TransformerLensOrg):
    https://github.com/TransformerLensOrg/TransformerLens
    Core algorithm unchanged, with local additions: input validation
    (bos token and max_length checks), numpy reshape in place of einops, and
    the output column renamed from "tokens" to "input_ids".

    Joins all text separated by EOS tokens, tokenizes, then reshapes into
    (num_sequences, max_length) chunks.

    Args:
        dataset: HuggingFace text dataset.
        tokenizer: HuggingFace tokenizer with bos_token_id and eos_token_id.
        streaming: If True, disables parallel tokenization.
        max_length: Context window length.
        column_name: Name of the text column.
        add_bos_token: Prepend BOS to each sequence (reduces usable length by 1).
        num_proc: Number of processes for dataset.map().

    Returns:
        Dataset with an "input_ids" column of torch tensors.
    """
    dataset = dataset.select_columns(column_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    if add_bos_token and tokenizer.bos_token_id is None:
        raise ValueError(
            f"add_bos_token=True but tokenizer has no bos_token_id. "
            f"Use add_bos_token=False for {type(tokenizer).__name__}."
        )
    seq_len = max_length - 1 if add_bos_token else max_length
    if seq_len < 1:
        raise ValueError(
            f"max_length={max_length} is too small"
            f"{' with add_bos_token=True (needs at least 2)' if add_bos_token else ' (needs at least 1)'}."
        )

    def tokenize_function(examples):
        text = examples[column_name]
        full_text = tokenizer.eos_token.join(text)

        if not full_text.strip():
            return {"input_ids": np.array([], dtype=np.int64)}

        num_chunks = 20
        chunk_length = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_length : (i + 1) * chunk_length]
            for i in range(num_chunks)
        ]
        tokens = tokenizer(chunks, return_tensors="np", padding=True)[
            "input_ids"
        ].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]
        num_tokens = len(tokens)

        if num_tokens < seq_len:
            num_batches = 1
            tokens = tokens[:seq_len]
            if len(tokens) < seq_len:
                padding = np.full(seq_len - len(tokens), tokenizer.pad_token_id)
                tokens = np.concatenate([tokens, padding], axis=0)
        else:
            num_batches = num_tokens // seq_len
            tokens = tokens[: seq_len * num_batches]

        tokens = tokens.reshape(num_batches, seq_len)

        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"input_ids": tokens}

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=(num_proc if not streaming else None),
        remove_columns=[column_name],
    )
    if len(tokenized_dataset) == 0 or "input_ids" not in tokenized_dataset.column_names:
        raise ValueError(
            "Tokenization produced no sequences. Check that the input text is not empty."
        )
    tokenized_dataset.set_format(type="torch", columns=["input_ids"])
    return tokenized_dataset
