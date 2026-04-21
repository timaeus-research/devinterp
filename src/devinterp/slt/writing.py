"""Zarr writing infrastructure for streaming sampling results.

ZarrWriter buffers per-chain data and writes to zarr arrays, optionally
using a thread pool for async I/O.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent import futures
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import assert_never

import numpy as np
import torch
import zarr
from zarr.core.metadata import ArrayV3Metadata

logger = logging.getLogger(__name__)


# ─── Dtype validation (used by zarr_schema.py) ──────────────────────────────

SAFE_DTYPE_STRINGS = frozenset(
    [
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "bool",
        "complex64",
        "complex128",
    ]
)


def validate_dtype_str_torch_numpy_compat(dtype_str: str) -> None:
    if dtype_str not in SAFE_DTYPE_STRINGS:
        raise ValueError(
            f"Unsupported dtype: '{dtype_str}'. Use one of: {sorted(SAFE_DTYPE_STRINGS)}"
        )


# ─── Zarr dimension helpers ──────────────────────────────────────────────────


def _zarr_dims(arr: zarr.Array) -> tuple[str, ...]:
    assert isinstance(arr.metadata, ArrayV3Metadata)
    dims = arr.metadata.dimension_names or ()
    assert all(d is not None for d in dims), f"All dimensions must be named, got {dims}"
    return dims  # type: ignore[return-value]


def _indices_from_dims(
    arr: zarr.Array, **dim_indices: int | slice
) -> tuple[int | slice, ...]:
    dims = _zarr_dims(arr)
    indices = [dim_indices.pop(dim, slice(None)) for dim in dims]
    if dim_indices:
        raise ValueError(
            f"Unexpected dimensions: {list(dim_indices)}. Valid: {list(dims)}"
        )
    return tuple(indices)


# ─── Write execution ────────────────────────────────────────────────────────


def _execute_write(
    arr: zarr.Array,
    data: np.ndarray | torch.Tensor,
    dim_indices: dict[str, int | slice],
    *,
    executor: ThreadPoolExecutor | None = None,
    write_futures: list[Future] | None = None,
) -> None:
    indices = _indices_from_dims(arr, **dim_indices)

    match data:
        case torch.Tensor():
            np_data = data.numpy(force=True).copy()
        case np.ndarray():
            np_data = data
        case _ as invalid:
            assert_never(invalid)

    if executor is not None:
        assert write_futures is not None
        write_futures.append(executor.submit(arr.set_basic_selection, indices, np_data))
    else:
        arr.set_basic_selection(indices, np_data)


# ─── Buffer ──────────────────────────────────────────────────────────────────


class _Buffer:
    """Accumulates complete rows before flushing to zarr."""

    def __init__(self, buffer: torch.Tensor):
        self.buffer = buffer
        self.row_ptr = 0
        self.size = 0

    @property
    def capacity(self) -> int:
        return self.buffer.shape[0]

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def push(self, data: torch.Tensor, row_index: int) -> None:
        if self.is_full():
            raise RuntimeError("Buffer full. Call flush before writing more data.")
        assert row_index == self.row_ptr + self.size, (
            f"Row index mismatch: expected {self.row_ptr + self.size}, got {row_index}"
        )
        self.buffer[self.size].copy_(data)
        self.size += 1

    def flush(self) -> tuple[np.ndarray, slice] | None:
        if self.size == 0:
            return None
        np_data = self.buffer[: self.size].numpy(force=True).copy()
        row_slice = slice(self.row_ptr, self.row_ptr + self.size)
        self.row_ptr += self.size
        self.size = 0
        return np_data, row_slice


# ─── ZarrWriter ──────────────────────────────────────────────────────────────


@dataclass
class ZarrWriter:
    """Buffered writer for streaming per-chain data to zarr arrays.

    Use ZarrWriter.open() for a context manager that manages the thread pool.
    """

    arrays: dict[str, zarr.Array]
    chain_buffer_size: int
    buffer_device: torch.device
    executor: ThreadPoolExecutor | None = None
    _write_futures: list[Future] = field(default_factory=list, init=False)
    _chain_buffers: dict[tuple[str, int], _Buffer] = field(
        default_factory=dict, init=False
    )

    @classmethod
    @contextmanager
    def open(
        cls,
        arrays: dict[str, zarr.Array],
        chain_buffer_size: int,
        buffer_device: torch.device,
        max_write_threads: int = 4,
    ) -> Iterator[ZarrWriter]:
        pool = (
            ThreadPoolExecutor(
                max_workers=max_write_threads, thread_name_prefix="zarr_write"
            )
            if max_write_threads > 0
            else nullcontext()
        )
        with pool as executor:
            if executor is not None:
                logger.info("zarr_write thread pool: %d max threads", max_write_threads)
            writer = cls(
                arrays=arrays,
                chain_buffer_size=chain_buffer_size,
                buffer_device=buffer_device,
                executor=executor,
            )
            try:
                yield writer
            finally:
                writer.flush_all_buffers()
                writer.wait_for_writes()

    def write(
        self, path: str, data: np.ndarray | torch.Tensor, /, **dim_indices: int | slice
    ) -> None:
        """Immediate write (for scalars, init_loss, fixed observables)."""
        _execute_write(
            self.arrays[path],
            data,
            dim_indices,
            executor=self.executor,
            write_futures=self._write_futures,
        )

    def push(
        self,
        path: str,
        data: torch.Tensor,
        /,
        *,
        chain: int,
        **dim_indices: int | slice,
    ) -> None:
        """Push a complete row into a chain buffer."""
        arr = self.arrays[path]
        dims = _zarr_dims(arr)
        assert dims[0] == "chain"
        row_dim = dims[1]
        assert row_dim in dim_indices
        row_index = dim_indices[row_dim]
        assert isinstance(row_index, int)

        buf = self._get_or_create_buffer(path, chain)
        buf.push(data, row_index=row_index)

    def flush_full_buffers(self) -> None:
        self._flush_buffers(only_full=True)

    def flush_all_buffers(self) -> None:
        self._flush_buffers(only_full=False)

    def wait_for_writes(self) -> None:
        if not self._write_futures:
            return

        done, notdone = futures.wait(
            self._write_futures, return_when=futures.ALL_COMPLETED
        )
        exceptions = [f.exception() for f in done if f.exception()]
        self._write_futures.clear()

        if exceptions:
            if len(exceptions) == 1:
                raise exceptions[0]  # type: ignore[misc]
            raise ExceptionGroup(f"{len(exceptions)} zarr write(s) failed", exceptions)  # type: ignore[arg-type]

        logger.info("All %d zarr writes completed successfully", len(done))

    def _flush_buffers(self, only_full: bool) -> None:
        for (path, chain), buf in self._chain_buffers.items():
            if only_full and not buf.is_full():
                continue
            result = buf.flush()
            if result is not None:
                np_data, row_slice = result
                arr = self.arrays[path]
                dims = _zarr_dims(arr)
                _execute_write(
                    arr,
                    np_data,
                    {"chain": chain, dims[1]: row_slice},
                    executor=self.executor,
                    write_futures=self._write_futures,
                )

    def _get_or_create_buffer(self, path: str, chain: int) -> _Buffer:
        key = (path, chain)
        if key not in self._chain_buffers:
            arr = self.arrays[path]
            self._chain_buffers[key] = _Buffer(
                buffer=torch.zeros(
                    self.chain_buffer_size,
                    *arr.shape[2:],
                    dtype=getattr(torch, str(arr.dtype)),
                    device=self.buffer_device,
                ),
            )
        return self._chain_buffers[key]
