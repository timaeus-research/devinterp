"""Zarr schema for creating xarray-compatible zarr stores.

Creates a zarr store with typed arrays and group attributes in one
batch call, producing a store that xr.open_datatree reads back as
an equivalent DataTree.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import zarr
from zarr.abc.store import Store as ZarrStore
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.metadata import ArrayV3Metadata

from devinterp.slt.writing import validate_dtype_str_torch_numpy_compat

logger = logging.getLogger(__name__)


# ─── Fill values ─────────────────────────────────────────────────────────────


def _numpy_fill_value(dtype: np.dtype) -> np.generic:
    if dtype.kind == "f":
        return dtype.type(np.nan)
    elif dtype.kind == "c":
        return dtype.type(complex(np.nan, np.nan))
    elif dtype.kind in ("U", "S", "O"):
        return dtype.type("")
    return dtype.type(0)


_XARRAY_FILL_VALUE_NAN_B64 = "AAAAAAAA+H8="


def _build_array_metadata(
    *,
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    np_dtype: np.dtype,
    chunks: tuple[int, ...] | None = None,
    attrs: dict[str, Any] | None = None,
) -> ArrayV3Metadata:
    """Build zarr ArrayV3Metadata matching xarray encoding conventions."""
    from zarr.codecs import BytesCodec, ZstdCodec
    from zarr.core.chunk_key_encodings import DefaultChunkKeyEncoding
    from zarr.core.dtype import get_data_type_from_native_dtype

    if chunks is None:
        chunks = shape
    fill_attrs = (
        {"_FillValue": _XARRAY_FILL_VALUE_NAN_B64}
        if np_dtype.kind in ("f", "c")
        else {}
    )
    attrs = fill_attrs | (attrs or {})
    return ArrayV3Metadata(
        shape=shape,
        data_type=get_data_type_from_native_dtype(np_dtype),
        chunk_grid=RegularChunkGrid(chunk_shape=chunks),
        chunk_key_encoding=DefaultChunkKeyEncoding(separator="/"),
        codecs=[BytesCodec(), ZstdCodec(level=0, checksum=False)],
        fill_value=_numpy_fill_value(np_dtype),
        attributes=attrs or None,
        dimension_names=dims if dims else None,
    )


# ─── DataArraySpec ───────────────────────────────────────────────────────────


@dataclass
class DataArraySpec:
    """Specification for a zarr array: shape, dims, dtype, chunks."""

    shape: tuple[int, ...]
    dims: tuple[str, ...]
    dtype_str: str
    chunks: tuple[int, ...] = None  # type: ignore
    attrs: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.chunks is None:
            self.chunks = self.shape
        if self.attrs is None:
            self.attrs = {}
        validate_dtype_str_torch_numpy_compat(self.dtype_str)
        assert len(self.shape) == len(self.dims)
        assert len(self.chunks) == len(self.shape)

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype_str)

    @property
    def torch_dtype(self) -> torch.dtype:
        attr = getattr(torch, self.dtype_str)
        assert isinstance(attr, torch.dtype)
        return attr

    def to_metadata(self) -> ArrayV3Metadata:
        return _build_array_metadata(
            shape=self.shape,
            dims=self.dims,
            np_dtype=self.numpy_dtype,
            chunks=self.chunks,
            attrs=self.attrs,
        )


# ─── ZarrSchema ──────────────────────────────────────────────────────────────


@dataclass
class ZarrSchema:
    """Schema for a flat zarr store (arrays at root level with group attributes)."""

    arrays_meta: dict[str, DataArraySpec] = field(default_factory=dict)
    group_attrs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def create_hierarchy(
        self, store: ZarrStore
    ) -> tuple[zarr.Group, dict[str, zarr.Array]]:
        """Create the zarr store. Returns (root_group, arrays_dict)."""
        root = zarr.open_group(store, mode="w")

        root_attrs = self.group_attrs.get("")
        if root_attrs:
            root.update_attributes(root_attrs)

        nodes = {path: spec.to_metadata() for path, spec in self.arrays_meta.items()}
        created = dict(root.create_hierarchy(nodes, overwrite=True))

        arrays: dict[str, zarr.Array] = {}
        for path in self.arrays_meta:
            node = created[path]
            assert isinstance(node, zarr.Array), (
                f"Expected Array at {path!r}, got {type(node).__name__}"
            )
            arrays[path] = node

        logger.info("Zarr store created: %d arrays.", len(arrays))
        return root, arrays
