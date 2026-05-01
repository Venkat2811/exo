"""Binary codec for MLX KV-cache state, on the way to / from tensorpuffer.

Scope: handle the common cases — a list of ``KVCache`` and
``RotatingKVCache`` (and ``None``) entries, one per layer, as built by
``make_kv_cache`` in ``exo/worker/engines/mlx/cache.py``. Other cache
flavours (``QuantizedKVCache``, ``ArraysCache``, ``DeepseekV4Cache``,
``CacheList``) cause :func:`encode` to return ``None`` so the caller
falls through to a normal cold prefill — same posture the in-memory
matcher already takes for SSM caches without snapshots.

Format (little-endian throughout, no padding)::

    magic:    "TPMX"  (4 B)
    version:  u32     (= 2)        ← bumped from 1 when RotatingKVCache landed
    nlayers:  u32
    per layer:
        kind:  u8       (0 = None, 1 = KVCache, 2 = RotatingKVCache)
        for kind == 1 (KVCache):
            offset: u32
            (keys present)?u8 (values present)?u8
            for each present tensor:
                dtype: u8   (0=bf16, 1=f16, 2=f32, 3=u8, 4=u16, 5=u32, 6=i32, 7=i64)
                ndim:  u8
                shape: u32 × ndim
                nbytes: u64
                data:  raw nbytes
        for kind == 2 (RotatingKVCache):
            offset: u32
            _idx:   u32
            keep:   u32
            max_size: u32
            (keys present)?u8 (values present)?u8
            for each present tensor: same per-tensor blob as above

Versioning: any decoder that reads version 2 must understand kind=0/1/2.
Older v1 blobs aren't supported by this version of the codec — re-stash
to upgrade.

The codec is intentionally narrow. It exists so exo can lift the most
common KV-state into bytes for puffer storage; smarter formats (e.g.
quantized + zstd) come later.
"""

from __future__ import annotations

import struct
from io import BytesIO
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx
    from mlx_lm.models.cache import KVCache

_MAGIC = b"TPMX"
_VERSION = 2

_KIND_NONE = 0
_KIND_KVCACHE = 1
_KIND_ROTATING = 2

_DTYPE_CODE: dict[str, int] = {
    "bfloat16": 0,
    "float16":  1,
    "float32":  2,
    "uint8":    3,
    "uint16":   4,
    "uint32":   5,
    "int32":    6,
    "int64":    7,
}
_DTYPE_FROM_CODE: dict[int, str] = {v: k for k, v in _DTYPE_CODE.items()}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _entry_kind(entry: object) -> int:
    """Map a cache entry to one of the supported kind codes, or
    ``-1`` for unsupported (caller falls through to cold prefill).

    We check by class name to avoid importing every rare cache type at
    module load time.
    """
    if entry is None:
        return _KIND_NONE
    cls = type(entry).__name__
    if cls == "KVCache":
        return _KIND_KVCACHE
    if cls == "RotatingKVCache":
        return _KIND_ROTATING
    return -1


def _array_to_numpy(a: "mx.array") -> np.ndarray:
    """Eval + materialize an mx.array as a numpy array, preserving dtype.

    Bfloat16 has no numpy equivalent; we store it via uint16 reinterpret
    so the bytes round-trip exactly.
    """
    import mlx.core as mx

    mx.eval(a)
    if a.dtype == mx.bfloat16:
        return np.array(a.view(mx.uint16))
    return np.array(a)


def _numpy_to_array(buf: bytes, dtype_code: int, shape: Tuple[int, ...]) -> "mx.array":
    import mlx.core as mx

    name = _DTYPE_FROM_CODE[dtype_code]
    if name == "bfloat16":
        np_arr = np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        return mx.array(np_arr).view(mx.bfloat16)
    np_arr = np.frombuffer(buf, dtype=np.dtype(name)).reshape(shape)
    return mx.array(np_arr)


def _dtype_name(a: "mx.array") -> str:
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        return "bfloat16"
    return str(a.dtype).split(".")[-1]


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def _write_tensor(out: BytesIO, arr: "mx.array") -> None:
    np_arr = _array_to_numpy(arr)
    dtype_code = _DTYPE_CODE[_dtype_name(arr)]
    shape = tuple(int(d) for d in np_arr.shape)
    out.write(struct.pack("<BB", dtype_code, len(shape)))
    out.write(struct.pack(f"<{len(shape)}I", *shape))
    buf = np_arr.tobytes()
    out.write(struct.pack("<Q", len(buf)))
    out.write(buf)


def encode(cache: List[Optional[object]]) -> Optional[bytes]:
    """Serialize a list of KV cache entries. Returns ``None`` if any entry
    is an unsupported type — caller should treat that as "skip stash".
    """
    kinds: list[int] = []
    for entry in cache:
        kind = _entry_kind(entry)
        if kind < 0:
            return None
        kinds.append(kind)

    out = BytesIO()
    out.write(_MAGIC)
    out.write(struct.pack("<II", _VERSION, len(cache)))

    for entry, kind in zip(cache, kinds):
        out.write(struct.pack("<B", kind))
        if kind == _KIND_NONE:
            continue

        keys: Optional["mx.array"] = entry.keys
        values: Optional["mx.array"] = entry.values
        offset = int(getattr(entry, "offset", 0))
        out.write(struct.pack("<I", offset))

        if kind == _KIND_ROTATING:
            # RotatingKVCache carries three extra ints
            out.write(struct.pack(
                "<III",
                int(getattr(entry, "_idx", 0)),
                int(getattr(entry, "keep", 0)),
                int(getattr(entry, "max_size", 0)),
            ))

        out.write(struct.pack("<BB", 1 if keys is not None else 0,
                              1 if values is not None else 0))
        for arr in (keys, values):
            if arr is None:
                continue
            _write_tensor(out, arr)

    return out.getvalue()


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def _read_tensor(blob: bytes, pos: int) -> tuple["mx.array", int]:
    if pos + 2 > len(blob):
        raise ValueError("short read on tensor header")
    dtype_code, ndim = blob[pos], blob[pos + 1]
    pos += 2
    shape = struct.unpack_from(f"<{ndim}I", blob, pos)
    pos += 4 * ndim
    (nbytes,) = struct.unpack_from("<Q", blob, pos)
    pos += 8
    arr = _numpy_to_array(blob[pos : pos + nbytes], dtype_code, shape)
    pos += nbytes
    return arr, pos


def decode(blob: bytes) -> Optional[List[Optional[object]]]:
    """Inverse of :func:`encode`. Returns ``None`` if the magic / version
    don't match (caller should treat as miss + fall through).
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache  # local import — heavy

    if len(blob) < 12 or blob[:4] != _MAGIC:
        return None
    version, nlayers = struct.unpack_from("<II", blob, 4)
    if version != _VERSION:
        return None

    pos = 12
    out: list[Optional[object]] = []
    try:
        while len(out) < nlayers:
            if pos + 1 > len(blob):
                return None
            kind = blob[pos]
            pos += 1

            if kind == _KIND_NONE:
                out.append(None)
                continue

            if kind not in (_KIND_KVCACHE, _KIND_ROTATING):
                return None

            (offset,) = struct.unpack_from("<I", blob, pos)
            pos += 4
            extra = None
            if kind == _KIND_ROTATING:
                idx, keep, max_size = struct.unpack_from("<III", blob, pos)
                pos += 12
                extra = (idx, keep, max_size)

            has_keys, has_values = blob[pos], blob[pos + 1]
            pos += 2

            keys = None
            values = None
            if has_keys:
                keys, pos = _read_tensor(blob, pos)
            if has_values:
                values, pos = _read_tensor(blob, pos)

            if kind == _KIND_KVCACHE:
                kv = KVCache()
                kv.offset = offset
                kv.keys = keys
                kv.values = values
                out.append(kv)
            else:
                kv = RotatingKVCache.__new__(RotatingKVCache)
                kv.offset = offset
                kv._idx = extra[0]
                kv.keep = extra[1]
                kv.max_size = extra[2]
                kv.keys = keys
                kv.values = values
                out.append(kv)
    except (struct.error, ValueError):
        return None

    return out
