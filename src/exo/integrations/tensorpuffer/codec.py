"""Binary codec for MLX KV-cache state, on the way to / from tensorpuffer.

Scope (M0): handle the common case — a list of ``KVCache`` (and ``None``)
entries, one per layer, as built by ``make_kv_cache`` in
``exo/worker/engines/mlx/cache.py``. Other cache flavours
(``RotatingKVCache``, ``QuantizedKVCache``, ``ArraysCache``,
``DeepseekV4Cache``, ``CacheList``) cause :func:`encode` to return ``None``
so the caller falls through to a normal cold prefill — same posture the
in-memory matcher already takes for SSM caches without snapshots.

Format (little-endian throughout, no padding)::

    magic:    "TPMX"  (4 B)
    version:  u32     (= 1)
    nlayers:  u32
    per layer:
        flags: u8       (bit0 = has-keys, bit1 = has-values, bit2 = is-None)
        offset: u32     (KVCache.offset)
        for each present tensor (keys then values):
            dtype: u8   (0=bf16, 1=f16, 2=f32, 3=u8, 4=u16, 5=u32 — extend as needed)
            ndim:  u8
            shape: u32 × ndim
            nbytes: u64
            data:  raw nbytes

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
_VERSION = 1

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


def _is_simple_kvcache(entry: object) -> bool:
    """True iff entry is a vanilla ``mlx_lm.models.cache.KVCache`` (not a
    rotating / quantized / arrays / cache-list / deepseek-v4 variant).

    We check by class name to avoid importing every rare cache type at
    module load time.
    """
    if entry is None:
        return True
    cls = type(entry).__name__
    return cls == "KVCache"


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


def encode(cache: List[Optional["KVCache"]]) -> Optional[bytes]:
    """Serialize a list of KV cache entries. Returns ``None`` if any entry
    is an unsupported type — caller should treat that as "skip stash".
    """
    for entry in cache:
        if not _is_simple_kvcache(entry):
            return None

    out = BytesIO()
    out.write(_MAGIC)
    out.write(struct.pack("<II", _VERSION, len(cache)))

    for entry in cache:
        if entry is None:
            out.write(struct.pack("<BI", 0b100, 0))  # flags=isNone, offset=0
            continue
        keys: Optional["mx.array"] = entry.keys
        values: Optional["mx.array"] = entry.values
        flags = 0
        if keys is not None:
            flags |= 0b001
        if values is not None:
            flags |= 0b010
        offset = int(getattr(entry, "offset", 0))
        out.write(struct.pack("<BI", flags, offset))

        for arr in (keys, values):
            if arr is None:
                continue
            np_arr = _array_to_numpy(arr)
            dtype_code = _DTYPE_CODE[_dtype_name(arr)]
            shape = tuple(int(d) for d in np_arr.shape)
            out.write(struct.pack("<BB", dtype_code, len(shape)))
            out.write(struct.pack(f"<{len(shape)}I", *shape))
            buf = np_arr.tobytes()
            out.write(struct.pack("<Q", len(buf)))
            out.write(buf)

    return out.getvalue()


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def decode(blob: bytes) -> Optional[List[Optional["KVCache"]]]:
    """Inverse of :func:`encode`. Returns ``None`` if the magic / version
    don't match (caller should treat as miss + fall through).
    """
    from mlx_lm.models.cache import KVCache  # local import — heavy

    if len(blob) < 12 or blob[:4] != _MAGIC:
        return None
    version, nlayers = struct.unpack_from("<II", blob, 4)
    if version != _VERSION:
        return None

    pos = 12
    out: list[Optional[KVCache]] = []
    while len(out) < nlayers:
        if pos + 5 > len(blob):
            return None
        flags = blob[pos]
        offset = struct.unpack_from("<I", blob, pos + 1)[0]
        pos += 5

        if flags & 0b100:
            out.append(None)
            continue

        kv = KVCache()
        kv.offset = offset

        for which in ("keys", "values"):
            if not (flags & (0b001 if which == "keys" else 0b010)):
                continue
            if pos + 2 > len(blob):
                return None
            dtype_code, ndim = blob[pos], blob[pos + 1]
            pos += 2
            shape = struct.unpack_from(f"<{ndim}I", blob, pos)
            pos += 4 * ndim
            (nbytes,) = struct.unpack_from("<Q", blob, pos)
            pos += 8
            arr = _numpy_to_array(blob[pos : pos + nbytes], dtype_code, shape)
            pos += nbytes
            setattr(kv, which, arr)

        out.append(kv)

    return out
