"""Round-trip a synthetic KV-cache list through encode/decode.

Run::

    PYTHONPATH=src python -m exo.integrations.tensorpuffer.codec_smoke
"""

from __future__ import annotations

import sys

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.integrations.tensorpuffer.codec import decode, encode


def _make_layer(n_kv_heads: int, head_dim: int, ntoks: int, dtype=mx.bfloat16) -> KVCache:
    kv = KVCache()
    kv.keys = mx.random.uniform(low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)).astype(dtype)
    kv.values = mx.random.uniform(low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)).astype(dtype)
    kv.offset = ntoks
    mx.eval(kv.keys, kv.values)
    return kv


def main() -> int:
    n_layers = 28
    cache = [_make_layer(8, 128, 220) for _ in range(n_layers)]
    blob = encode(cache)
    assert blob is not None, "encode returned None — only KVCache is supported in M0"
    print(f"encoded {n_layers} layers → {len(blob):,} bytes")
    decoded = decode(blob)
    assert decoded is not None, "decode failed"
    assert len(decoded) == n_layers
    for i, (orig, got) in enumerate(zip(cache, decoded)):
        assert got is not None
        assert got.offset == orig.offset, f"layer {i} offset mismatch"
        assert got.keys.shape == orig.keys.shape, f"layer {i} keys shape mismatch"
        # Bytewise equality on the underlying tensor data
        a = mx.allclose(got.keys.astype(mx.float32), orig.keys.astype(mx.float32), atol=0.0).item()
        b = mx.allclose(got.values.astype(mx.float32), orig.values.astype(mx.float32), atol=0.0).item()
        assert a and b, f"layer {i} value mismatch"
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
