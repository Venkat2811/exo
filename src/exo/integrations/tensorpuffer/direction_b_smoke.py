"""Direction B smoke: verify the in-tree KVPrefixCache patch wires
through tensorpuffer when ``TPUF_KVBM_ENABLE=1``.

Two phases in one process:

    Phase A — populate
        * Build a synthetic 28-layer × 220-token KV cache.
        * Call ``KVPrefixCache.add_kv_cache(prompt, cache)``.
        * Verify the patch logged ``[tpuf] stashed N bytes``.

    Phase B — read in a *fresh* KVPrefixCache instance
        * Construct a new KVPrefixCache (in-memory list is empty).
        * Call ``get_kv_cache(model, prompt_tokens)`` and assert the
          puffer fast-path kicks in (matched_index == 0, is_exact, the
          KVPrefixCache promoted the loaded cache to in-memory entry 0).

Run::

    TPUF_KVBM_ENABLE=1 \
    TPUF_S3_ENDPOINT=http://localhost:9100 ... \
    PYTHONPATH=src uv run python -m \
      exo.integrations.tensorpuffer.direction_b_smoke
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.worker.engines.mlx.cache import KVPrefixCache


def _make_layer(n_kv_heads: int, head_dim: int, ntoks: int, dtype=mx.bfloat16) -> KVCache:
    kv = KVCache()
    kv.keys = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)
    ).astype(dtype)
    kv.values = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)
    ).astype(dtype)
    kv.offset = ntoks
    mx.eval(kv.keys, kv.values)
    return kv


def main() -> int:
    if os.environ.get("TPUF_KVBM_ENABLE", "").lower() not in {"1", "true"}:
        print("DISABLED: set TPUF_KVBM_ENABLE=1")
        return 1

    n_layers = 28
    ntoks = 220
    cache = [_make_layer(8, 128, ntoks) for _ in range(n_layers)]
    prompt_tokens = mx.array(list(range(1000, 1000 + ntoks)), dtype=mx.int32)

    # ---------- Phase A: populate ----------
    a = KVPrefixCache(group=None)
    assert a._tpuf is not None, "tensorpuffer handle should be live"
    a.add_kv_cache(prompt_tokens, cache)
    print(f"[A] stashed {n_layers}-layer cache for {ntoks}-token prompt")

    # ---------- Phase B: fresh instance, expect puffer hit ----------
    b = KVPrefixCache(group=None)
    assert b._tpuf is not None
    assert len(b.prompts) == 0  # truly empty in-memory

    # Stub Model with a `layers` attribute and a make_cache() that returns
    # a fresh empty cache list (matches make_kv_cache's branch).
    fake_model = SimpleNamespace(
        layers=[None] * n_layers,
        make_cache=lambda: [KVCache() for _ in range(n_layers)],
    )

    cache_out, remaining, matched_idx, is_exact = b.get_kv_cache(
        fake_model, prompt_tokens
    )
    print(
        f"[B] get_kv_cache → matched_idx={matched_idx} is_exact={is_exact} "
        f"remaining={len(remaining)} cache_layers={len(cache_out)}"
    )
    assert matched_idx == 0, f"expected idx 0, got {matched_idx}"
    assert is_exact is True
    assert len(cache_out) == n_layers
    assert len(remaining) == 1, "exact-match returns last token only"
    assert int(remaining[0].item()) == int(prompt_tokens[-1].item())
    # The puffer-loaded cache should have been promoted to entry 0
    assert len(b.prompts) == 1
    assert len(b.caches) == 1

    # Sanity: layer-0 keys round-tripped (bytewise on bf16)
    src_keys = cache[0].keys
    dst_keys = b.caches[0][0].keys  # type: ignore[index]
    eq = mx.allclose(
        src_keys.astype(mx.float32), dst_keys.astype(mx.float32), atol=0.0
    ).item()
    assert eq, "layer-0 keys mismatch — codec lost data"

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
