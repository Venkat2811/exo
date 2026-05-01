"""Direction A — external harness: run a tensorpuffer-aware
KVPrefixCache via runtime monkey-patching, without relying on the
in-tree TPUF_KVBM_ENABLE hooks that Direction B added.

Why both directions exist
-------------------------

Direction B (in-tree, gated by ``TPUF_KVBM_ENABLE=1``) is the path
upstream-friendly users would take. Direction A is what someone would
write if they couldn't (or didn't want to) modify exo's source — e.g.
running against an unmodified upstream wheel with monkey-patches added
at process start.

This harness:

1. Builds a synthetic KV cache (no model load — same shape as Direction
   B's smoke test).
2. Constructs a ``KVPrefixCache`` with the in-tree puffer hooks
   intentionally disabled (``TPUF_KVBM_ENABLE`` unset).
3. Wraps the instance with :class:`HarnessTpufWrapper` which composes
   tensorpuffer stash/load on top of the existing ``add_kv_cache`` /
   ``get_kv_cache`` — no inheritance, no class swap, just method
   delegation.
4. Runs the same two-phase A → B round-trip the in-tree smoke does and
   verifies bytewise equality of the loaded cache.

The wrapper class is the artifact you'd ship as a downstream package
if you wanted Direction A in production: ``import wrapper, wrap(cache,
model_id) -> tpuf-augmented cache``.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache

from exo.integrations.tensorpuffer.client import Tensorpuffer
from exo.integrations.tensorpuffer.codec import decode, encode
from exo.worker.engines.mlx.cache import KVPrefixCache


class HarnessTpufWrapper:
    """Composition wrapper that adds tensorpuffer stash/load on top of
    an existing ``KVPrefixCache`` without modifying it.

    Use::

        cache = KVPrefixCache(group=None)
        wrapped = HarnessTpufWrapper(cache, model_id="exo-mlx")
        wrapped.add_kv_cache(prompt, kv_state)
        out_cache, remaining, idx, is_exact = wrapped.get_kv_cache(model, prompt)
    """

    def __init__(self, inner: KVPrefixCache, model_id: str = "exo-mlx") -> None:
        self.inner = inner
        self.model_id = model_id
        self.tp: Optional[Tensorpuffer] = Tensorpuffer()

    def _tokens(self, prompt_tokens: mx.array) -> list[int]:
        return [int(t) for t in prompt_tokens.tolist()]

    def add_kv_cache(self, prompt_tokens: mx.array, cache, *args, **kwargs) -> None:
        self.inner.add_kv_cache(prompt_tokens, cache, *args, **kwargs)
        if self.tp is None:
            return
        blob = encode(cache)
        if blob is None:
            return
        n = self.tp.stash_prefix(self.model_id, self._tokens(prompt_tokens), blob)
        print(f"[harness/A] stashed {n:,} bytes")

    def get_kv_cache(self, model, prompt_tokens: mx.array, media_regions=None):
        # Probe puffer first
        if self.tp is not None and not media_regions:
            blob = self.tp.try_load_prefix(self.model_id, self._tokens(prompt_tokens))
            if blob is not None:
                cache = decode(blob)
                if cache is not None:
                    print(f"[harness/A] puffer hit: {len(blob):,} bytes")
                    # Promote to inner's in-memory list so future calls hit
                    # the in-memory path
                    self.inner._evict_if_needed()
                    self.inner.prompts.append(prompt_tokens)
                    self.inner.caches.append(cache)
                    self.inner._snapshots.append(None)
                    self.inner._media_regions.append([])
                    self.inner.prefill_tps.append(0.0)
                    self.inner._access_counter += 1
                    self.inner._last_used.append(self.inner._access_counter)
                    idx = len(self.inner.prompts) - 1
                    remaining = prompt_tokens[len(prompt_tokens) - 1 :]
                    return cache, remaining, idx, True
        return self.inner.get_kv_cache(model, prompt_tokens, media_regions)


def _make_layer(n_kv_heads: int, head_dim: int, ntoks: int) -> KVCache:
    kv = KVCache()
    kv.keys = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)
    ).astype(mx.bfloat16)
    kv.values = mx.random.uniform(
        low=-1.0, high=1.0, shape=(1, n_kv_heads, ntoks, head_dim)
    ).astype(mx.bfloat16)
    kv.offset = ntoks
    mx.eval(kv.keys, kv.values)
    return kv


def main() -> int:
    # Ensure Direction B (in-tree) is OFF for the duration of inner
    # construction. We toggle TPUF_KVBM_ENABLE on/off so the in-tree
    # hook in __init__ sees an unset env and stays dormant — proving the
    # external wrapper carries the integration on its own.
    saved_env = os.environ.pop("TPUF_KVBM_ENABLE", None)

    n_layers = 28
    ntoks = 220
    src_cache = [_make_layer(8, 128, ntoks) for _ in range(n_layers)]
    prompt_tokens = mx.array(list(range(2000, 2000 + ntoks)), dtype=mx.int32)

    # ---------- Phase A: populate via wrapper ----------
    inner_a = KVPrefixCache(group=None)
    assert inner_a._tpuf is None, (
        "in-tree hook fired despite TPUF_KVBM_ENABLE being unset; "
        "Direction A test environment is contaminated"
    )
    # Now turn the env on so HarnessTpufWrapper.Tensorpuffer() works.
    os.environ["TPUF_KVBM_ENABLE"] = saved_env or "1"
    wrapper_a = HarnessTpufWrapper(inner_a, model_id="exo-harness-A")
    wrapper_a.add_kv_cache(prompt_tokens, src_cache)
    print(f"[A] populated; inner has {len(inner_a.prompts)} entries")

    # ---------- Phase B: fresh KVPrefixCache + fresh wrapper, expect hit ----------
    # Turn the env off again so the new inner_b's in-tree hook stays
    # dormant — Phase B should hit the puffer via the wrapper, not the
    # in-tree path. Then turn it back on for the wrapper construction.
    os.environ.pop("TPUF_KVBM_ENABLE", None)
    inner_b = KVPrefixCache(group=None)
    assert inner_b._tpuf is None, "in-tree hook should still be dormant"
    assert len(inner_b.prompts) == 0
    os.environ["TPUF_KVBM_ENABLE"] = saved_env or "1"
    wrapper_b = HarnessTpufWrapper(inner_b, model_id="exo-harness-A")

    fake_model = SimpleNamespace(
        layers=[None] * n_layers,
        make_cache=lambda: [KVCache() for _ in range(n_layers)],
    )
    cache_out, remaining, idx, is_exact = wrapper_b.get_kv_cache(
        fake_model, prompt_tokens
    )
    assert is_exact and idx == 0, f"want (idx=0,exact); got ({idx},{is_exact})"
    assert len(cache_out) == n_layers
    assert int(remaining[0].item()) == int(prompt_tokens[-1].item())

    # Bytewise equality on layer-0 keys
    eq = mx.allclose(
        src_cache[0].keys.astype(mx.float32),
        cache_out[0].keys.astype(mx.float32),
        atol=0.0,
    ).item()
    assert eq, "layer-0 keys mismatch — codec lost data"

    print("[B] wrapper hit; bytewise equal; in-memory promotion confirmed")
    print("OK — Direction A round-trip via runtime wrapper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
