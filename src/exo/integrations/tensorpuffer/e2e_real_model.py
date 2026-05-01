"""End-to-end real-model proof for the tensorpuffer integration.

Loads ``Qwen/Qwen3-0.6B`` (already in the local HF cache on dev
machines) via mlx-lm, runs a real prefill via
``mlx_lm.models.cache.make_prompt_cache`` + the model's forward pass,
and threads the resulting KV cache through exo's ``KVPrefixCache``.

The script accepts a ``--phase`` argument so the same code can drive
both processes:

    --phase=cold    (process A) populate. Builds a fresh
                    KVPrefixCache, runs the forward pass, calls
                    add_kv_cache → tensorpuffer stash.
    --phase=warm    (process B) consume. Builds a fresh
                    KVPrefixCache, queries get_kv_cache, asserts
                    the puffer fast-path fired (matched_idx is set,
                    is_exact is True, no model.forward needed).

Both phases time the prefill and print the result. The wrapping
shell harness runs them in sequence, kills the model between them,
and reports the cold/warm speedup.

Mode selection mirrors Direction A vs B:

    --mode=B  (default)  use the in-tree hook in
                         exo.worker.engines.mlx.cache.KVPrefixCache.
                         TPUF_KVBM_ENABLE must be set BEFORE this
                         module is imported (so the in-tree hook is
                         live at __init__ time).

    --mode=A             use the external HarnessTpufWrapper from
                         direction_a_harness.py. KVPrefixCache stays
                         vanilla; the wrapper handles stash/load.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import KVCache, make_prompt_cache

# Defer heavy imports until after env is set up
def _load_kvprefix_cache():
    from exo.worker.engines.mlx.cache import KVPrefixCache  # noqa: WPS433

    return KVPrefixCache


def _load_wrapper():
    from exo.integrations.tensorpuffer.direction_a_harness import (
        HarnessTpufWrapper,
    )

    return HarnessTpufWrapper


def _load_model(repo: str):
    print(f"loading {repo} via mlx_lm.load() …", flush=True)
    t0 = time.time()
    model, tokenizer = mlx_lm.load(repo)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    return model, tokenizer


def _prefill(model, prompt_tokens: mx.array, cache):
    """Run a real forward pass over `prompt_tokens` to populate `cache`.
    Returns wall-clock seconds.
    """
    t0 = time.time()
    inputs = prompt_tokens[None, :]   # add batch dim
    _ = model(inputs, cache=cache)
    mx.eval(*[c.keys for c in cache if hasattr(c, "keys") and c.keys is not None])
    return time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["cold", "warm"], required=True)
    ap.add_argument("--mode", choices=["A", "B"], default="B")
    ap.add_argument(
        "--repo",
        default="Qwen/Qwen3-0.6B",
        help="HF repo id, mlx-loadable",
    )
    ap.add_argument("--prompt-tokens", type=int, default=128)
    args = ap.parse_args()

    # Sanity: env must be set up by the caller
    for required in ("TPUF_S3_ENDPOINT", "TPUF_S3_BUCKET", "TPUF_S3_ACCESS_KEY"):
        if required not in os.environ:
            print(f"missing env: {required}", file=sys.stderr)
            return 2

    # Direction B requires TPUF_KVBM_ENABLE before KVPrefixCache import.
    # The wrapping shell harness is responsible for that; verify here.
    if args.mode == "B" and os.environ.get("TPUF_KVBM_ENABLE", "") != "1":
        print("--mode=B requires TPUF_KVBM_ENABLE=1", file=sys.stderr)
        return 2

    model, tokenizer = _load_model(args.repo)

    # Build a deterministic prompt of approximately --prompt-tokens
    # tokens by repeating a paragraph until we cross the threshold.
    paragraph = (
        "TensorPuffer reuses KV-cache state across processes by stashing "
        "rkyv-archived layer bytes through foyer to S3, keyed by "
        "BLAKE3 over (engine_domain, model_id, token_ids). "
    )
    text = paragraph
    while True:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= args.prompt_tokens:
            break
        text += paragraph
    prompt_tokens = mx.array(ids[: args.prompt_tokens], dtype=mx.int32)
    print(f"prompt: {len(prompt_tokens)} tokens", flush=True)

    KVPrefixCache = _load_kvprefix_cache()

    if args.mode == "B":
        cache_view = KVPrefixCache(group=None)
        if cache_view._tpuf is None:
            print("ERROR: in-tree _tpuf is None — Direction B not active", file=sys.stderr)
            return 3
    else:
        cache_view = KVPrefixCache(group=None)
        Wrapper = _load_wrapper()
        cache_view = Wrapper(cache_view, model_id=os.environ.get("TPUF_KVBM_MODEL_ID", "exo-e2e"))

    fake_model = SimpleNamespace(layers=model.layers, make_cache=lambda: make_prompt_cache(model))

    if args.phase == "cold":
        # Cold path: ask the cache for a hit (expect MISS), then run a
        # real prefill and call add_kv_cache to populate puffer.
        out_cache, remaining, idx, is_exact = cache_view.get_kv_cache(fake_model, prompt_tokens)
        if idx is not None:
            print("WARNING: cold phase saw an existing entry — bucket not clean", flush=True)
        # `remaining == prompt_tokens` on a miss
        cache = make_prompt_cache(model)
        wall = _prefill(model, prompt_tokens, cache)
        print(f"COLD prefill {len(prompt_tokens)} tokens in {wall*1000:.1f} ms")
        # Mode A wraps add/get; mode B uses the inner add directly.
        if args.mode == "B":
            cache_view.add_kv_cache(prompt_tokens, cache)
        else:
            cache_view.add_kv_cache(prompt_tokens, cache)
        print("COLD stashed via puffer")
        return 0

    # phase == warm — profile the steps so we can see where the time goes
    from exo.integrations.tensorpuffer.client import Tensorpuffer
    from exo.integrations.tensorpuffer.codec import decode as codec_decode

    # Probe the puffer directly so we can split the load time from the
    # codec time.
    direct_tp = Tensorpuffer()
    t0 = time.time()
    blob = direct_tp.try_load_prefix(
        os.environ.get("TPUF_KVBM_MODEL_ID", "exo-e2e"),
        [int(t) for t in prompt_tokens.tolist()],
    )
    t_probe = time.time() - t0
    if blob is None:
        print("FAIL: direct probe missed", file=sys.stderr)
        return 5
    print(f"  step direct probe (tpuf load): {t_probe*1000:.1f} ms ({len(blob):,} bytes)")
    t0 = time.time()
    decoded = codec_decode(blob)
    t_decode = time.time() - t0
    print(f"  step codec.decode:             {t_decode*1000:.1f} ms")
    direct_tp.free()

    # Now the full integrated path through KVPrefixCache
    t0 = time.time()
    out_cache, remaining, idx, is_exact = cache_view.get_kv_cache(fake_model, prompt_tokens)
    wall = time.time() - t0
    print(f"WARM get_kv_cache wall = {wall*1000:.1f} ms")
    if idx is None or not is_exact:
        print(
            f"FAIL: warm phase did not hit puffer (idx={idx}, is_exact={is_exact})",
            file=sys.stderr,
        )
        return 4
    print(f"WARM hit: idx={idx} is_exact={is_exact} cache_layers={len(out_cache)}")

    # Sanity: the loaded cache should let us decode the next token without
    # re-running prefill from scratch. We validate by sampling 1 step.
    next_input = remaining[None, :]
    t0 = time.time()
    logits = model(next_input, cache=out_cache)
    mx.eval(logits)
    decode_wall = time.time() - t0
    next_id = int(mx.argmax(logits[0, -1, :]).item())
    print(
        f"WARM 1-step decode after restore: {decode_wall*1000:.1f} ms, "
        f"next_token_id={next_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
