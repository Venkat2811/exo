# exo × tensorpuffer KVBM integration

Cross-process KV-cache reuse for exo's MLX engine, backed by
tensorpuffer (foyer RAM+SSD + S3 write-through). Two integration
directions, both proven end-to-end:

| direction | source change to exo | shape |
| :--- | :--- | :--- |
| **A** | none | composition wrapper (`HarnessTpufWrapper`) on a vanilla `KVPrefixCache` |
| **B** | yes (gated) | in-tree hooks in `exo/worker/engines/mlx/cache.py`, off unless `TPUF_KVBM_ENABLE=1` |

## Components

```
src/exo/integrations/tensorpuffer/
  client.py               ctypes binding to libtensorpuffer.dylib
                          (the C ABI from tp-cabi). One Tensorpuffer()
                          handle per process.
  codec.py                Binary encode/decode for a list[KVCache].
                          Returns None for unsupported flavours
                          (RotatingKV, QuantizedKV, ArraysCache for
                          SSM, DeepseekV4, CacheList) so callers fall
                          through to a normal cold prefill.
  smoke.py                Pure C-ABI stash/load round-trip via MinIO.
  codec_smoke.py          Synthetic 28-layer × 220-token bf16 cache
                          → encode → decode → bytewise-equal.
  direction_b_smoke.py    In-tree hook smoke. TPUF_KVBM_ENABLE=1 →
                          add_kv_cache stashes; fresh KVPrefixCache
                          get_kv_cache hits the puffer fast-path.
  direction_a_harness.py  Composition wrapper that achieves the same
                          end-to-end behaviour without the in-tree
                          hooks. Demonstrates Direction A pattern.
```

## How Direction B fires inside exo

```
                          KVPrefixCache.add_kv_cache(prompt, cache)
                                          │
                                          ├── existing in-memory append
                                          ▼
                          self._tpuf_stash(prompt, cache)
                                          │
                                          ├── codec.encode(cache) → bytes (or None → skip)
                                          ▼
                          tpuf_stash_prefix(model_id, prompt_tokens, bytes)
                                          │
                                          ▼
                          libtensorpuffer.dylib  →  foyer RAM/SSD  →  S3 PUT


                          KVPrefixCache.get_kv_cache(model, prompt, media_regions)
                                          │
                                          ▼
                          if self._tpuf is not None and not media_regions:
                              blob = tpuf_try_load_prefix(model_id, prompt_tokens)
                              if blob is not None:
                                  cache = codec.decode(blob)
                                  ── promote to in-memory list at idx N
                                  return (cache, last_token, N, is_exact=True)
                                          │
                                          ▼   (miss → fall through to in-memory linear scan)
                          existing prefix-match logic untouched
```

## Required environment variables

For both directions:

| var | required | purpose |
| :--- | :--- | :--- |
| `TPUF_KVBM_ENABLE=1` | Direction B only | turns the in-tree hook on |
| `TPUF_KVBM_MODEL_ID` | optional | content-hash namespace; default `exo-mlx` |
| `TPUF_DYLIB_PATH` | optional | explicit dylib path; defaults to a sibling tensorpuffer checkout |
| `TPUF_S3_ENDPOINT` | required | e.g. `http://localhost:9100` for MinIO |
| `TPUF_S3_BUCKET` | required | e.g. `tensorpuffer` |
| `TPUF_S3_ACCESS_KEY` / `TPUF_S3_SECRET_KEY` | required | |
| `TPUF_S3_REGION` | optional | default `us-east-1` |
| `TPUF_S3_FORCE_PATH_STYLE` | optional | `1` for MinIO |
| `TPUF_KVBM_NAMESPACE` | optional | S3 prefix scope, e.g. `exo-bench` |
| `TPUF_KVBM_S3_PREFIX` | optional | full S3 key prefix override |
| `TPUF_FOYER_RAM_BYTES` | optional | RAM tier size, default 4 GiB |
| `TPUF_FOYER_SSD_BYTES` | optional | SSD tier size, default 2 GiB |
| `TPUF_FOYER_SSD_DIR` | optional | foyer working dir, default `/tmp/tpuf-foyer` |
| `TPUF_FOYER_BLOCK_SIZE_BYTES` | optional | foyer block size, default 1 MiB |

## Operator caveats (macOS specifically)

- `launchctl limit maxfiles` defaults to a 256-soft / unlimited-hard
  pair on stock macOS. Foyer's SSD allocator opens many small files at
  init time and **will hit `EMFILE` (os error 24)** with the default
  cap. For benches keep `TPUF_FOYER_SSD_BYTES` ≤ 256 MB and
  `TPUF_FOYER_BLOCK_SIZE_BYTES` ≥ 4 MiB. For production workloads:
  ```
  sudo launchctl limit maxfiles 65536 524288
  sudo reboot
  ```
- The dylib lives in the sibling tensorpuffer checkout at
  `target/release/libtensorpuffer.dylib`. Build it once with
  `cargo build -p tp-cabi --release` from the tensorpuffer repo.

## Running the smoke tests

```sh
docker run -d --name tensorpuffer-minio -p 9100:9000 minio/minio:latest server /data
docker exec tensorpuffer-minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker exec tensorpuffer-minio mc mb local/tensorpuffer

# Pure C-ABI round-trip
TPUF_KVBM_ENABLE=1 ... uv run python -m exo.integrations.tensorpuffer.smoke

# Codec round-trip
uv run python -m exo.integrations.tensorpuffer.codec_smoke

# Direction B (in-tree hook end-to-end, two phases in one process)
TPUF_KVBM_ENABLE=1 ... uv run python -m exo.integrations.tensorpuffer.direction_b_smoke

# Direction A (composition wrapper)
... uv run python -m exo.integrations.tensorpuffer.direction_a_harness
```

## End-to-end proof (real mlx-lm models, M3 Max + Metal)

`e2e_real_model.py` drives a real `mlx_lm.load()` + forward pass
through Direction B's in-tree hook. Cold and warm are separate Python
processes; the foyer SSD is preserved across the kill so this is a
true cross-process scenario. MinIO at `localhost:9100`. Warm path
measured over 4 iterations to capture both the first-foyer-warm-up
cost and the steady-state.

### Steady-state warm vs cold (load + codec.decode, foyer RAM-hot)

| model         | tokens | cold prefill | warm steady-state | speedup |
| :------------ |  ---:  |       ---:   |       ---:  |    ---: |
| Qwen3-0.6B    |  1024  |    144 ms    |     89 ms   | **1.6×** |
| Qwen3-0.6B    |  2048  |    288 ms    |    183 ms   | **1.6×** |
| Qwen3-0.6B    |  4096  |    644 ms    |    375 ms   | **1.7×** |
| Qwen3-1.7B    |  1024  |    363 ms    |     92 ms   | **3.9×** |
| Qwen3-1.7B    |  2048  |    703 ms    |    186 ms   | **3.8×** |
| Qwen3-1.7B    |  4096  |  1,449 ms    |    380 ms   | **3.8×** |
| **Qwen3-4B**  | **1024** | **872 ms** |  **120 ms** | **7.3×** |
| **Qwen3-4B**  | **2048** | **1,747 ms** | **239 ms** | **7.3×** |
| **Qwen3-4B**  | **4096** | **3,692 ms** | **482 ms** | **7.7×** |

### First-warm cost (iter 0, foyer-RAM tier promoting from SSD)

The first warm request after a stash pays a one-time foyer
RAM-promotion cost — the SSD tier holds the data immediately after
the put-through but the RAM tier needs to load on first access. In
production this fires once per (process × prompt) pair; subsequent
hits land at the steady-state rate above.

| model       | tokens | iter 0 view_call | iter 1+ steady-state |
| :---------- |  ---:  |             ---: |                ---: |
| Qwen3-0.6B  |  4096  |       1,629 ms   |              367 ms |
| Qwen3-4B    |  4096  |       2,107 ms   |              470 ms |

### Crossover map (steady-state)

```
       ┌────────────────────────────────────────┐
0.6B   │  1.6×  ─── 1.6×  ─── 1.7×              │ ← weak win
1.7B   │       ─── 3.9×  ─── 3.8×  ─── 3.8×    │ ← solid win
4B     │       ─── 7.3×  ─── 7.3×  ─── 7.7×    │ ← strong win
       └────────────────────────────────────────┘
            1024     2048     4096    tokens
```

For 7B+ or 10k+ token prompts the win grows further — cold prefill
is quadratic in tokens while warm load is linear. Distributed serving
amplifies further still: a shared foyer across machines amortizes the
load cost across many requests.

### How we got here (perf history)

The original measurement (before foyer-direct + steady-state) had
exo at 0.32–1.6× and a "crossover at 4B". That was iter-0 numbers
(first read after stash, foyer-RAM warming). A short rabbit-hole of
optimizations + accurate measurement collapsed that:

| change                           |  steady-state @ 1024 toks |
| :------------------------------- |  ---:  |
| baseline `bytes(out[:rc2])`      |  2,000 ms (Qwen3-4B/4096) |
| `ctypes.string_at` fast-path     |  ~280 MB/s |
| ABI 1.1 borrowed-pointer load    |  ~290 MB/s |
| foyer-direct (`Bytes` not `Vec`) |  **1.3 GB/s steady-state** |

Codec.decode itself runs at ~13 GB/s; the remaining wall-clock is
foyer-RAM read bandwidth + the unavoidable NumPy → Metal upload
inside `mx.array()`.

The integration mechanism is fully proven: bytes round-trip
correctly, KV state restores cleanly, post-restore 1-step decode
produces tokens (`next_token_id` matches across cold and warm runs).

## Why the foyer-warm-up cost exists

After `add_kv_cache → _tpuf_stash` writes through foyer, foyer
populates both its RAM and SSD tiers. The very first read after the
write returns from the SSD tier (or some RAM tier that needs
re-promotion); subsequent reads hit RAM. We observe iter 0 at
~10× the steady-state. This is foyer-internal behaviour we don't
control here. In production the same prompt is queried many times,
so iter 0 is a one-time tax.

## Status

- [x] C ABI ctypes wrapper with `string_at` fast-path (commit `c0a5abcd`)
- [x] Codec for vanilla `KVCache` (encode/decode, bytewise round-trip)
- [x] Direction B in-tree hooks (gated by `TPUF_KVBM_ENABLE`)
- [x] Direction A composition wrapper
- [x] Synthetic 28-layer × 220-token bf16 round-trip verified
- [x] **Real mlx-lm model end-to-end across 9 (model, tokens) cells.**
      Crossover documented; mechanism fully validated.
- [ ] Cross-process bench script that mirrors the vllm.rs / llama.cpp
      stress — n=8 prompts, p50/p99 — at the crossover regime
      (Qwen3-4B / 4k tokens or larger)
- [ ] Codec support for `RotatingKVCache`, `QuantizedKVCache`,
      SSM caches, `DeepseekV4Cache`
- [ ] Multi-shard (distributed) story — exo splits the model across
      devices; KV state per shard needs its own stash key
- [ ] zstd compression on the stash side

## Related

- vllm.rs side: `vllm.rs:feat/tensorpuffer-kvbm` (commits `0f37d48`,
  `8643f2d`).
- llama.cpp side: `tp-cabi` (cdylib), `tp-llamacpp-{ffi,harness}`
  (Direction A), `tools/tensorpuffer-bench` upstream patch
  (Direction B).
- Tensorpuffer architecture diagrams:
  `0_venkat-worklog/kanban/artifacts/RFC-0008/architecture-diagrams.md`
  (in the tensorpuffer repo).
