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

## Status

- [x] C ABI ctypes wrapper, ABI version check, last-error mirror
- [x] Codec for vanilla `KVCache` (encode/decode, bytewise round-trip)
- [x] Direction B in-tree hooks (gated by `TPUF_KVBM_ENABLE`)
- [x] Direction A composition wrapper
- [x] Synthetic 28-layer × 220-token bf16 round-trip verified
- [ ] Live mlx-lm model bench (cold A → fresh-process B with puffer hit)
- [ ] Cross-process bench script that mirrors the vllm.rs / llama.cpp
      stress — n=8 prompts, p50/p99
- [ ] Codec support for `RotatingKVCache`, `QuantizedKVCache`,
      SSM caches, `DeepseekV4Cache`
- [ ] Multi-shard (distributed) story — exo splits the model across
      devices; KV state per shard needs its own stash key

## Related

- vllm.rs side: `vllm.rs:feat/tensorpuffer-kvbm` (commits `0f37d48`,
  `8643f2d`).
- llama.cpp side: `tp-cabi` (cdylib), `tp-llamacpp-{ffi,harness}`
  (Direction A), `tools/tensorpuffer-bench` upstream patch
  (Direction B).
- Tensorpuffer architecture diagrams:
  `0_venkat-worklog/kanban/artifacts/RFC-0008/architecture-diagrams.md`
  (in the tensorpuffer repo).
