"""Smoke test for the ctypes wrapper.

Run from a shell with MinIO + env vars set::

    TPUF_KVBM_ENABLE=1 \
    TPUF_S3_ENDPOINT=http://localhost:9100 TPUF_S3_REGION=us-east-1 \
    TPUF_S3_BUCKET=tensorpuffer TPUF_S3_ACCESS_KEY=minioadmin \
    TPUF_S3_SECRET_KEY=minioadmin TPUF_S3_FORCE_PATH_STYLE=1 \
    TPUF_KVBM_NAMESPACE=exo-smoke \
    TPUF_FOYER_RAM_BYTES=$((512*1024*1024)) \
    TPUF_FOYER_SSD_BYTES=$((2*1024*1024*1024)) \
    TPUF_FOYER_SSD_DIR=/tmp/exo_smoke_foyer \
    python -m exo.integrations.tensorpuffer.smoke

Asserts a stash → load round-trip plus a deliberate miss.
"""

from __future__ import annotations

import os
import sys

from exo.integrations.tensorpuffer.client import Tensorpuffer, is_enabled


def main() -> int:
    if not is_enabled():
        print("DISABLED: TPUF_KVBM_ENABLE=1 not set or dylib missing")
        return 1

    tp = Tensorpuffer()
    model_id = "exo-smoke-llama-3.2-3b"
    tokens = [1, 2, 3, 4, 5, 100, 200, 300, 1234]
    blob = b"\x00\x01\x02\x03 hello tensorpuffer from exo " * 64
    print(f"stashing {len(blob)} bytes under {model_id} / {len(tokens)} tokens")
    n = tp.stash_prefix(model_id, tokens, blob)
    assert n == len(blob), f"stash returned {n}, expected {len(blob)}"

    print("loading by content hash …")
    got = tp.try_load_prefix(model_id, tokens)
    assert got == blob, f"load mismatch: got {len(got) if got else None} bytes"

    print("miss probe (different token list) …")
    miss = tp.try_load_prefix(model_id, tokens + [99999])
    assert miss is None, "expected miss, got bytes"

    tp.free()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
