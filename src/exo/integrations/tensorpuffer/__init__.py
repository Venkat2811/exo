"""Tensorpuffer integration for exo's MLX KV cache.

Two integration modes (matching the llama.cpp / vllm.rs naming):

- **Direction A**: external Python harness imports
  :class:`TensorpufferKVPrefixCache` and constructs it instead of the
  vanilla :class:`exo.worker.engines.mlx.cache.KVPrefixCache`. No exo
  source changes; the harness owns the swap.

- **Direction B**: in-tree patches inside
  :mod:`exo.worker.engines.mlx.cache`. The vanilla ``KVPrefixCache``
  picks up tensorpuffer hooks automatically when ``TPUF_KVBM_ENABLE=1``
  is set in the environment. No external harness needed.

Both directions share the same C ABI (``libtensorpuffer.dylib`` from
``crates/tp-cabi``) and the same MLX cache codec.
"""

from exo.integrations.tensorpuffer.client import (
    Tensorpuffer,
    TensorpufferError,
    is_enabled,
)

__all__ = ["Tensorpuffer", "TensorpufferError", "is_enabled"]
