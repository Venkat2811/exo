"""ctypes wrapper around ``libtensorpuffer.dylib`` (the C ABI from
``crates/tp-cabi``).

The dylib path is discovered in this order:

1. ``TPUF_DYLIB_PATH`` env var (explicit override).
2. Common search paths next to the exo checkout:
     ``../tensorpuffer/target/release/libtensorpuffer.dylib``
     ``~/Documents/p/venkat-github/tensorpuffer/target/release/libtensorpuffer.dylib``
3. ``DYLD_LIBRARY_PATH`` / system default (loader search).

The library is loaded once per process (LRU-cached). Returns ``None`` if
no candidate file exists — callers must handle the disabled case.

ABI version expected: 1.x. We refuse to load if the major version
doesn't match.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_EXPECTED_ABI_MAJOR = 1
_DYLIB_NAME = "libtensorpuffer.dylib"


class TensorpufferError(RuntimeError):
    """Raised on any unexpected condition from the C ABI."""


class LoadedBlob:
    """Buffer-protocol wrapper around a ctypes-allocated load buffer.

    Holds the ctypes ``(c_uint8 * N)`` array alive for the lifetime of
    the wrapper so consumers can decode through ``memoryview`` /
    ``np.frombuffer`` without the ``string_at`` memcpy that
    :meth:`Tensorpuffer.try_load_prefix` performs.

    Compression (TPC1 envelope) is handled inside the C ABI — the
    bytes returned here are already decompressed when the codec at
    stash time was Lz4 / Zstd.

    Slicing returns ``memoryview`` slices, which themselves don't copy.
    Iteration / ``__len__`` / ``__getitem__`` all delegate to the
    inner memoryview.
    """

    __slots__ = ("_buf", "_len", "memoryview", "numpy_view")

    def __init__(self, buf, length: int) -> None:
        self._buf = buf
        self._len = length
        self.memoryview = memoryview(buf)[:length]
        # Lazy: only build the numpy view if asked. memoryview is enough
        # for struct.unpack_from + np.frombuffer.
        self.numpy_view = None

    def __len__(self) -> int:
        return self._len

    def as_numpy_uint8(self):
        import numpy as np

        if self.numpy_view is None:
            self.numpy_view = np.frombuffer(self._buf, dtype=np.uint8, count=self._len)
        return self.numpy_view


def _candidate_dylib_paths() -> Iterable[Path]:
    override = os.environ.get("TPUF_DYLIB_PATH")
    if override:
        yield Path(override).expanduser()
    here = Path(__file__).resolve()
    # Walk up to find a sibling tensorpuffer/ checkout
    for ancestor in here.parents:
        candidate = (
            ancestor.parent / "tensorpuffer" / "target" / "release" / _DYLIB_NAME
        )
        if candidate.exists():
            yield candidate
            break
    # Common fixed location on Venkat2811's machine
    yield (
        Path.home()
        / "Documents"
        / "p"
        / "venkat-github"
        / "tensorpuffer"
        / "target"
        / "release"
        / _DYLIB_NAME
    )


@lru_cache(maxsize=1)
def _load_lib() -> ctypes.CDLL | None:
    for path in _candidate_dylib_paths():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            _bind_symbols(lib)
            version = lib.tpuf_abi_version()
            major = (version >> 16) & 0xFFFF
            if major != _EXPECTED_ABI_MAJOR:
                raise TensorpufferError(
                    f"libtensorpuffer ABI major mismatch: have {major}, expected {_EXPECTED_ABI_MAJOR}"
                )
            return lib
    return None


def _bind_symbols(lib: ctypes.CDLL) -> None:
    lib.tpuf_abi_version.restype = ctypes.c_uint32
    lib.tpuf_abi_version.argtypes = []

    lib.tpuf_init_from_env.restype = ctypes.c_void_p
    lib.tpuf_init_from_env.argtypes = []

    lib.tpuf_free.restype = None
    lib.tpuf_free.argtypes = [ctypes.c_void_p]

    lib.tpuf_stash_prefix.restype = ctypes.c_int64
    lib.tpuf_stash_prefix.argtypes = [
        ctypes.c_void_p,                        # handle
        ctypes.c_char_p,                        # model_id
        ctypes.POINTER(ctypes.c_uint32),        # token_ids
        ctypes.c_size_t,                        # n_tokens
        ctypes.POINTER(ctypes.c_uint8),         # state
        ctypes.c_size_t,                        # state_size_bytes
    ]

    lib.tpuf_try_load_prefix.restype = ctypes.c_int64
    lib.tpuf_try_load_prefix.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]

    lib.tpuf_last_error.restype = ctypes.c_char_p
    lib.tpuf_last_error.argtypes = []


def is_enabled() -> bool:
    """``True`` iff ``TPUF_KVBM_ENABLE=1`` and the dylib is loadable."""
    if os.environ.get("TPUF_KVBM_ENABLE", "").lower() not in {"1", "true"}:
        return False
    return _load_lib() is not None


class Tensorpuffer:
    """Thin handle around a single ``tpuf_handle_t`` from the C ABI."""

    def __init__(self) -> None:
        lib = _load_lib()
        if lib is None:
            raise TensorpufferError(
                "libtensorpuffer.dylib not found; set TPUF_DYLIB_PATH or build "
                "the tp-cabi crate."
            )
        self._lib = lib
        h = lib.tpuf_init_from_env()
        if not h:
            raise TensorpufferError(self._last_error() or "tpuf_init_from_env returned NULL")
        self._handle: int | None = h

    @classmethod
    def from_env_or_none(cls) -> "Tensorpuffer | None":
        """Construct iff :func:`is_enabled`. Returns ``None`` otherwise."""
        return cls() if is_enabled() else None

    def stash_prefix(self, model_id: str, token_ids: list[int] | bytes, state: bytes) -> int:
        """Store ``state`` keyed by content-hash of ``(model_id, token_ids)``.

        Returns the number of bytes stashed (== ``len(state)`` on success).
        Raises :class:`TensorpufferError` on I/O failure.
        """
        if self._handle is None:
            raise TensorpufferError("Tensorpuffer handle is closed")
        toks_arr = self._tokens_array(token_ids)
        state_arr = (ctypes.c_uint8 * len(state)).from_buffer_copy(state)
        rc = self._lib.tpuf_stash_prefix(
            ctypes.c_void_p(self._handle),
            model_id.encode("utf-8"),
            toks_arr,
            len(toks_arr),
            state_arr,
            len(state),
        )
        if rc < 0:
            raise TensorpufferError(self._last_error() or f"tpuf_stash_prefix returned {rc}")
        return int(rc)

    def try_load_prefix(self, model_id: str, token_ids: list[int] | bytes) -> bytes | None:
        """Return the stashed bytes for ``(model_id, token_ids)`` or ``None`` on miss.

        Backwards-compatible wrapper that materializes the full blob into a
        Python ``bytes`` object via a single C-level memcpy. For warm-path
        latency-sensitive callers prefer :meth:`try_load_prefix_view`,
        which returns a buffer-protocol object that the codec can decode
        without any Python-side copy of the 100+ MB blob.
        """
        view = self.try_load_prefix_view(model_id, token_ids)
        if view is None:
            return None
        return ctypes.string_at(ctypes.addressof(view._buf), len(view))

    def try_load_prefix_view(
        self, model_id: str, token_ids: list[int] | bytes
    ) -> "LoadedBlob | None":
        """Zero-copy variant of :meth:`try_load_prefix`.

        Returns a :class:`LoadedBlob` whose ``.memoryview`` /
        ``.numpy_view`` exposes the ctypes-allocated buffer directly,
        without the ~100 MB ``string_at`` memcpy. The :class:`LoadedBlob`
        keeps the ctypes array alive for as long as the caller holds a
        reference, so the views stay valid.

        Decoders should use ``np.frombuffer(blob.memoryview, ...)`` or
        ``struct.unpack_from(fmt, blob.memoryview, off)`` — both are
        zero-copy on the buffer protocol.
        """
        if self._handle is None:
            raise TensorpufferError("Tensorpuffer handle is closed")
        toks_arr = self._tokens_array(token_ids)
        # First probe with a zero-size buffer to learn the actual blob size.
        probe = (ctypes.c_uint8 * 0)()
        rc = self._lib.tpuf_try_load_prefix(
            ctypes.c_void_p(self._handle),
            model_id.encode("utf-8"),
            toks_arr,
            len(toks_arr),
            probe,
            0,
        )
        if rc == 0:
            return None
        if rc == -1:
            raise TensorpufferError(self._last_error() or "tpuf_try_load_prefix error")
        size = -rc if rc < 0 else rc
        if size <= 0:
            return None
        out = (ctypes.c_uint8 * size)()
        rc2 = self._lib.tpuf_try_load_prefix(
            ctypes.c_void_p(self._handle),
            model_id.encode("utf-8"),
            toks_arr,
            len(toks_arr),
            out,
            size,
        )
        if rc2 <= 0:
            return None
        return LoadedBlob(out, int(rc2))

    def free(self) -> None:
        if self._handle is not None:
            self._lib.tpuf_free(ctypes.c_void_p(self._handle))
            self._handle = None

    def __del__(self) -> None:
        try:
            self.free()
        except Exception:
            pass

    def _last_error(self) -> str | None:
        ptr = self._lib.tpuf_last_error()
        if not ptr:
            return None
        return ctypes.string_at(ptr).decode("utf-8", errors="replace")

    @staticmethod
    def _tokens_array(tokens: list[int] | bytes) -> ctypes.Array:
        if isinstance(tokens, bytes):
            # Already u32 little-endian packed.
            assert len(tokens) % 4 == 0, "raw token bytes must be a multiple of 4"
            n = len(tokens) // 4
            arr = (ctypes.c_uint32 * n)()
            ctypes.memmove(arr, tokens, len(tokens))
            return arr
        n = len(tokens)
        arr = (ctypes.c_uint32 * n)(*tokens)
        return arr
