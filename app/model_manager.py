"""Lazy, thread-safe model singleton with idle eviction.

StyleTTS2 (model + voice styles) is expensive to load, so we load it once and
reuse it. To avoid pinning VRAM forever, an idle timer evicts a model that has
not been used for ``MODEL_IDLE_TIMEOUT_MIN`` minutes; it is transparently
reloaded on the next request.

Ported from the audio-transcribe-api project to keep the two services aligned.
"""
import gc
import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("app.models")


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class LazyModel:
    """Holds a single lazily-loaded object guarded by a lock."""

    def __init__(self, name: str, loader: Callable[[], object], idle_timeout_sec: int):
        self._name = name
        self._loader = loader
        self._idle_timeout = idle_timeout_sec
        self._obj: Optional[object] = None
        self._last_used = 0.0
        self._lock = threading.Lock()

    def get(self) -> object:
        with self._lock:
            if self._obj is None:
                logger.info("Loading model: %s", self._name)
                self._obj = self._loader()
            self._last_used = time.monotonic()
            return self._obj

    def maybe_evict(self) -> bool:
        if self._idle_timeout <= 0:
            return False
        with self._lock:
            if self._obj is None:
                return False
            idle = time.monotonic() - self._last_used
            if idle < self._idle_timeout:
                return False
            logger.info("Evicting idle model: %s (idle %.0fs)", self._name, idle)
            self._obj = None
        gc.collect()
        _empty_cuda_cache()
        return True

    def is_loaded(self) -> bool:
        with self._lock:
            return self._obj is not None


_registry: list[LazyModel] = []


def register(model: LazyModel) -> LazyModel:
    _registry.append(model)
    return model


def evict_idle() -> None:
    for model in _registry:
        model.maybe_evict()
