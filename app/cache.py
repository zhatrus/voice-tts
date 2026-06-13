"""Tiny on-disk audio cache.

IVR prompts repeat a lot, so caching rendered audio keyed by the full request
(text + voice + speed + format + sample rate) turns repeats into a file read.
Controlled by CACHE_ENABLED.
"""
import hashlib
import logging

from .config import get_settings

logger = logging.getLogger("app.cache")


def _key(text: str, voice: str, speed: float, fmt: str, sample_rate: int) -> str:
    raw = f"{text}\x00{voice}\x00{speed}\x00{fmt}\x00{sample_rate}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(text: str, voice: str, speed: float, fmt: str, sample_rate: int, ext: str) -> bytes | None:
    s = get_settings()
    if not s.cache_enabled:
        return None
    path = s.cache_dir / f"{_key(text, voice, speed, fmt, sample_rate)}.{ext}"
    if path.exists():
        return path.read_bytes()
    return None


def put(text: str, voice: str, speed: float, fmt: str, sample_rate: int, ext: str, data: bytes) -> None:
    s = get_settings()
    if not s.cache_enabled:
        return
    s.cache_dir.mkdir(parents=True, exist_ok=True)
    path = s.cache_dir / f"{_key(text, voice, speed, fmt, sample_rate)}.{ext}"
    try:
        path.write_bytes(data)
    except Exception:
        logger.exception("Failed to write cache file %s", path)
