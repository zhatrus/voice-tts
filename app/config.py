"""Centralized configuration loaded from environment variables."""
import os
from functools import lru_cache
from pathlib import Path


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def resolve_device(raw: str | None = None) -> str:
    """Map user-facing DEVICE values to a valid torch device ("cuda" | "cpu")."""
    value = (raw if raw is not None else os.getenv("DEVICE", "gpu")).strip().lower()
    wants_gpu = value in {"gpu", "cuda", "auto"}
    if wants_gpu:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return "cpu"


class Settings:
    def __init__(self) -> None:
        # Engine / models
        self.model_repo = os.getenv("MODEL_REPO", "patriotyk/styletts2_ukrainian_multispeaker")
        self.voices_space_repo = os.getenv("VOICES_SPACE_REPO", "patriotyk/styletts2-ukrainian")
        self.device = resolve_device()
        self.hf_token = os.getenv("HF_TOKEN") or None

        # Keep the loaded model in VRAM for this many idle minutes before
        # evicting it. 0 disables eviction (model stays loaded forever).
        self.model_idle_timeout_min = _get_int("MODEL_IDLE_TIMEOUT_MIN", 30)

        # Output defaults
        self.default_voice = os.getenv("DEFAULT_VOICE") or None
        self.default_format = os.getenv("DEFAULT_FORMAT", "wav")
        self.default_sample_rate = _get_int("DEFAULT_SAMPLE_RATE", 24000)

        # Text front-end
        self.verbalize = _get_bool("VERBALIZE", True)
        self.max_text_len = _get_int("MAX_TEXT_LEN", 3000)

        # Cache
        self.cache_enabled = _get_bool("CACHE_ENABLED", True)

        # Storage
        base = Path(os.getenv("DATA_DIR", "/tmp/ukrainian-voice"))
        self.data_dir = base
        self.cache_dir = Path(os.getenv("CACHE_DIR", str(base / "cache")))
        self.clones_dir = Path(os.getenv("CLONES_DIR", str(base / "clones")))

        # Security
        self.api_key = os.getenv("API_KEY") or None

        # Server
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = _get_int("PORT", 8000)
        self.workers = _get_int("WORKERS", 1)
        self.log_level = os.getenv("LOG_LEVEL", "info")

    # The native sample rate StyleTTS2 produces.
    NATIVE_SAMPLE_RATE = 24000

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clones_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
