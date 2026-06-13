"""Voice style registry: preset speakers (downloaded) + cloned voices (on disk).

A "voice" in StyleTTS2 is a small style tensor passed to the model as ``s_prev``.
Presets ship as ``voices/*.pt`` inside the upstream HF Space; cloned voices are
style tensors we compute from a reference recording and store under
``DATA_DIR/clones``.
"""
import glob
import logging
import os
import re

from .config import get_settings

logger = logging.getLogger("app.voices")

_preset_dir: str | None = None


def _presets_dir() -> str:
    """Download (once, cached) the preset style files and return their folder."""
    global _preset_dir
    if _preset_dir is not None:
        return _preset_dir
    from huggingface_hub import snapshot_download

    s = get_settings()
    root = snapshot_download(
        repo_id=s.voices_space_repo,
        repo_type="space",
        allow_patterns=["voices/*.pt"],
        token=s.hf_token,
    )
    _preset_dir = os.path.join(root, "voices")
    return _preset_dir


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def safe_clone_id(name: str) -> str:
    """Filesystem-safe id for a clone, preserving Ukrainian letters."""
    cleaned = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    return cleaned or "clone"


def list_preset_names() -> list[str]:
    return sorted(_stem(p) for p in glob.glob(os.path.join(_presets_dir(), "*.pt")))


def list_clone_names() -> list[str]:
    s = get_settings()
    return sorted(_stem(p) for p in glob.glob(str(s.clones_dir / "*.pt")))


def list_voices() -> dict:
    presets = list_preset_names()
    clones = list_clone_names()
    return {"presets": presets, "clones": clones, "count": len(presets) + len(clones)}


def load_styles(device: str) -> dict:
    """Load all preset + clone style tensors onto ``device``."""
    import torch

    styles: dict[str, object] = {}
    for path in glob.glob(os.path.join(_presets_dir(), "*.pt")):
        styles[_stem(path)] = torch.load(path, map_location=device)

    s = get_settings()
    for path in glob.glob(str(s.clones_dir / "*.pt")):
        # clones override presets if names collide
        styles[_stem(path)] = torch.load(path, map_location=device)

    logger.info("Loaded %d voice styles (%s presets + clones)", len(styles), device)
    return styles


def save_clone_style(name: str, style) -> str:
    """Persist a computed clone style tensor; returns its voice id."""
    import torch

    s = get_settings()
    s.clones_dir.mkdir(parents=True, exist_ok=True)
    voice_id = safe_clone_id(name)
    path = s.clones_dir / f"{voice_id}.pt"
    torch.save(style, path)
    logger.info("Saved clone voice '%s' → %s", voice_id, path)
    return voice_id


def resolve_default_voice() -> str | None:
    s = get_settings()
    if s.default_voice:
        return s.default_voice
    names = list_preset_names()
    return names[0] if names else None
