"""StyleTTS2 Ukrainian engine: load, synthesize, and clone voices.

The model and all voice styles are held together in one lazily-loaded object so
they load and evict as a unit (see model_manager). Synthesis follows the exact
front-end the model was trained with (see text.py).
"""
import logging
import threading

from . import voices
from .config import get_settings
from .model_manager import LazyModel, register
from .text import to_ipa_chunks

logger = logging.getLogger("app.tts")

# The model is not safe to call from several threads at once; serialize all
# inference (synthesis + cloning) through one lock.
_synth_lock = threading.Lock()

# Short neutral phrase used only to drive predict_style_multi when cloning.
_CLONE_SEED_TEXT = "Привіт, це зразок голосу для синтезу українською мовою."


class Engine:
    def __init__(self, model, styles: dict, device: str):
        self.model = model
        self.styles = styles
        self.device = device


def _load_engine() -> Engine:
    from styletts2_inference.models import StyleTTS2

    s = get_settings()
    logger.info("Init StyleTTS2 repo=%s device=%s", s.model_repo, s.device)
    model = StyleTTS2(hf_path=s.model_repo, device=s.device)
    styles = voices.load_styles(s.device)
    return Engine(model, styles, s.device)


_engine = register(
    LazyModel(
        "styletts2-uk",
        _load_engine,
        idle_timeout_sec=get_settings().model_idle_timeout_min * 60,
    )
)


def preload() -> None:
    """Eagerly load the model (used at startup to surface config errors early)."""
    _engine.get()


def is_loaded() -> bool:
    return _engine.is_loaded()


def synthesize(text: str, voice: str, speed: float = 1.0):
    """Return (sample_rate, float32 mono numpy array) at the native 24 kHz."""
    import torch

    eng: Engine = _engine.get()  # type: ignore[assignment]
    if voice not in eng.styles:
        raise KeyError(voice)
    style = eng.styles[voice]

    chunks = to_ipa_chunks(text, verbalize=get_settings().verbalize)
    if not chunks:
        raise ValueError("Text produced no speakable content")

    wavs = []
    with _synth_lock:
        for ps in chunks:
            tokens = eng.model.tokenizer.encode(ps)
            wavs.append(eng.model(tokens, speed=speed, s_prev=style))

    audio = torch.concatenate(wavs).cpu().numpy()
    return get_settings().NATIVE_SAMPLE_RATE, audio


def clone_voice(name: str, reference_wav_path: str) -> str:
    """Compute a style from a reference recording, persist it, return its id."""
    eng: Engine = _engine.get()  # type: ignore[assignment]
    seed_chunks = to_ipa_chunks(_CLONE_SEED_TEXT, verbalize=False)
    with _synth_lock:
        tokens = eng.model.tokenizer.encode(seed_chunks[0])
        style = eng.model.predict_style_multi(reference_wav_path, tokens)

    voice_id = voices.save_clone_style(name, style)
    eng.styles[voice_id] = style  # available immediately, no reload needed
    return voice_id
