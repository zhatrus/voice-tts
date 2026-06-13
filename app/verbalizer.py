"""Ukrainian text verbalization: expand numbers, dates and acronyms to words.

Wraps the CTranslate2 m2m100 model ``skypro1111/m2m100-ukr-verbalization-ct2``
(the same model the upstream StyleTTS2 Ukrainian Space uses). This matters a lot
for telephony: without it the engine would mispronounce "25,50 грн", phone
numbers, dates, etc.

The whole component is best-effort: if loading or inference fails, callers fall
back to the raw text, so the core TTS keeps working even when VERBALIZE is on.
"""
import logging

logger = logging.getLogger("app.verbalizer")

_REPO = "skypro1111/m2m100-ukr-verbalization-ct2"


class Verbalizer:
    """Lazy CTranslate2 m2m100 number/acronym expander (runs on CPU, int8)."""

    def __init__(self) -> None:
        import ctranslate2
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        model_path = snapshot_download(repo_id=_REPO)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._tokenizer.src_lang = "uk"
        self._translator = ctranslate2.Translator(
            model_path, device="cpu", compute_type="int8"
        )
        self._lang_token = self._tokenizer.lang_code_to_token["uk"]

    def process(self, text: str) -> str:
        text = text.strip()
        if not text:
            return text
        source = self._tokenizer.convert_ids_to_tokens(self._tokenizer.encode(text))
        results = self._translator.translate_batch(
            [source],
            target_prefix=[[self._lang_token]],
            beam_size=4,
            max_decoding_length=512,
        )
        target = results[0].hypotheses[0][1:]  # drop the language token
        return self._tokenizer.decode(self._tokenizer.convert_tokens_to_ids(target))


_verbalizer: Verbalizer | None = None
_failed = False


def verbalize(text: str) -> str:
    """Expand numbers/acronyms; on any failure return the input unchanged."""
    global _verbalizer, _failed
    if _failed:
        return text
    try:
        if _verbalizer is None:
            logger.info("Loading verbalizer model: %s", _REPO)
            _verbalizer = Verbalizer()
        return _verbalizer.process(text)
    except Exception:
        logger.exception("Verbalization failed — falling back to raw text")
        _failed = True
        return text
