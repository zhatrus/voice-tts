"""Ukrainian Voice (TTS) API — synchronous text-to-speech.

Endpoints:
  GET  /health              -> {"status": "ok"}
  GET  /voices              -> available preset + cloned voices
  POST /tts                 -> synthesize speech, returns an audio file
  POST /voices/clone        -> create a voice from a reference recording
  POST /v1/audio/speech     -> OpenAI-compatible alias of /tts

TTS is fast enough to answer in one request, so there is no job queue. The model
is loaded once and evicted after an idle period (see model_manager).
"""
import asyncio
import logging
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from . import audio, cache, tts, voices
from .auth import require_api_key
from .config import get_settings
from .model_manager import evict_idle

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "info").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("app.main")

# OpenAI response_format -> our format names
_OPENAI_FORMAT = {"mp3": "mp3", "wav": "wav", "pcm": "wav", "opus": "ogg", "flac": "wav", "aac": "mp3"}


async def _maintenance_loop() -> None:
    while True:
        await asyncio.sleep(300)
        try:
            evict_idle()
        except Exception:  # noqa: BLE001
            logger.exception("Maintenance loop iteration failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    s.ensure_dirs()
    logger.info("Starting: repo=%s device=%s data=%s", s.model_repo, s.device, s.data_dir)

    async def _warm():
        try:
            await asyncio.get_running_loop().run_in_executor(None, tts.preload)
            logger.info("Model preloaded")
        except Exception:  # noqa: BLE001
            logger.exception("Model preload failed (will retry on first request)")

    tasks = [
        asyncio.create_task(_maintenance_loop()),
        asyncio.create_task(_warm()),
    ]
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="Ukrainian Voice (TTS) API", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    speed: float = 1.0
    format: str | None = None
    sample_rate: int | None = None


class OpenAISpeechRequest(BaseModel):
    input: str
    voice: str | None = None
    response_format: str | None = None
    speed: float = 1.0
    model: str | None = None


def _resolve_params(voice, fmt, sample_rate):
    s = get_settings()
    voice = voice or voices.resolve_default_voice()
    if not voice:
        raise HTTPException(status_code=503, detail="No voices available yet")
    fmt = fmt or s.default_format
    if fmt not in audio.supported_formats():
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{fmt}'. Use one of: {audio.supported_formats()}",
        )
    sample_rate = sample_rate or s.default_sample_rate
    return voice, fmt, sample_rate


def _render(text: str, voice: str, speed: float, fmt: str, sample_rate: int) -> bytes:
    """Blocking: synthesize + encode. Runs in a threadpool."""
    in_sr, wav = tts.synthesize(text, voice, speed)
    return audio.encode(wav, in_sr, fmt, sample_rate)


async def _speak(text: str, voice, speed: float, fmt, sample_rate) -> Response:
    s = get_settings()
    text = (text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")
    if len(text) > s.max_text_len:
        raise HTTPException(status_code=413, detail=f"Text longer than {s.max_text_len} chars")

    voice, fmt, sample_rate = _resolve_params(voice, fmt, sample_rate)
    ext = audio.file_ext(fmt)

    cached = cache.get(text, voice, speed, fmt, sample_rate, ext)
    if cached is not None:
        data = cached
    else:
        try:
            data = await asyncio.get_running_loop().run_in_executor(
                None, _render, text, voice, speed, fmt, sample_rate
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown voice: {voice}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        cache.put(text, voice, speed, fmt, sample_rate, ext, data)

    return Response(
        content=data,
        media_type=audio.content_type(fmt),
        headers={"Content-Disposition": f'inline; filename="speech.{ext}"'},
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": tts.is_loaded()}


@app.get("/voices", dependencies=[Depends(require_api_key)])
def list_voices_endpoint():
    return voices.list_voices()


@app.post("/tts", dependencies=[Depends(require_api_key)])
async def tts_endpoint(req: TTSRequest):
    return await _speak(req.text, req.voice, req.speed, req.format, req.sample_rate)


@app.post("/v1/audio/speech", dependencies=[Depends(require_api_key)])
async def openai_speech_endpoint(req: OpenAISpeechRequest):
    fmt = _OPENAI_FORMAT.get((req.response_format or "").lower()) if req.response_format else None
    return await _speak(req.input, req.voice, req.speed, fmt, None)


@app.post("/voices/clone", dependencies=[Depends(require_api_key)])
async def clone_endpoint(name: str = Form(...), file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    suffix = os.path.splitext(file.filename)[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, dir=str(get_settings().data_dir))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())
        voice_id = await asyncio.get_running_loop().run_in_executor(
            None, tts.clone_voice, name, tmp_path
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    logger.info("Cloned voice '%s'", voice_id)
    return {"voice_id": voice_id}


if __name__ == "__main__":
    import uvicorn

    s = get_settings()
    uvicorn.run("app.main:app", host=s.host, port=s.port, workers=s.workers, log_level=s.log_level)
