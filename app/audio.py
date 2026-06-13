"""Encode the model's 24 kHz float audio into delivery formats via ffmpeg.

ffmpeg (already in the image) handles resampling and codecs, so we avoid pulling
in audio-encoding Python deps. Telephony formats force 8 kHz mono:
  - wav_8k   : 8 kHz PCM16 WAV
  - ulaw_8k  : 8 kHz G.711 µ-law WAV (classic narrowband telephony)
"""
import subprocess

import numpy as np

# format -> (ffmpeg output args, forced sample rate or None, content type, ext)
_FORMATS = {
    "wav":     (["-c:a", "pcm_s16le", "-f", "wav"], None, "audio/wav", "wav"),
    "wav_8k":  (["-c:a", "pcm_s16le", "-f", "wav"], 8000, "audio/wav", "wav"),
    "ulaw_8k": (["-c:a", "pcm_mulaw", "-f", "wav"], 8000, "audio/wav", "wav"),
    "mp3":     (["-c:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"], None, "audio/mpeg", "mp3"),
    "ogg":     (["-c:a", "libvorbis", "-f", "ogg"], None, "audio/ogg", "ogg"),
}


def supported_formats() -> list[str]:
    return list(_FORMATS)


def content_type(fmt: str) -> str:
    return _FORMATS[fmt][2]


def file_ext(fmt: str) -> str:
    return _FORMATS[fmt][3]


def encode(audio: np.ndarray, in_sr: int, fmt: str, sample_rate: int) -> bytes:
    if fmt not in _FORMATS:
        raise ValueError(f"Unsupported format: {fmt}")
    out_args, forced_sr, _, _ = _FORMATS[fmt]
    out_sr = forced_sr or sample_rate

    pcm = np.ascontiguousarray(np.clip(audio, -1.0, 1.0), dtype=np.float32).tobytes()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(in_sr), "-ac", "1", "-i", "pipe:0",
        "-ar", str(out_sr), "-ac", "1", *out_args, "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')}")
    return proc.stdout
