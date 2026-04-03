from __future__ import annotations

import wave
import tempfile
from pathlib import Path

from core.audio_utils import TARGET_SAMPLE_RATE, record_wav
from asr.vosk_asr.vosk_asr import VoskASR


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VOSK_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vosk"
    / "vosk-model-small-en-us-0.15"
)


def _read_wav_pcm_bytes(wav_path: str | Path) -> bytes:
    wav_path = Path(wav_path)

    with wave.open(str(wav_path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        num_frames = wf.getnframes()

        print("DEBUG WAV INFO")
        print("channels:", channels)
        print("sample_width:", sample_width)
        print("frame_rate:", frame_rate)
        print("num_frames:", num_frames)

        if channels != 1:
            raise ValueError(f"Expected mono WAV, got {channels} channels")

        if sample_width != 2:
            raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}")

        if frame_rate != TARGET_SAMPLE_RATE:
            raise ValueError(
                f"Expected {TARGET_SAMPLE_RATE} Hz WAV, got {frame_rate} Hz"
            )

        return wf.readframes(num_frames)


def transcribe_microphone(
    duration: float = 3.0,
    model_path: str | Path = DEFAULT_VOSK_MODEL_PATH,
) -> tuple[str, Path]:
    temp_dir = Path(tempfile.gettempdir())
    wav_path = temp_dir / "vosk_asr_input.wav"

    record_wav(wav_path, duration=duration, sample_rate=TARGET_SAMPLE_RATE)

    asr = VoskASR(model_path=model_path, sample_rate=TARGET_SAMPLE_RATE)
    pcm_bytes = _read_wav_pcm_bytes(wav_path)
    text = asr.transcribe_wav_bytes(pcm_bytes)

    return text, wav_path


def transcribe_wav_file(
    wav_path: str | Path,
    model_path: str | Path = DEFAULT_VOSK_MODEL_PATH,
) -> str:
    asr = VoskASR(model_path=model_path, sample_rate=TARGET_SAMPLE_RATE)
    pcm_bytes = _read_wav_pcm_bytes(wav_path)
    return asr.transcribe_wav_bytes(pcm_bytes)