# Run this file from the root directory
# cd C:\diss\Dissertation-Workspace
# python -m asr.whisper_asr.whisper_integration

from __future__ import annotations

from pathlib import Path
import tempfile

from core.audio_utils import TARGET_SAMPLE_RATE, record_wav
from asr.whisper_asr.whisper_asr import WhisperASR


def transcribe_microphone(
    duration: float = 3.0,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
) -> tuple[str, Path]:
    """
    Record from microphone, save a temporary WAV, and transcribe it with Whisper.
    Returns:
        (transcript_text, wav_path)
    """
    temp_dir = Path(tempfile.gettempdir())
    wav_path = temp_dir / "whisper_asr_input.wav"

    record_wav(wav_path, duration=duration, sample_rate=TARGET_SAMPLE_RATE)

    asr = WhisperASR(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    text = asr.transcribe_file(wav_path)

    return text, wav_path


def transcribe_wav_file(
    wav_path: str | Path,
    model_size: str = "base",
    device: str = "cuda",
    compute_type: str = "float16",
) -> str:
    asr = WhisperASR(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
    )
    return asr.transcribe_file(wav_path)


if __name__ == "__main__":
    transcript, wav_path = transcribe_microphone(duration=5.0)
    print("WAV:", wav_path)
    print("Transcript:", transcript)