from __future__ import annotations

from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write

TARGET_SAMPLE_RATE = 16000


def record_wav(
    output_path: str | Path,
    duration: float,
    sample_rate: int = TARGET_SAMPLE_RATE,
) -> Path:
    """
    Record mono audio from the microphone and save it as a WAV file.
    """
    output_path = Path(output_path)

    num_samples = int(duration * sample_rate)
    print(f"Recording for {duration:.1f} seconds... Speak now.")

    audio = sd.rec(
        num_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    audio_int16 = np.int16(audio.squeeze() * 32767)
    write(output_path, sample_rate, audio_int16)

    print(f"Saved recording to: {output_path}")
    return output_path