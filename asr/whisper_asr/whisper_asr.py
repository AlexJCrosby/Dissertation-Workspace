from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# -------------------------------------------------------------------
# Windows CUDA DLL path setup for faster-whisper
# -------------------------------------------------------------------

def _add_windows_cuda_paths() -> None:
    """
    Add NVIDIA CUDA/cuDNN DLL folders from the current virtual environment
    to PATH so faster-whisper / CTranslate2 can find them on Windows.
    """
    project_root = Path(__file__).resolve().parents[2]
    venv_site_packages = project_root / ".venv" / "Lib" / "site-packages"

    candidate_paths = [
        venv_site_packages / "nvidia" / "cublas" / "bin",
        venv_site_packages / "nvidia" / "cudnn" / "bin",
    ]

    existing_paths = [str(path) for path in candidate_paths if path.exists()]

    if not existing_paths:
        return

    current_path = os.environ.get("PATH", "")
    for dll_path in reversed(existing_paths):
        if dll_path not in current_path:
            current_path = dll_path + os.pathsep + current_path

    os.environ["PATH"] = current_path

    # Python 3.8+ on Windows supports explicit DLL directories
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        for dll_path in existing_paths:
            try:
                os.add_dll_directory(dll_path)
            except OSError:
                pass


_add_windows_cuda_paths()

from faster_whisper import WhisperModel  # noqa: E402


class WhisperASR:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Optional[WhisperModel] = None

    def load_model(self) -> None:
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe_file(self, audio_path: str | Path) -> str:
        self.load_model()

        if self.model is None:
            raise RuntimeError("Whisper model failed to load.")

        segments, info = self.model.transcribe(str(audio_path))
        text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
        return " ".join(text_parts)

    def transcribe_file_verbose(self, audio_path: str | Path) -> dict:
        self.load_model()

        if self.model is None:
            raise RuntimeError("Whisper model failed to load.")

        segments, info = self.model.transcribe(str(audio_path))
        segment_list = []
        text_parts = []

        for segment in segments:
            seg_text = segment.text.strip()
            segment_list.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": seg_text,
                }
            )
            if seg_text:
                text_parts.append(seg_text)

        return {
            "text": " ".join(text_parts),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segment_list,
        }


def main() -> None:
    test_audio = Path("temp_gui_recording.wav")

    if not test_audio.exists():
        raise FileNotFoundError(f"Test audio not found: {test_audio}")

    asr = WhisperASR(model_size="base", device="cuda", compute_type="float16")
    result = asr.transcribe_file_verbose(test_audio)

    print("Language:", result["language"])
    print("Probability:", result["language_probability"])
    print("Transcript:", result["text"])


if __name__ == "__main__":
    main()