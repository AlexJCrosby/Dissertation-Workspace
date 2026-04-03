from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vosk import Model, KaldiRecognizer


class VoskASR:
    def __init__(
        self,
        model_path: str | Path,
        sample_rate: int = 16000,
    ) -> None:
        self.model_path = Path(model_path)
        self.sample_rate = sample_rate
        self.model: Optional[Model] = None

    def load_model(self) -> None:
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Vosk model not found: {self.model_path}")
            self.model = Model(str(self.model_path))

    def transcribe_wav_bytes(self, wav_bytes: bytes) -> str:
        self.load_model()

        if self.model is None:
            raise RuntimeError("Vosk model failed to load.")

        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        chunk_size = 4000
        accepted_text_parts: list[str] = []
        last_partial_text = ""

        for i in range(0, len(wav_bytes), chunk_size):
            chunk = wav_bytes[i:i + chunk_size]

            if recognizer.AcceptWaveform(chunk):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    accepted_text_parts.append(text)
            else:
                partial = json.loads(recognizer.PartialResult())
                last_partial_text = partial.get("partial", "").strip()

        final_result = json.loads(recognizer.FinalResult())
        final_text = final_result.get("text", "").strip()

        if final_text:
            accepted_text_parts.append(final_text)

        if accepted_text_parts:
            return " ".join(accepted_text_parts).strip()

        return last_partial_text