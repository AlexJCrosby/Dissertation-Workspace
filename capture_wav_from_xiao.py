from __future__ import annotations

import serial
from pathlib import Path

PORT = "COM6"          # change if needed
BAUD = 115200
OUTPUT_PATH = Path("recorded_from_xiao.wav")

START_MARKER = b"WAV_BEGIN\n"
END_MARKER = b"\nWAV_END"


def main() -> None:
    print(f"Opening {PORT} at {BAUD} baud...")

    with serial.Serial(PORT, BAUD, timeout=1) as ser:
        print("Waiting for WAV_BEGIN...")

        # 1. Wait for start marker line-by-line
        while True:
            line = ser.readline()
            if not line:
                continue

            try:
                decoded = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                decoded = ""

            if decoded:
                print(f"[SERIAL] {decoded}")

            if line == START_MARKER or decoded == "WAV_BEGIN":
                break

        print("Start marker detected. Capturing WAV bytes...")

        # 2. Capture raw bytes until end marker is found
        data = bytearray()

        while True:
            chunk = ser.read(1024)
            if not chunk:
                continue

            data.extend(chunk)

            end_index = data.find(END_MARKER)
            if end_index != -1:
                wav_bytes = data[:end_index]
                OUTPUT_PATH.write_bytes(wav_bytes)
                print(f"Saved WAV to: {OUTPUT_PATH.resolve()}")
                print(f"WAV byte length: {len(wav_bytes)}")
                break


if __name__ == "__main__":
    main()