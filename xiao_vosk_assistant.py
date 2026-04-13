from __future__ import annotations

from pathlib import Path
import sys
import time

import serial
from serial import SerialException

from asr.vosk_asr.vosk_integration import transcribe_wav_file
from assistant_vosk import TaskManager, parse_intent, handle_command

PORT = "COM6"
BAUD = 115200
OUTPUT_WAV = Path("recorded_from_xiao.wav")

START_MARKER = b"WAV_BEGIN\n"
END_MARKER = b"\nWAV_END"


def wait_for_start_marker(ser: serial.Serial) -> None:
    print("Waiting for WAV_BEGIN...")

    while True:
        line = ser.readline()
        if not line:
            continue

        decoded = line.decode("utf-8", errors="ignore").strip()
        if decoded:
            print(f"[SERIAL] {decoded}")

        if line == START_MARKER or decoded == "WAV_BEGIN":
            print("Start marker detected.")
            return


def capture_wav_bytes(ser: serial.Serial) -> bytes:
    print("Capturing WAV bytes...")
    data = bytearray()

    while True:
        chunk = ser.read(1024)
        if not chunk:
            continue

        data.extend(chunk)

        end_index = data.find(END_MARKER)
        if end_index != -1:
            wav_bytes = bytes(data[:end_index])
            return wav_bytes


def save_wav_file(wav_bytes: bytes, output_path: Path) -> Path:
    output_path.write_bytes(wav_bytes)
    print(f"Saved WAV to: {output_path.resolve()}")
    print(f"WAV byte length: {len(wav_bytes)}")
    return output_path


def process_one_recording(
    ser: serial.Serial,
    task_manager: TaskManager,
    output_path: Path,
) -> None:
    wait_for_start_marker(ser)
    wav_bytes = capture_wav_bytes(ser)
    wav_path = save_wav_file(wav_bytes, output_path)

    print("Running Vosk transcription...")
    transcript = transcribe_wav_file(wav_path)
    print(f"Transcript: {transcript}")

    if not transcript.strip():
        print("No speech detected.")
        return

    command = parse_intent(transcript)
    print(f"Parsed intent: {command.intent.value}")
    if command.task_text:
        print(f"Task text: {command.task_text}")

    handle_command(command, task_manager)


def main() -> None:
    task_manager = TaskManager(storage_path="tasks.json")

    print("XIAO Vosk Assistant")
    print(f"Task storage: {task_manager.storage_path.resolve()}")
    print(f"Opening {PORT} at {BAUD} baud...")

    try:
        with serial.Serial(PORT, BAUD, timeout=1) as ser:
            # Give the serial connection a moment to settle
            time.sleep(2)

            print("System ready. Press the wearable button to record.")
            print("Press Ctrl+C to stop.\n")

            while True:
                try:
                    process_one_recording(ser, task_manager, OUTPUT_WAV)
                    print("\nReady for next recording.\n")
                except Exception as exc:
                    print(f"\nERROR while processing recording: {exc}\n")
                    print("Waiting for next recording...\n")

    except SerialException as exc:
        print(f"Could not open serial port {PORT}: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nExiting assistant.")


if __name__ == "__main__":
    main()