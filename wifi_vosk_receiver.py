from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from assistant_vosk import TaskManager, parse_intent, handle_command
from asr.vosk_asr.vosk_integration import transcribe_wav_file

HOST = "0.0.0.0"
PORT = 5001
OUTPUT_DIR = Path("received_audio")
PROJECT_ROOT = Path(__file__).resolve().parent
TASKS_PATH = PROJECT_ROOT / "tasks.json"


def recv_exact(conn: socket.socket, num_bytes: int) -> bytes:
    """Read exactly num_bytes from a TCP socket or raise ConnectionError."""
    chunks: list[bytes] = []
    remaining = num_bytes

    while remaining > 0:
        chunk = conn.recv(min(4096, remaining))
        if not chunk:
            raise ConnectionError(
                f"Socket closed before expected payload was fully received. Remaining: {remaining} bytes"
            )
        chunks.append(chunk)
        remaining -= len(chunk)

    return b"".join(chunks)



def recv_json_line(conn: socket.socket, max_bytes: int = 4096) -> dict[str, Any]:
    """
    Read a single newline-terminated JSON header.

    Protocol example:
    {"type":"audio_upload","format":"wav","sample_rate":16000,"channels":1,
     "sample_width":2,"audio_size":96044,"device_id":"xiao-esp32s3"}\n
    Followed immediately by exactly audio_size bytes of WAV data.
    """
    data = bytearray()

    while len(data) < max_bytes:
        ch = conn.recv(1)
        if not ch:
            raise ConnectionError("Socket closed while reading JSON header")
        if ch == b"\n":
            break
        data.extend(ch)
    else:
        raise ValueError("JSON header exceeded maximum allowed size")

    try:
        return json.loads(data.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON header: {exc}") from exc



def validate_header(header: dict[str, Any]) -> None:
    required = [
        "type",
        "format",
        "sample_rate",
        "channels",
        "sample_width",
        "audio_size",
        "device_id",
    ]

    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f"Header missing required fields: {missing}")

    if header["type"] != "audio_upload":
        raise ValueError(f"Unsupported type: {header['type']}")

    if header["format"] != "wav":
        raise ValueError(f"Unsupported format: {header['format']}")

    if int(header["sample_rate"]) != 16000:
        raise ValueError(f"Expected 16000 Hz audio, got {header['sample_rate']}")

    if int(header["channels"]) != 1:
        raise ValueError(f"Expected mono audio, got {header['channels']} channel(s)")

    if int(header["sample_width"]) != 2:
        raise ValueError(f"Expected 16-bit PCM audio, got sample width {header['sample_width']}")

    if int(header["audio_size"]) <= 44:
        raise ValueError(f"Audio payload too small to be a WAV file: {header['audio_size']} bytes")



def save_received_wav(wav_bytes: bytes, header: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device_id = str(header.get("device_id", "unknown_device"))
    sequence = int(header.get("sequence", 0))
    filename = f"{device_id}_recording_{sequence:04d}.wav"
    wav_path = OUTPUT_DIR / filename

    wav_path.write_bytes(wav_bytes)
    return wav_path



def process_received_wav(wav_path: Path, task_manager: TaskManager) -> None:
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

    task_manager.load_tasks()
    handle_command(command, task_manager)



def send_status(conn: socket.socket, ok: bool, message: str) -> None:
    payload = {"ok": ok, "message": message}
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))



def handle_client(conn: socket.socket, addr: tuple[str, int], task_manager: TaskManager) -> None:
    print(f"\nConnection from {addr[0]}:{addr[1]}")
    try:
        header = recv_json_line(conn)
        print(f"Header: {header}")
        validate_header(header)

        audio_size = int(header["audio_size"])
        wav_bytes = recv_exact(conn, audio_size)
        wav_path = save_received_wav(wav_bytes, header)

        print(f"Saved WAV to: {wav_path.resolve()}")
        print(f"WAV byte length: {len(wav_bytes)}")

        process_received_wav(wav_path, task_manager)
        send_status(conn, True, "Audio received and processed")

    except Exception as exc:
        print(f"ERROR while handling client: {exc}")
        try:
            send_status(conn, False, str(exc))
        except Exception:
            pass



def main() -> None:
    task_manager = TaskManager(storage_path=TASKS_PATH)

    print("Wi-Fi Vosk Receiver")
    print(f"Listening on {HOST}:{PORT}")
    print(f"Task storage: {TASKS_PATH.resolve()}")
    print("Press Ctrl+C to stop.\n")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen(5)

        try:
            while True:
                conn, addr = server.accept()
                with conn:
                    handle_client(conn, addr, task_manager)
                    print("\nReady for next recording.\n")
        except KeyboardInterrupt:
            print("\nExiting receiver.")


if __name__ == "__main__":
    main()
