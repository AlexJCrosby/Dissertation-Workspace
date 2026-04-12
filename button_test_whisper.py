import serial
from asr.whisper_asr.whisper_integration import transcribe_microphone

ser = serial.Serial("COM6", 115200, timeout=1)

while True:
    line = ser.readline().decode(errors="ignore").strip()
    if line == "BUTTON":
        print("Button pressed - recording")
        transcript, wav_path = transcribe_microphone(duration=3.0)
        print("Transcript:", transcript)