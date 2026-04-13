import serial

from asr.vosk_asr.vosk_integration import transcribe_microphone
from assistant import parse_intent, handle_command, TaskManager

# Initialise task system
task_manager = TaskManager()

# Serial connection
ser = serial.Serial("COM6", 115200, timeout=1)

print("System ready. Waiting for button press...")

while True:
    line = ser.readline().decode(errors="ignore").strip()

    if line == "BUTTON":
        print("\nButton pressed - recording")

        try:
            transcript, wav_path = transcribe_microphone(duration=3.0)

            print("Transcript:", transcript)

            # Skip empty or failed recognition
            if not transcript:
                print("No speech detected.")
                continue

            # Parse intent
            command = parse_intent(transcript)

            # Execute command
            response = handle_command(command, task_manager)

            print("Response:", response)

        except Exception as e:
            print("ERROR:", e)