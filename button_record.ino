const int LED_PIN = D2;
const int BUTTON_PIN = D1;

int lastStableButtonState = HIGH;
int lastReading = HIGH;

unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 30;

// Recording state
bool isRecording = false;
unsigned long recordingStartTime = 0;
const unsigned long RECORDING_DURATION_MS = 3000;  

void startRecording() {
  isRecording = true;
  recordingStartTime = millis();
  digitalWrite(LED_PIN, HIGH);

  Serial.println("RECORDING_START");

}

void stopRecording() {
  isRecording = false;
  digitalWrite(LED_PIN, LOW);

  Serial.println("RECORDING_STOP");

  Serial.println("AUDIO_READY_TO_SEND");
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(500);
  Serial.println("DEVICE_READY");
}

void loop() {
  int reading = digitalRead(BUTTON_PIN);

  // Debounce logic
  if (reading != lastReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != lastStableButtonState) {
      lastStableButtonState = reading;

      // Button pressed
      if (lastStableButtonState == LOW && !isRecording) {
        startRecording();
      }
    }
  }

  lastReading = reading;

  // Stop recording after fixed duration
  if (isRecording && (millis() - recordingStartTime >= RECORDING_DURATION_MS)) {
    stopRecording();
  }
}