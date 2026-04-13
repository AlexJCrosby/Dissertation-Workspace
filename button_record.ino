#include <ESP_I2S.h>

I2SClass I2S;

const int LED_PIN = D2;
const int BUTTON_PIN = D1;

int lastStableButtonState = HIGH;
int lastReading = HIGH;

unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 30;

bool isRecording = false;
unsigned long recordingStartTime = 0;
const unsigned long RECORDING_DURATION_MS = 3000;

const uint32_t SAMPLE_RATE = 16000;
const uint8_t SAMPLE_BITS = 16;

const int MIC_CLK_PIN = 42;
const int MIC_DATA_PIN = 41;

const size_t MAX_SAMPLES = (SAMPLE_RATE * RECORDING_DURATION_MS) / 1000;

int16_t *audioBuffer = nullptr;
size_t sampleCount = 0;
int maxAmplitude = 0;

bool initMicrophone() {
  I2S.setPinsPdmRx(MIC_CLK_PIN, MIC_DATA_PIN);

  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S microphone");
    return false;
  }

  return true;
}

void startRecording() {
  if (audioBuffer == nullptr) {
    Serial.println("Audio buffer not allocated");
    return;
  }

  isRecording = true;
  recordingStartTime = millis();
  sampleCount = 0;
  maxAmplitude = 0;

  digitalWrite(LED_PIN, HIGH);
  Serial.println("RECORDING_START");
}

void stopRecording() {
  isRecording = false;
  digitalWrite(LED_PIN, LOW);

  Serial.println("RECORDING_STOP");
  Serial.print("Captured samples: ");
  Serial.println(sampleCount);

  Serial.print("Peak amplitude: ");
  Serial.println(maxAmplitude);

  if (sampleCount > 0) {
    Serial.println("AUDIO_READY_TO_SEND");
  } else {
    Serial.println("NO_AUDIO_CAPTURED");
  }
}

void captureAudioSample() {
  if (!isRecording) return;
  if (sampleCount >= MAX_SAMPLES) return;

  int sample = I2S.read();

  if (sample == 0 || sample == -1 || sample == 1) {
    return;
  }

  if (sample > 32767) sample = 32767;
  if (sample < -32768) sample = -32768;

  int16_t s = (int16_t)sample;
  audioBuffer[sampleCount++] = s;

  int absValue = abs((int)s);
  if (absValue > maxAmplitude) {
    maxAmplitude = absValue;
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(115200);
  delay(500);

  if (psramFound()) {
    audioBuffer = (int16_t *)ps_malloc(MAX_SAMPLES * sizeof(int16_t));
  } else {
    audioBuffer = (int16_t *)malloc(MAX_SAMPLES * sizeof(int16_t));
  }

  if (audioBuffer == nullptr) {
    Serial.println("Failed to allocate audio buffer");
    while (true) delay(1000);
  }

  if (!initMicrophone()) {
    while (true) delay(1000);
  }

  Serial.println("DEVICE_READY");
}

void loop() {
  int reading = digitalRead(BUTTON_PIN);

  if (reading != lastReading) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != lastStableButtonState) {
      lastStableButtonState = reading;

      if (lastStableButtonState == LOW && !isRecording) {
        startRecording();
      }
    }
  }

  lastReading = reading;

  if (isRecording) {
    captureAudioSample();

    if (millis() - recordingStartTime >= RECORDING_DURATION_MS) {
      stopRecording();
    }
  }
}