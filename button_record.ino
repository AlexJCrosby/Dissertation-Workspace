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
const uint16_t NUM_CHANNELS = 1;

const int MIC_CLK_PIN = 42;
const int MIC_DATA_PIN = 41;

const size_t MAX_SAMPLES = (SAMPLE_RATE * RECORDING_DURATION_MS) / 1000;

int16_t *audioBuffer = nullptr;
size_t sampleCount = 0;

bool initMicrophone() {
  I2S.setPinsPdmRx(MIC_CLK_PIN, MIC_DATA_PIN);

  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S microphone");
    return false;
  }

  return true;
}

void writeWavHeader(Stream &out, uint32_t dataSize) {
  const uint32_t byteRate = SAMPLE_RATE * NUM_CHANNELS * (SAMPLE_BITS / 8);
  const uint16_t blockAlign = NUM_CHANNELS * (SAMPLE_BITS / 8);
  const uint32_t chunkSize = 36 + dataSize;

  out.write((const uint8_t *)"RIFF", 4);
  out.write((uint8_t *)&chunkSize, 4);
  out.write((const uint8_t *)"WAVE", 4);

  out.write((const uint8_t *)"fmt ", 4);
  uint32_t subchunk1Size = 16;
  uint16_t audioFormat = 1;  // PCM
  out.write((uint8_t *)&subchunk1Size, 4);
  out.write((uint8_t *)&audioFormat, 2);
  out.write((uint8_t *)&NUM_CHANNELS, 2);
  out.write((uint8_t *)&SAMPLE_RATE, 4);
  out.write((uint8_t *)&byteRate, 4);
  out.write((uint8_t *)&blockAlign, 2);
  out.write((uint8_t *)&SAMPLE_BITS, 2);

  out.write((const uint8_t *)"data", 4);
  out.write((uint8_t *)&dataSize, 4);
}

void sendWavOverSerial() {
  if (sampleCount == 0) {
    Serial.println("NO_AUDIO_TO_SEND");
    return;
  }

  const uint32_t dataSize = sampleCount * sizeof(int16_t);

  Serial.println("WAV_BEGIN");
  delay(100);

  writeWavHeader(Serial, dataSize);
  Serial.write((uint8_t *)audioBuffer, dataSize);

  delay(100);
  Serial.println();
  Serial.println("WAV_END");
}

void startRecording() {
  if (audioBuffer == nullptr) {
    Serial.println("Audio buffer not allocated");
    return;
  }

  isRecording = true;
  recordingStartTime = millis();
  sampleCount = 0;

  digitalWrite(LED_PIN, HIGH);
  Serial.println("RECORDING_START");
}

void stopRecording() {
  isRecording = false;
  digitalWrite(LED_PIN, LOW);

  Serial.println("RECORDING_STOP");
  Serial.print("Captured samples: ");
  Serial.println(sampleCount);

  if (sampleCount == 0) {
    Serial.println("NO_AUDIO_CAPTURED");
    return;
  }

  long long sum = 0;
  int16_t minSample = 32767;
  int16_t maxSample = -32768;
  size_t zeroCount = 0;

  for (size_t i = 0; i < sampleCount; i++) {
    int16_t s = audioBuffer[i];
    sum += s;

    if (s == 0) zeroCount++;
    if (s < minSample) minSample = s;
    if (s > maxSample) maxSample = s;
  }

  float mean = (float)sum / sampleCount;

  long long sumAbsFromMean = 0;
  int peakFromMean = 0;

  for (size_t i = 0; i < sampleCount; i++) {
    int centred = (int)audioBuffer[i] - (int)mean;
    int absCentred = abs(centred);

    sumAbsFromMean += absCentred;
    if (absCentred > peakFromMean) {
      peakFromMean = absCentred;
    }
  }

  float avgAbsFromMean = (float)sumAbsFromMean / sampleCount;

  Serial.print("Mean sample value: ");
  Serial.println(mean, 2);

  Serial.print("Zero samples: ");
  Serial.println(zeroCount);

  Serial.print("Min sample: ");
  Serial.println(minSample);

  Serial.print("Max sample: ");
  Serial.println(maxSample);

  Serial.print("Peak deviation from mean: ");
  Serial.println(peakFromMean);

  Serial.print("Average abs deviation from mean: ");
  Serial.println(avgAbsFromMean, 2);

  Serial.println("First 10 samples:");
  for (size_t i = 0; i < 10 && i < sampleCount; i++) {
    Serial.println(audioBuffer[i]);
  }

  Serial.println("AUDIO_READY_TO_SEND");

  // Temporary validation path
  sendWavOverSerial();
}

void captureAudioSample() {
  if (!isRecording) return;
  if (sampleCount >= MAX_SAMPLES) return;

  static int16_t tempBuffer[256];

  size_t bytesRead = I2S.readBytes((char *)tempBuffer, sizeof(tempBuffer));
  size_t samplesRead = bytesRead / sizeof(int16_t);

  for (size_t i = 0; i < samplesRead && sampleCount < MAX_SAMPLES; i++) {
    audioBuffer[sampleCount++] = tempBuffer[i];
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