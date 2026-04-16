#include <WiFi.h>
#include <ESP_I2S.h>

// =============================
// Wi-Fi config
// =============================
const char *WIFI_SSID = "HomeWiFi_2G";
const char *WIFI_PASS = "tyycc9296j";

// Companion device IP address on the same network.
const char *SERVER_HOST = "192.168.1.140";
const uint16_t SERVER_PORT = 5001;

const char *DEVICE_ID = "xiao-esp32s3";

// =============================
// Hardware config
// =============================
I2SClass I2S;
WiFiClient client;

const int LED_PIN = D2;
const int BUTTON_PIN = D1;

const int MIC_CLK_PIN = 42;
const int MIC_DATA_PIN = 41;

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
const size_t MAX_SAMPLES = (SAMPLE_RATE * RECORDING_DURATION_MS) / 1000;

int16_t *audioBuffer = nullptr;
size_t sampleCount = 0;
uint32_t recordingSequence = 1;


bool initMicrophone() {
  I2S.setPinsPdmRx(MIC_CLK_PIN, MIC_DATA_PIN);

  if (!I2S.begin(I2S_MODE_PDM_RX, SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    Serial.println("Failed to initialize I2S microphone");
    return false;
  }

  return true;
}


void connectToWiFi() {
  Serial.print("Connecting to Wi-Fi");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.println("Wi-Fi connected");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}


void appendWavHeader(uint8_t *header, uint32_t dataSize) {
  const uint32_t byteRate = SAMPLE_RATE * NUM_CHANNELS * (SAMPLE_BITS / 8);
  const uint16_t blockAlign = NUM_CHANNELS * (SAMPLE_BITS / 8);
  const uint32_t chunkSize = 36 + dataSize;
  const uint16_t audioFormat = 1;  // PCM
  const uint32_t subchunk1Size = 16;

  memcpy(header + 0, "RIFF", 4);
  memcpy(header + 4, &chunkSize, 4);
  memcpy(header + 8, "WAVE", 4);

  memcpy(header + 12, "fmt ", 4);
  memcpy(header + 16, &subchunk1Size, 4);
  memcpy(header + 20, &audioFormat, 2);
  memcpy(header + 22, &NUM_CHANNELS, 2);
  memcpy(header + 24, &SAMPLE_RATE, 4);
  memcpy(header + 28, &byteRate, 4);
  memcpy(header + 32, &blockAlign, 2);
  memcpy(header + 34, &SAMPLE_BITS, 2);

  memcpy(header + 36, "data", 4);
  memcpy(header + 40, &dataSize, 4);
}


bool sendAudioOverWiFi() {
  if (sampleCount == 0) {
    Serial.println("NO_AUDIO_TO_SEND");
    return false;
  }

  const uint32_t pcmDataSize = sampleCount * sizeof(int16_t);
  const uint32_t wavSize = 44 + pcmDataSize;

  Serial.print("Connecting to receiver ");
  Serial.print(SERVER_HOST);
  Serial.print(":");
  Serial.println(SERVER_PORT);

  if (!client.connect(SERVER_HOST, SERVER_PORT)) {
    Serial.println("Failed to connect to Wi-Fi receiver");
    return false;
  }

  String header = "{";
  header += "\"type\":\"audio_upload\",";
  header += "\"format\":\"wav\",";
  header += "\"sample_rate\":" + String(SAMPLE_RATE) + ",";
  header += "\"channels\":" + String(NUM_CHANNELS) + ",";
  header += "\"sample_width\":2,";
  header += "\"audio_size\":" + String(wavSize) + ",";
  header += "\"device_id\":\"" + String(DEVICE_ID) + "\",";
  header += "\"sequence\":" + String(recordingSequence);
  header += "}\n";

  client.print(header);

  uint8_t wavHeader[44];
  appendWavHeader(wavHeader, pcmDataSize);
  client.write(wavHeader, sizeof(wavHeader));

  const size_t CHUNK_SAMPLES = 512;
  size_t sentSamples = 0;

  while (sentSamples < sampleCount) {
    size_t remaining = sampleCount - sentSamples;
    size_t chunkSamples = remaining < CHUNK_SAMPLES ? remaining : CHUNK_SAMPLES;
    size_t chunkBytes = chunkSamples * sizeof(int16_t);

    size_t written = client.write((const uint8_t *)(audioBuffer + sentSamples), chunkBytes);
    if (written != chunkBytes) {
      Serial.println("Partial/failed audio write over Wi-Fi");
      client.stop();
      return false;
    }

    sentSamples += chunkSamples;
  }

  client.flush();
  Serial.println("Audio upload sent. Waiting for receiver response...");

  unsigned long waitStart = millis();
  while (!client.available() && (millis() - waitStart) < 5000) {
    delay(10);
  }

  if (client.available()) {
    String response = client.readStringUntil('\n');
    Serial.print("Receiver response: ");
    Serial.println(response);
  } else {
    Serial.println("No acknowledgment received from receiver");
  }

  client.stop();
  recordingSequence++;
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

  bool ok = sendAudioOverWiFi();
  if (ok) {
    Serial.println("AUDIO_TRANSFER_COMPLETE");
  } else {
    Serial.println("AUDIO_TRANSFER_FAILED");
  }
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

  connectToWiFi();
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
