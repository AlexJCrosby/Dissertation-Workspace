const int LED_PIN = D2;
const int BUTTON_PIN = D1;

int lastButtonState = HIGH;

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  Serial.begin(115200);
}

void loop() {
  int currentButtonState = digitalRead(BUTTON_PIN);

  if (lastButtonState == HIGH && currentButtonState == LOW) {
    digitalWrite(LED_PIN, HIGH);
    Serial.println("BUTTON");
  }

  if (currentButtonState == HIGH) {
    digitalWrite(LED_PIN, LOW);
  }

  lastButtonState = currentButtonState;
  delay(20);
}