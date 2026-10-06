#include <SoftwareSerial.h>

const int PIN_THUMB  = A0;
const int PIN_INDEX  = A1;
const int PIN_MIDDLE = A2;
const int PIN_RING   = A3;
const int PIN_PINKY  = A4;

const int BT_RX = 2;
const int BT_TX = 3;

SoftwareSerial BTSerial(BT_RX, BT_TX);

const unsigned long SEND_INTERVAL_MS = 50;
unsigned long lastSend = 0;

const int SMOOTH_N = 4;
int bufThumb[SMOOTH_N], bufIndex[SMOOTH_N],
    bufMiddle[SMOOTH_N], bufRing[SMOOTH_N],
    bufPinky[SMOOTH_N];

int bufPos = 0;
bool bufFilled = false;

int smooth(int *buf, int newVal) {
  buf[bufPos] = newVal;
  int n = bufFilled ? SMOOTH_N : (bufPos + 1);

  long sum = 0;
  for (int i = 0; i < n; i++)
    sum += buf[i];

  return (int)(sum / n);
}

void setup() {
  Serial.begin(9600);
  BTSerial.begin(9600);

  pinMode(PIN_THUMB, INPUT);
  pinMode(PIN_INDEX, INPUT);
  pinMode(PIN_MIDDLE, INPUT);
  pinMode(PIN_RING, INPUT);
  pinMode(PIN_PINKY, INPUT);

  Serial.println("Glove sender ready.");
}

void loop() {
  unsigned long now = millis();

  if (now - lastSend >= SEND_INTERVAL_MS) {
    lastSend = now;

    int rawThumb  = analogRead(PIN_THUMB);
    int rawIndex  = analogRead(PIN_INDEX);
    int rawMiddle = analogRead(PIN_MIDDLE);
    int rawRing   = analogRead(PIN_RING);
    int rawPinky  = analogRead(PIN_PINKY);

    int thumb  = smooth(bufThumb, rawThumb);
    int index_ = smooth(bufIndex, rawIndex);
    int middle = smooth(bufMiddle, rawMiddle);
    int ring   = smooth(bufRing, rawRing);
    int pinky  = smooth(bufPinky, rawPinky);

    bufPos = (bufPos + 1) % SMOOTH_N;

    if (bufPos == 0)
      bufFilled = true;

    String line = String(thumb) + "," +
                  String(index_) + "," +
                  String(middle) + "," +
                  String(ring) + "," +
                  String(pinky);

    BTSerial.println(line);
    Serial.println(line);
  }
}
