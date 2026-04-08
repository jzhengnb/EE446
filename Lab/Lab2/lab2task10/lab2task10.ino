#include <PDM.h>
#include <Arduino_APDS9960.h>
#include <Arduino_BMI270_BMM150.h>
#include <math.h>

// microphone
short sampleBuffer[256];
volatile int samplesRead = 0;
int level = 0;

void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

// thresholds
const int MIC_THRESHOLD = 700; // sound if level >
const int CLEAR_THRESHOLD = 50; // dark if clear <
const float MOTION_THRESHOLD = 0.05; // moving if motionValue >
const int PROX_THRESHOLD = 120; // near if proximity <

float lastX = 0, lastY = 0, lastZ = 0;
bool firstMove = true;

int clear = 0;
int proximity = 255;

void setup() {
  Serial.begin(115200);
  delay(500);

  // microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start PDM microphone.");
    while (1);
  }

  // IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  // APDS9960
  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }
}

void loop() {
  // read microphone
  if (samplesRead) {
    long sum = 0;
    for (int i = 0; i < samplesRead; i++) {
      sum += abs(sampleBuffer[i]);
    }
    level = sum / samplesRead;
    samplesRead = 0;
  }

  // read motion from IMU accelerometer
  float x, y, z;
  float motionValue = 0;

  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(x, y, z);

    if (!firstMove) {
      float dx = x - lastX;
      float dy = y - lastY;
      float dz = z - lastZ;
      motionValue = sqrt(dx * dx + dy * dy + dz * dz);
    }

    lastX = x;
    lastY = y;
    lastZ = z;
    firstMove = false;
  }

  // read clear channel
  int r, g, b;
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clear);
  }

  // read proximity
  if (APDS.proximityAvailable()) {
    proximity = APDS.readProximity();
  }

  bool sound = (level > MIC_THRESHOLD);
  bool dark = (clear < CLEAR_THRESHOLD);
  bool moving = (motionValue > MOTION_THRESHOLD);
  bool near = (proximity < PROX_THRESHOLD);

  String situation;
  if (!sound && !dark && !moving && !near) {
    situation = "QUIET_BRIGHT_STEADY_FAR";
  }
  else if (sound && !dark && !moving && !near) {
    situation = "NOISY_BRIGHT_STEADY_FAR";
  }
  else if (!sound && dark && !moving && near) {
    situation = "QUIET_DARK_STEADY_NEAR";
  }
  else if (sound && !dark && moving && near) {
    situation = "NOISY_BRIGHT_MOVING_NEAR";
  }

  Serial.print("raw,mic=");
  Serial.print(level);
  Serial.print(",clear=");
  Serial.print(clear);
  Serial.print(",motion=");
  Serial.print(motionValue, 3);
  Serial.print(",prox=");
  Serial.println(proximity);

  Serial.print("flags,sound=");
  Serial.print(sound ? 1 : 0);
  Serial.print(",dark=");
  Serial.print(dark ? 1 : 0);
  Serial.print(",moving=");
  Serial.print(moving ? 1 : 0);
  Serial.print(",near=");
  Serial.println(near ? 1 : 0);

  Serial.print("state,");
  Serial.println(situation);

  delay(1500);
}