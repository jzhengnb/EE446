#include <Arduino_HS300x.h>
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_APDS9960.h>
#include <math.h>
#include <string.h>

// Thresholds
const float RH_THRESHOLD = 20.0;// humid_jump if rh > baseline by this threshold
const float TEMP_THRESHOLD = 1.0;// temp_rise if temp > baseline by this threshold
const float MAG_THRESHOLD = 50.0;// mag_shift if magnetic magnitude changes by this threshold
const int CLEAR_THRESHOLD = 100;// light change if clear changes by this threshold
const int COLOR_THRESHOLD = 80;// color change if rgb changes by this threshold

const unsigned long COOLDOWN = 3000;// cooldown between same events

float rh = 0.0;
float temp = 0.0;
float magValue = 0.0;
int r = 0, g = 0, b = 0, clear = 0;

// Baseline values
float baseRH = 0.0;
float baseTemp = 0.0;
float baseMag = 0.0;
int baseR = 0;
int baseG = 0;
int baseB = 0;
int baseClear = 0;

bool baselineSet = false;

String lastEvent = "BASELINE_NORMAL";
unsigned long lastEventTime = 0;

void setup() {
  Serial.begin(115200);
  delay(1500);

  if (!HS300x.begin()) {
    Serial.println("Failed to initialize humidity/temperature sensor.");
    while (1);
  }

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU.");
    while (1);
  }

  if (!APDS.begin()) {
    Serial.println("Failed to initialize APDS9960 sensor.");
    while (1);
  }
}

void loop() {
  // Read humidity and temperature
  rh = HS300x.readHumidity();
  temp = HS300x.readTemperature();

  // Read magnetometer
  if (IMU.magneticFieldAvailable()) {
    float x, y, z;
    IMU.readMagneticField(x, y, z);
    magValue = sqrt(x * x + y * y + z * z);
  }

  // Read color
  if (APDS.colorAvailable()) {
    APDS.readColor(r, g, b, clear);
  }

  // Set baseline
  if (!baselineSet) {
    baseRH = rh;
    baseTemp = temp;
    baseMag = magValue;
    baseR = r;
    baseG = g;
    baseB = b;
    baseClear = clear;
    baselineSet = true;
  }

  bool humid_jump = ((rh - baseRH) > RH_THRESHOLD);
  bool temp_rise = ((temp - baseTemp) > TEMP_THRESHOLD);
  bool mag_shift = (fabs(magValue - baseMag) > MAG_THRESHOLD);

  bool light_or_color_change =
    (abs(clear - (int)baseClear) > CLEAR_THRESHOLD) || (abs(r - (int)baseR) > COLOR_THRESHOLD) || 
    (abs(g - (int)baseG) > COLOR_THRESHOLD) || (abs(b - (int)baseB) > COLOR_THRESHOLD);

  String FINAL_LABEL = "BASELINE_NORMAL";

  if (humid_jump || temp_rise) {
    FINAL_LABEL = "BREATH_OR_WARM_AIR_EVENT";
  }
  else if (mag_shift) {
    FINAL_LABEL = "MAGNETIC_DISTURBANCE_EVENT";
  }
  else if (light_or_color_change) {
    FINAL_LABEL = "LIGHT_OR_COLOR_CHANGE_EVENT";
  } else {
    FINAL_LABEL = "BASELINE_NORMAL";
  }

  // Cooldown for same event
  unsigned long now = millis();

  if (FINAL_LABEL != "BASELINE_NORMAL") {
    if (FINAL_LABEL == lastEvent && (now - lastEventTime < COOLDOWN)) {
      FINAL_LABEL = "BASELINE_NORMAL";
    } else {
      lastEvent = FINAL_LABEL;
      lastEventTime = now;
    }
  }

  Serial.print("raw,rh=");
  Serial.print(rh, 2);
  Serial.print(",temp=");
  Serial.print(temp, 2);
  Serial.print(",mag=");
  Serial.print(magValue, 2);
  Serial.print(",r=");
  Serial.print(r);
  Serial.print(",g=");
  Serial.print(g);
  Serial.print(",b=");
  Serial.print(b);
  Serial.print(",clear=");
  Serial.println(clear);

  Serial.print("flags,humid_jump=");
  Serial.print(humid_jump ? 1 : 0);
  Serial.print(",temp_rise=");
  Serial.print(temp_rise ? 1 : 0);
  Serial.print(",mag_shift=");
  Serial.print(mag_shift ? 1 : 0);
  Serial.print(",light_or_color_change=");
  Serial.println(light_or_color_change ? 1 : 0);

  Serial.print("event,");
  Serial.println(FINAL_LABEL);

  delay(1500);
}