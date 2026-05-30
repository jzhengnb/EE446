#include "TensorFlowLite.h"
#include "fall_detection_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <ArduinoBLE.h>
#include <math.h>
#include <Arduino_BMI270_BMM150.h>

// ============================================================
// BLE settings
// ============================================================
BLEService fallService("19B10000-E8F2-537E-4F6C-D104768A1214");

// Phone can subscribe to this characteristic.
// It sends strings like:
// NORMAL,prob=0.1234
// FALL_ALERT,prob=0.9876
BLEStringCharacteristic fallStatusCharacteristic(
  "19B10001-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify,
  64
);

// 0 = normal, 1 = fall
BLEByteCharacteristic fallStateCharacteristic(
  "19B10002-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify
);

bool bleReady = false;

// ============================================================
// Model settings
// ============================================================
const int kWindowSize = 151;
const int kNumAxes = 3;
const int kInputSize = kWindowSize * kNumAxes;

// ============================================================
// Normalization constants
// ============================================================
// Replace these with your real zscore_stats.json values if you have them.
const float AX_MEAN = 0.0f;
const float AY_MEAN = 0.0f;
const float AZ_MEAN = 0.0f;

const float AX_STD = 1.0f;
const float AY_STD = 1.0f;
const float AZ_STD = 1.0f;

// ============================================================
// Detection settings
// ============================================================
const float kFallThreshold = 0.5f;
const unsigned long kSampleIntervalMs = 20;

// ============================================================
// TensorFlow Lite Micro globals
// ============================================================
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// If BLE + model causes memory problems, try 70 * 1024 or 90 * 1024.
constexpr int kTensorArenaSize = 80 * 1024;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// ============================================================
// Helper functions
// ============================================================

int8_t quantizeInput(float x) {
  float scale = input->params.scale;
  int zero_point = input->params.zero_point;

  int32_t q = (int32_t)roundf(x / scale + zero_point);

  if (q > 127) q = 127;
  if (q < -128) q = -128;

  return (int8_t)q;
}

float dequantizeOutput(int8_t q) {
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  return scale * ((float)q - zero_point);
}

void printTensorInfo() {
  Serial.println("========== Tensor Info ==========");

  Serial.print("Input dims: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Input type: ");
  Serial.println(input->type);

  Serial.print("Input scale: ");
  Serial.println(input->params.scale, 8);

  Serial.print("Input zero point: ");
  Serial.println(input->params.zero_point);

  Serial.print("Output dims: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Output type: ");
  Serial.println(output->type);

  Serial.print("Output scale: ");
  Serial.println(output->params.scale, 8);

  Serial.print("Output zero point: ");
  Serial.println(output->params.zero_point);

  Serial.println("=================================");
}

void setupBLE() {
  if (!BLE.begin()) {
    Serial.println("ERROR: Failed to initialize BLE.");
    bleReady = false;
    return;
  }

  BLE.setLocalName("Nano33-Fall-Detector");
  BLE.setDeviceName("Nano33-Fall-Detector");
  BLE.setAdvertisedService(fallService);

  fallService.addCharacteristic(fallStatusCharacteristic);
  fallService.addCharacteristic(fallStateCharacteristic);

  BLE.addService(fallService);

  fallStatusCharacteristic.writeValue("BOOTING");
  fallStateCharacteristic.writeValue((byte)0);

  BLE.advertise();

  bleReady = true;

  Serial.println("BLE initialized.");
  Serial.println("Device name: Nano33-Fall-Detector");
  Serial.println("Use nRF Connect or LightBlue on your phone.");
  Serial.println("Subscribe to characteristic:");
  Serial.println("19B10001-E8F2-537E-4F6C-D104768A1214");
}

void sendBLEStatus(bool fallDetected, float probability) {
  if (!bleReady) {
    return;
  }

  String message;

  if (fallDetected) {
    message = "FALL_ALERT,prob=";
    fallStateCharacteristic.writeValue((byte)1);
  } else {
    message = "NORMAL,prob=";
    fallStateCharacteristic.writeValue((byte)0);
  }

  message += String(probability, 4);

  fallStatusCharacteristic.writeValue(message.c_str());

  Serial.print("BLE update: ");
  Serial.println(message);
}

void blinkFallAlert() {
  digitalWrite(LED_BUILTIN, HIGH);

  unsigned long startTime = millis();
  while (millis() - startTime < 1000) {
    BLE.poll();
    delay(10);
  }

  digitalWrite(LED_BUILTIN, LOW);
}

// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);

  // Do not block forever waiting for Serial.
  // This lets the board still run from battery.
  unsigned long serialStart = millis();
  while (!Serial && millis() - serialStart < 3000);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println();
  Serial.println("======================================");
  Serial.println("TinyML Fall Detection with BLE Alert");
  Serial.println("Arduino Nano 33 BLE");
  Serial.println("Model: CNN Full INT8 Quantized");
  Serial.println("Input: 151 x 3 accelerometer window");
  Serial.println("======================================");

  setupBLE();

  if (!IMU.begin()) {
    Serial.println("ERROR: Failed to initialize IMU.");
    sendBLEStatus(true, 1.0f);
    while (1) {
      BLE.poll();
      delay(100);
    }
  }

  Serial.println("IMU initialized.");

  model = tflite::GetModel(fall_detection_dnn_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch.");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Expected version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);

    while (1) {
      BLE.poll();
      delay(100);
    }
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
    model,
    resolver,
    tensor_arena,
    kTensorArenaSize,
    error_reporter
  );

  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();

  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed.");
    Serial.println("Try increasing kTensorArenaSize.");

    while (1) {
      BLE.poll();
      delay(100);
    }
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully.");
  printTensorInfo();

  sendBLEStatus(false, 0.0f);

  Serial.println("Starting real-time fall detection...");
}

// ============================================================
// Main loop
// ============================================================

void loop() {
  BLE.poll();

  int input_index = 0;

  Serial.println();
  Serial.println("Collecting 151-sample window...");

  for (int i = 0; i < kWindowSize; i++) {
    BLE.poll();

    float ax, ay, az;

    while (!IMU.accelerationAvailable()) {
      BLE.poll();
      delay(1);
    }

    IMU.readAcceleration(ax, ay, az);

    // Normalize using training-time assumptions.
    float ax_norm = (ax - AX_MEAN) / AX_STD;
    float ay_norm = (ay - AY_MEAN) / AY_STD;
    float az_norm = (az - AZ_MEAN) / AZ_STD;

    // Input order must match training:
    // [ax0, ay0, az0], [ax1, ay1, az1], ...
    input->data.int8[input_index++] = quantizeInput(ax_norm);
    input->data.int8[input_index++] = quantizeInput(ay_norm);
    input->data.int8[input_index++] = quantizeInput(az_norm);

    delay(kSampleIntervalMs);
  }

  Serial.println("Running inference...");

  TfLiteStatus invoke_status = interpreter->Invoke();

  if (invoke_status != kTfLiteOk) {
    Serial.println("ERROR: Inference failed.");
    return;
  }

  int8_t output_q = output->data.int8[0];
  float fall_probability = dequantizeOutput(output_q);

  Serial.print("Raw output int8: ");
  Serial.println(output_q);

  Serial.print("Fall probability: ");
  Serial.println(fall_probability, 4);

  bool fallDetected = fall_probability >= kFallThreshold;

  if (fallDetected) {
    Serial.println("RESULT: POSSIBLE FALL DETECTED");
    sendBLEStatus(true, fall_probability);
    blinkFallAlert();
  } else {
    Serial.println("RESULT: Normal activity");
    sendBLEStatus(false, fall_probability);
    digitalWrite(LED_BUILTIN, LOW);
  }

  BLE.poll();
  delay(500);
}