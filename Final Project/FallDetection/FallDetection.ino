#include "TensorFlowLite.h"
#include "fall_detection_model.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <math.h>

// ============================================================
// IMU library selection
// ============================================================
// For Arduino Nano 33 BLE Sense Rev2, use 1.
// For original Nano 33 BLE / Nano 33 BLE Sense, use 0.
#define USE_REV2_IMU 1

#if USE_REV2_IMU
  #include <Arduino_BMI270_BMM150.h>
#else
  #include <Arduino_LSM9DS1.h>
#endif

// ============================================================
// Model settings
// ============================================================
const int kWindowSize = 151;
const int kNumAxes = 3;
const int kInputSize = kWindowSize * kNumAxes;

// UniMiB-SHAR uses accelerometer windows.
// Input shape should be: [1, 151, 3]

// ============================================================
// Normalization constants
// ============================================================
// IMPORTANT:
// Replace these with values from your zscore_stats.json if available.
// For first compile/upload test, these placeholder values are okay.
// But for meaningful real Arduino predictions, use the same normalization
// used during training.

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

// UniMiB-SHAR is commonly treated around 50 Hz.
// 20 ms/sample gives about 3 seconds per 151-sample window.
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

// Increase this if AllocateTensors() fails.
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

// ============================================================
// Setup
// ============================================================

void setup() {
  Serial.begin(115200);
  while (!Serial);

  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println();
  Serial.println("======================================");
  Serial.println("TinyML Fall Detection Deployment");
  Serial.println("Arduino Nano 33 BLE");
  Serial.println("Model: CNN Full INT8 Quantized");
  Serial.println("Input: 151 x 3 accelerometer window");
  Serial.println("Output: fall probability");
  Serial.println("======================================");

  if (!IMU.begin()) {
    Serial.println("ERROR: Failed to initialize IMU.");
    while (1);
  }

  Serial.println("IMU initialized.");

  model = tflite::GetModel(fall_detection_cnn_int8_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch.");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Expected version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (1);
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
    while (1);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model loaded successfully.");
  printTensorInfo();

  Serial.println("Starting real-time fall detection...");
}

// ============================================================
// Main loop
// ============================================================

void loop() {
  int input_index = 0;

  Serial.println();
  Serial.println("Collecting 151-sample window...");

  for (int i = 0; i < kWindowSize; i++) {
    float ax, ay, az;

    while (!IMU.accelerationAvailable()) {
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

  if (fall_probability >= kFallThreshold) {
    Serial.println("RESULT: POSSIBLE FALL DETECTED");
    digitalWrite(LED_BUILTIN, HIGH);
    delay(1000);
    digitalWrite(LED_BUILTIN, LOW);
  } else {
    Serial.println("RESULT: Normal activity");
    digitalWrite(LED_BUILTIN, LOW);
  }

  delay(500);
}
