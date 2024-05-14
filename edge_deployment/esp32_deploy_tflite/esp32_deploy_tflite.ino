/**
 * Use TensorFlow Lite model on real accelerometer data to detect anomalies
 * 
 * NOTE: You will need to install the TensorFlow Lite library:
 * https://www.tensorflow.org/lite/microcontrollers
 * 
 * Author: Dhruvkumar Vyas
 * Date: May 6, 2020
 * 
 * License: Beerware
 */

// Library includes
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
// #include <Adafruit_MSA301.h>

// Local AI Model
#include "keras_model_deploy.h"

// Import TensorFlow stuff
#include <TensorFlowLite_ESP32.h>
#include "micro_ops.h"
#include "micro_error_reporter.h" 
#include "micro_interpreter.h" 
#include "micro_mutable_op_resolver.h"
#include "version.h" 

// #include <Wire.h>
// #define I2C_SDA 14 // SDA Connected to GPIO 14
// #define I2C_SCL 15 // SCL Connected to GPIO 15

// We need our utils functions for calculating MAD
extern "C" {
#include "utils.h"
};

// Set to 1 to output debug info to Serial, 0 otherwise
#define DEBUG 1

// Pins
constexpr int BUZZER_PIN = 16;

// Settings
constexpr int NUM_AXES = 3;           // Number of axes on accelerometer
constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs
constexpr float THRESHOLD = 1.400e-03;    // Any MSE over this is an anomaly
constexpr int WAIT_TIME = 1000;       // ms between sample sets
constexpr int SAMPLE_RATE = 200;      // How fast to collect measurements (Hz)

// Globals
Adafruit_MPU6050 mpu;

// TFLite globals, used for compatibility with Arduino-style sketches
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  TfLiteTensor* model_output = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by combiling, running, and looking
  // for errors.
  constexpr int kTensorArenaSize = 1 * 2048;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace
 
/*******************************************************************************
 * Main
 */
 
void setup() {

  // Initialize Serial port for debugging
#if DEBUG
  Serial.begin(115200);
  while (!Serial);
#endif

  // Initialize accelerometer
  if (!mpu.begin()) {
#if DEBUG
    Serial.println("Failed to initialize MPU6050!");
#endif
    while (1);
  }

  // MPU6050 Settings
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

  // Configure buzzer pin
  pinMode(BUZZER_PIN, OUTPUT);

  // Set up logging (will report to Serial, even within TFLite functions)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure
  model = tflite::GetModel(keras_model_deploy);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model version does not match Schema");
    while(1);
  }

  // Pull in only needed operations (should match NN layers)
  // Available ops:
  //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
  static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
  micro_mutable_op_resolver.AddBuiltin(
    tflite::BuiltinOperator_FULLY_CONNECTED,
    tflite::ops::micro::Register_FULLY_CONNECTED(),
    1, 3);

  // Build an interpreter to run the model
  static tflite::MicroInterpreter static_interpreter(
    model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
    error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed");
    while(1);
  }

  // Assign model input and output buffers (tensors) to pointers
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);


  
  
}

void loop() {

  float sample[MAX_MEASUREMENTS][NUM_AXES];
  float measurements[MAX_MEASUREMENTS];
  float mad[NUM_AXES];
  float y_val[NUM_AXES];
  float mse;
  TfLiteStatus invoke_status;
  
  // Timestamps for collecting samples
  static unsigned long timestamp = millis();
  static unsigned long prev_timestamp = timestamp;

  // Take a given time worth of measurements
  int i = 0;
  while (i < MAX_MEASUREMENTS) {
    if (millis() >= timestamp + (1000 / SAMPLE_RATE)) {
  
      // Update timestamps to maintain sample rate
      prev_timestamp = timestamp;
      timestamp = millis();

      // Take sample measurement
      // msa.read();

      /* Get new sensor events with the readings */
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);

      // Add readings to array
      sample[i][0] = g.gyro.x;
      sample[i][1] = g.gyro.y;
      sample[i][2] = g.gyro.z;

      // Update sample counter
      i++;
    }
  }
  
  // For each axis, compute the MAD (scale up by 1.4826)
  for (int axis = 0; axis < NUM_AXES; axis++) {
    for (int i = 0; i < MAX_MEASUREMENTS; i++) {
      measurements[i] = sample[i][axis];
    }
    mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
  }

  // Print out MAD calculations
#if DEBUG
  Serial.print("MAD: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.print(mad[axis], 7);
    Serial.print(" ");
  }
  Serial.println();
#endif

  // Copy MAD values to input buffer/tensor
  for (int axis = 0; axis < NUM_AXES; axis++) {
    model_input->data.f[axis] = mad[axis];
  }

  // Run inference
  invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on input");
  }

  // Read predicted y value from output buffer (tensor)
  for (int axis = 0; axis < NUM_AXES; axis++) {
    y_val[axis] = model_output->data.f[axis];
  }

  // Calculate MSE between given and predicted MAD values
  mse = calc_mse(mad, y_val, NUM_AXES);

  // Print out result
#if DEBUG
  Serial.print("Inference result: ");
  for (int axis = 0; axis < NUM_AXES; axis++) {
    Serial.print(y_val[axis], 7);
  }
  Serial.println();
  Serial.print("MSE: ");
  Serial.println(mse, 7);
#endif

  // Compare to threshold
  if (mse > THRESHOLD) {
    digitalWrite(BUZZER_PIN, HIGH);
#if DEBUG
    Serial.println("DANGER!!!");
#endif
  } else {
    digitalWrite(BUZZER_PIN, LOW);
  }
#if DEBUG
  Serial.println();
#endif

  delay(WAIT_TIME);

}






// // V2.
// /**
//  * Use TensorFlow Lite model on real accelerometer data to detect anomalies
//  * 
//  * NOTE: You will need to install the TensorFlow Lite library:
//  * https://www.tensorflow.org/lite/microcontrollers
//  * 
//  * Author: Dhruvkumar Vyas
//  * Date: May 6, 2020
//  * 
//  * License: Beerware
//  */

// // Library includes
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>

// //Telegram
// #include <WiFiClientSecure.h>
// #include <UniversalTelegramBot.h>


// //Time
// #include <WiFi.h>
// #include <time.h>
// // #include <Adafruit_MSA301.h>

// // Local AI Model
// #include "keras_model_deploy.h"

// // Import TensorFlow stuff
// #include <TensorFlowLite_ESP32.h>
// #include "micro_ops.h"
// #include "micro_error_reporter.h" 
// #include "micro_interpreter.h" 
// #include "micro_mutable_op_resolver.h"
// #include "version.h" 

// // Telegram
// #define BOTtoken "7076377067:AAE98GG2-br73-A3hq0tqD1_jQQGqxycLyc"
// #define CHAT_ID "1224139730"

// // #include <Wire.h>
// // #define I2C_SDA 14 // SDA Connected to GPIO 14
// // #define I2C_SCL 15 // SCL Connected to GPIO 15

// // We need our utils functions for calculating MAD
// extern "C" {
// #include "utils.h"
// };

// // Set to 1 to output debug info to Serial, 0 otherwise
// #define DEBUG 1

// // Pins
// constexpr int BUZZER_PIN = 16;

// // Settings
// constexpr int NUM_AXES = 3;           // Number of axes on accelerometer
// constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
// constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs
// constexpr float THRESHOLD = 1.400e-03;    // Any MSE over this is an anomaly
// constexpr int WAIT_TIME = 1000;       // ms between sample sets
// constexpr int SAMPLE_RATE = 200;      // How fast to collect measurements (Hz)


// //Time
// const char* ssid     = "Dhruv Vyas";
// const char* password = "123456789";

// const char* ntpServer = "pool.ntp.org";
// const long  gmtOffset_sec = 19800;
// const int   daylightOffset_sec = 3600;

// char timeStr[50];

// // Globals
// Adafruit_MPU6050 mpu;

// //Telegram
// WiFiClientSecure client;
// UniversalTelegramBot bot(BOTtoken, client);

// // TFLite globals, used for compatibility with Arduino-style sketches
// namespace {
//   tflite::ErrorReporter* error_reporter = nullptr;
//   const tflite::Model* model = nullptr;
//   tflite::MicroInterpreter* interpreter = nullptr;
//   TfLiteTensor* model_input = nullptr;
//   TfLiteTensor* model_output = nullptr;

//   // Create an area of memory to use for input, output, and other TensorFlow
//   // arrays. You'll need to adjust this by combiling, running, and looking
//   // for errors.
//   constexpr int kTensorArenaSize = 1 * 1024;
//   uint8_t tensor_arena[kTensorArenaSize];
// } // namespace
 
// /*******************************************************************************
//  * Main
//  */
 
// void setup() {

// // Time
// // Connect to Wi-Fi
//   Serial.print("Connecting to ");
//   Serial.println(ssid);
//   WiFi.mode(WIFI_STA);
//   WiFi.begin(ssid, password);
//   client.setCACert(TELEGRAM_CERTIFICATE_ROOT);
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print(".");
//   }
//   Serial.println("");
//   Serial.println("WiFi connected.");
  
//   // Init and get the time
//   configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
//   printLocalTime();

//   // //disconnect WiFi as it's no longer needed
//   // WiFi.disconnect(true);
//   // WiFi.mode(WIFI_OFF);



//   // Initialize Serial port for debugging
// #if DEBUG
//   Serial.begin(115200);
//   while (!Serial);
// #endif

//   // Initialize accelerometer
//   if (!mpu.begin()) {
// #if DEBUG
//     Serial.println("Failed to initialize MPU6050!");
// #endif
//     while (1);
//   }

//   // MPU6050 Settings
//   mpu.setGyroRange(MPU6050_RANGE_250_DEG);
//   mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);

//   // Configure buzzer pin
//   pinMode(BUZZER_PIN, OUTPUT);

//   // Set up logging (will report to Serial, even within TFLite functions)
//   static tflite::MicroErrorReporter micro_error_reporter;
//   error_reporter = &micro_error_reporter;

//   // Map the model into a usable data structure
//   model = tflite::GetModel(keras_model_deploy);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     error_reporter->Report("Model version does not match Schema");
//     while(1);
//   }

//   // Pull in only needed operations (should match NN layers)
//   // Available ops:
//   //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
//   static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
//   micro_mutable_op_resolver.AddBuiltin(
//     tflite::BuiltinOperator_FULLY_CONNECTED,
//     tflite::ops::micro::Register_FULLY_CONNECTED(),
//     1, 3);

//   // Build an interpreter to run the model
//   static tflite::MicroInterpreter static_interpreter(
//     model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
//     error_reporter);
//   interpreter = &static_interpreter;

//   // Allocate memory from the tensor_arena for the model's tensors
//   TfLiteStatus allocate_status = interpreter->AllocateTensors();
//   if (allocate_status != kTfLiteOk) {
//     error_reporter->Report("AllocateTensors() failed");
//     while(1);
//   }

//   // Assign model input and output buffers (tensors) to pointers
//   model_input = interpreter->input(0);
//   model_output = interpreter->output(0);


  
  
// }

// void loop() {

//   //Time
//   struct tm timeinfo;
//   if(!getLocalTime(&timeinfo)){
//     Serial.println("Failed to obtain time");
//     return;
//   }
  
//   strftime(timeStr, sizeof(timeStr), "%A, %B %d %Y %H:%M:%S", &timeinfo);
//   Serial.println(timeStr);





//   float sample[MAX_MEASUREMENTS][NUM_AXES];
//   float measurements[MAX_MEASUREMENTS];
//   float mad[NUM_AXES];
//   float y_val[NUM_AXES];
//   float mse;
//   TfLiteStatus invoke_status;
  
//   // Timestamps for collecting samples
//   static unsigned long timestamp = millis();
//   static unsigned long prev_timestamp = timestamp;

//   // Take a given time worth of measurements
//   int i = 0;
//   while (i < MAX_MEASUREMENTS) {
//     if (millis() >= timestamp + (1000 / SAMPLE_RATE)) {
  
//       // Update timestamps to maintain sample rate
//       prev_timestamp = timestamp;
//       timestamp = millis();

//       // Take sample measurement
//       // msa.read();

//       /* Get new sensor events with the readings */
//       sensors_event_t a, g, temp;
//       mpu.getEvent(&a, &g, &temp);

//       // Add readings to array
//       sample[i][0] = g.gyro.x;
//       sample[i][1] = g.gyro.y;
//       sample[i][2] = g.gyro.z;

//       // Update sample counter
//       i++;
//     }
//   }
  
//   // For each axis, compute the MAD (scale up by 1.4826)
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     for (int i = 0; i < MAX_MEASUREMENTS; i++) {
//       measurements[i] = sample[i][axis];
//     }
//     mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
//   }

//   // Print out MAD calculations
// #if DEBUG
//   Serial.print("MAD: ");
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     Serial.print(mad[axis], 7);
//     Serial.print(" ");
//   }
//   Serial.println();
// #endif

//   // Copy MAD values to input buffer/tensor
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     model_input->data.f[axis] = mad[axis];
//   }

//   // Run inference
//   invoke_status = interpreter->Invoke();
//   if (invoke_status != kTfLiteOk) {
//     error_reporter->Report("Invoke failed on input");
//   }

//   // Read predicted y value from output buffer (tensor)
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     y_val[axis] = model_output->data.f[axis];
//   }

//   // Calculate MSE between given and predicted MAD values
//   mse = calc_mse(mad, y_val, NUM_AXES);

//   // Print out result
// #if DEBUG
//   Serial.print("Inference result: ");
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     Serial.print(y_val[axis], 7);
//   }
//   Serial.println();
//   Serial.print("MSE: ");
//   Serial.println(mse, 7);
// #endif

//   // Compare to threshold
//   if (mse > THRESHOLD) {
//     digitalWrite(BUZZER_PIN, HIGH);
//     bot.sendMessage(CHAT_ID, String("\U0001F6A8 DANGER!!! Anomaly Detected! \U000026A0 ") + timeStr);
// #if DEBUG
//     Serial.println(String("\U0001F6A8 DANGER!!! \U000026A0 ") + timeStr);
//     Serial.println("Message sent to Telegram!");
// #endif
//   } else {
//     digitalWrite(BUZZER_PIN, LOW);
//     bot.sendMessage(CHAT_ID, String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
// #if DEBUG
//     Serial.println(String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
//     Serial.println("Message sent to Telegram!");
// #endif
//   }
// #if DEBUG
//   Serial.println();
// #endif

//   delay(WAIT_TIME);

// }


// //Time
// void printLocalTime(){
//   struct tm timeinfo;
//   if(!getLocalTime(&timeinfo)){
//     Serial.println("Failed to obtain time");
//     return;
//   }
  
//   strftime(timeStr, sizeof(timeStr), "%A, %B %d %Y %H:%M:%S", &timeinfo);
//   Serial.println(timeStr);
  
// }







// //V.3
// /**
//  * Use TensorFlow Lite model on real accelerometer data to detect anomalies
//  * 
//  * NOTE: You will need to install the TensorFlow Lite library:
//  * https://www.tensorflow.org/lite/microcontrollers
//  * 
//  * Author: Dhruvkumar Vyas
//  * Date: May 6, 2020
//  * 
//  * License: Beerware
//  */

// // Library includes
// #include <Adafruit_MPU6050.h>
// #include <Adafruit_Sensor.h>


// //Telegram
// #include <WiFiClientSecure.h>
// #include <UniversalTelegramBot.h>


// //Time
// #include <WiFi.h>
// #include <time.h>
// // #include <Adafruit_MSA301.h>

// // Local AI Model
// #include "keras_model_deploy.h"

// // Import TensorFlow stuff
// #include <TensorFlowLite_ESP32.h>
// #include "micro_ops.h"
// #include "micro_error_reporter.h" 
// #include "micro_interpreter.h" 
// #include "micro_mutable_op_resolver.h"
// #include "version.h" 


// // Telegram
// #define BOTtoken "7076377067:AAE98GG2-br73-A3hq0tqD1_jQQGqxycLyc"
// #define CHAT_ID "1224139730"

// // #include <Wire.h>
// // #define I2C_SDA 14 // SDA Connected to GPIO 14
// // #define I2C_SCL 15 // SCL Connected to GPIO 15

// // We need our utils functions for calculating MAD
// extern "C" {
// #include "utils.h"
// };

// // Set to 1 to output debug info to Serial, 0 otherwise
// #define DEBUG 1

// // Pins
// constexpr int BUZZER_PIN = 16;

// // Settings
// constexpr int NUM_AXES = 3;           // Number of axes on accelerometer
// constexpr int MAX_MEASUREMENTS = 128; // Number of samples to keep in each axis
// constexpr float MAD_SCALE = 1.4826;   // Scale MAD to be inline with SciPy calcs
// // constexpr float THRESHOLD = 1.400e-03;    // Any MSE over this is an anomaly
// constexpr int WAIT_TIME = 1000;       // ms between sample sets
// constexpr int SAMPLE_RATE = 200;      // How fast to collect measurements (Hz)

// // MSE Values
// constexpr float NORMAL = 1.500e-03; // NORMAL
// constexpr float WEIGHT = 0.1; // Weight
// // No need to define for 'Ideal'


// //Time
// const char* ssid     = "Dhruv Vyas";
// const char* password = "123456789";

// const char* ntpServer = "pool.ntp.org";
// const long  gmtOffset_sec = 19800;
// const int   daylightOffset_sec = 3600;

// char timeStr[50];

// // Weight to Normal Flag
// bool WEIGHT_TO_NORMAL = 0;

// // Globals
// Adafruit_MPU6050 mpu;

// //Telegram
// WiFiClientSecure client;
// UniversalTelegramBot bot(BOTtoken, client);

// // TFLite globals, used for compatibility with Arduino-style sketches
// namespace {
//   tflite::ErrorReporter* error_reporter = nullptr;
//   const tflite::Model* model = nullptr;
//   tflite::MicroInterpreter* interpreter = nullptr;
//   TfLiteTensor* model_input = nullptr;
//   TfLiteTensor* model_output = nullptr;

//   // Create an area of memory to use for input, output, and other TensorFlow
//   // arrays. You'll need to adjust this by combiling, running, and looking
//   // for errors.
//   constexpr int kTensorArenaSize = 1 * 1024;
//   uint8_t tensor_arena[kTensorArenaSize];
// } // namespace
 
// /*******************************************************************************
//  * Main
//  */
 
// void setup() {

// // Time
// // Connect to Wi-Fi
//   Serial.print("Connecting to ");
//   Serial.println(ssid);
//   WiFi.mode(WIFI_STA);
//   WiFi.begin(ssid, password);
//   client.setCACert(TELEGRAM_CERTIFICATE_ROOT);
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(500);
//     Serial.print(".");
//   }
//   Serial.println("");
//   Serial.println("WiFi connected.");
  
//   // Init and get the time
//   configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);
//   printLocalTime();

//   // //disconnect WiFi as it's no longer needed
//   // WiFi.disconnect(true);
//   // WiFi.mode(WIFI_OFF);



//   // Initialize Serial port for debugging
// #if DEBUG
//   Serial.begin(115200);
//   while (!Serial);
// #endif

    
//   // Initialize accelerometer
//   if (!mpu.begin()) {
// #if DEBUG
//     Serial.println("Failed to initialize MPU6050!");
// #endif
//     while (1);
//   }

//   // MPU6050 Settings
//   mpu.setGyroRange(MPU6050_RANGE_250_DEG);
//   mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  

//   // Configure buzzer pin
//   pinMode(BUZZER_PIN, OUTPUT);

//   // Set up logging (will report to Serial, even within TFLite functions)
//   static tflite::MicroErrorReporter micro_error_reporter;
//   error_reporter = &micro_error_reporter;

//   // Map the model into a usable data structure
//   model = tflite::GetModel(keras_model_deploy);
//   if (model->version() != TFLITE_SCHEMA_VERSION) {
//     error_reporter->Report("Model version does not match Schema");
//     while(1);
//   }

//   // Pull in only needed operations (should match NN layers)
//   // Available ops:
//   //  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/kernels/micro_ops.h
//   static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
//   micro_mutable_op_resolver.AddBuiltin(
//     tflite::BuiltinOperator_FULLY_CONNECTED,
//     tflite::ops::micro::Register_FULLY_CONNECTED(),
//     1, 3);

//   // Build an interpreter to run the model
//   static tflite::MicroInterpreter static_interpreter(
//     model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
//     error_reporter);
//   interpreter = &static_interpreter;

//   // Allocate memory from the tensor_arena for the model's tensors
//   TfLiteStatus allocate_status = interpreter->AllocateTensors();
//   if (allocate_status != kTfLiteOk) {
//     error_reporter->Report("AllocateTensors() failed");
//     while(1);
//   }

//   // Assign model input and output buffers (tensors) to pointers
//   model_input = interpreter->input(0);
//   model_output = interpreter->output(0);


  
  
// }

// void loop() {

//   //Time
//   struct tm timeinfo;
//   if(!getLocalTime(&timeinfo)){
//     Serial.println("Failed to obtain time");
//     return;
//   }
  
//   strftime(timeStr, sizeof(timeStr), "%A, %B %d %Y %H:%M:%S", &timeinfo);
//   Serial.println(timeStr);





//   float sample[MAX_MEASUREMENTS][NUM_AXES];
//   float measurements[MAX_MEASUREMENTS];
//   float mad[NUM_AXES];
//   float y_val[NUM_AXES];
//   float mse;
//   TfLiteStatus invoke_status;
  
//   // Timestamps for collecting samples
//   static unsigned long timestamp = millis();
//   static unsigned long prev_timestamp = timestamp;

//   // Take a given time worth of measurements
//   int i = 0;
//   while (i < MAX_MEASUREMENTS) {
//     if (millis() >= timestamp + (1000 / SAMPLE_RATE)) {
  
//       // Update timestamps to maintain sample rate
//       prev_timestamp = timestamp;
//       timestamp = millis();

//       // Take sample measurement
//       // msa.read();

//       /* Get new sensor events with the readings */
//       sensors_event_t a, g, temp;
//       mpu.getEvent(&a, &g, &temp);

//       // Add readings to array
//       sample[i][0] = g.gyro.x;
//       sample[i][1] = g.gyro.y;
//       sample[i][2] = g.gyro.z;

//       // Update sample counter
//       i++;
//     }
//   }
  
//   // For each axis, compute the MAD (scale up by 1.4826)
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     for (int i = 0; i < MAX_MEASUREMENTS; i++) {
//       measurements[i] = sample[i][axis];
//     }
//     mad[axis] = MAD_SCALE * calc_mad(measurements, MAX_MEASUREMENTS);
//   }

//   // Print out MAD calculations
// #if DEBUG
//   Serial.print("MAD: ");
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     Serial.print(mad[axis], 7);
//     Serial.print(" ");
//   }
//   Serial.println();
// #endif

//   // Copy MAD values to input buffer/tensor
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     model_input->data.f[axis] = mad[axis];
//   }

//   // Run inference
//   invoke_status = interpreter->Invoke();
//   if (invoke_status != kTfLiteOk) {
//     error_reporter->Report("Invoke failed on input");
//   }

//   // Read predicted y value from output buffer (tensor)
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     y_val[axis] = model_output->data.f[axis];
//   }

//   // Calculate MSE between given and predicted MAD values
//   mse = calc_mse(mad, y_val, NUM_AXES);

//   // Print out result
// #if DEBUG
//   Serial.print("Inference result: ");
//   for (int axis = 0; axis < NUM_AXES; axis++) {
//     Serial.print(y_val[axis], 7);
//   }
//   Serial.println();
//   Serial.print("MSE: ");
//   Serial.println(mse, 7);
// #endif

//   // Compare to 'mse'
//   if (mse < NORMAL) {
//     WEIGHT_TO_NORMAL = 0;
//     digitalWrite(BUZZER_PIN, LOW);
//     bot.sendMessage(CHAT_ID, String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
// #if DEBUG
//     Serial.println(String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
//     Serial.println("Message sent to Telegram!");
// #endif
//   } else if (mse > WEIGHT) {
//     WEIGHT_TO_NORMAL = 1;
//     digitalWrite(BUZZER_PIN, HIGH);
//     bot.sendMessage(CHAT_ID, String("\U0001F6A8 DANGER!!! Anomaly Weight Detected! \U000026A0 ") + timeStr);
// #if DEBUG
//     Serial.println(String("\U0001F6A8 DANGER!!! Weight \U000026A0 ") + timeStr);
//     Serial.println("Message sent to Telegram!");
// #endif
//   } else { // Ideal
//       if (WEIGHT_TO_NORMAL){ // Now in 'Normal'
//         WEIGHT_TO_NORMAL = 0;
//         digitalWrite(BUZZER_PIN, LOW);
//         bot.sendMessage(CHAT_ID, String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
// #if DEBUG
//         Serial.println(String("\U00002705 Normal Operation! \U0001F600 ") + timeStr);
//         Serial.println("Message sent to Telegram!");
// #endif
//       }
//       else{
//         WEIGHT_TO_NORMAL = 0;
//         digitalWrite(BUZZER_PIN, HIGH);
//         bot.sendMessage(CHAT_ID, String("\U0001F6A8 DANGER!!! Anomaly Ideal Detected! \U000026A0 ") + timeStr);
// #if DEBUG
//         Serial.println(String("\U0001F6A8 DANGER!!! Ideal \U000026A0 ") + timeStr);
//         Serial.println("Message sent to Telegram!");
// #endif
//       }
//   }

// #if DEBUG
//   Serial.println();
// #endif

//   delay(WAIT_TIME);

// }


// //Time
// void printLocalTime(){
//   struct tm timeinfo;
//   if(!getLocalTime(&timeinfo)){
//     Serial.println("Failed to obtain time");
//     return;
//   }
  
//   strftime(timeStr, sizeof(timeStr), "%A, %B %d %Y %H:%M:%S", &timeinfo);
//   Serial.println(timeStr);
  
// }

