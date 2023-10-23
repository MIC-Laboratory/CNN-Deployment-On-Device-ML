#include <JPEGENC.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <SDHCI.h>
#include <stdio.h>  /* for sprintf */
#include "model.h"  /* quantized model */
#include <Camera.h>

#define BAUDRATE                (1152000)
#define TOTAL_PICTURE_COUNT     (0)
JPEG jpg;
int take_picture_count = 0;
unsigned long lTime;
/**
   Print error message
*/
// tensorflow intialization
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 30000*10;
uint8_t tensor_arena[kTensorArenaSize];



void printError(enum CamErr err)
{
  Serial.print("Error: ");
  switch (err)
  {
    case CAM_ERR_NO_DEVICE:
      Serial.println("No Device");
      break;
    case CAM_ERR_ILLEGAL_DEVERR:
      Serial.println("Illegal device error");
      break;
    case CAM_ERR_ALREADY_INITIALIZED:
      Serial.println("Already initialized");
      break;
    case CAM_ERR_NOT_INITIALIZED:
      Serial.println("Not initialized");
      break;
    case CAM_ERR_NOT_STILL_INITIALIZED:
      Serial.println("Still picture not initialized");
      break;
    case CAM_ERR_CANT_CREATE_THREAD:
      Serial.println("Failed to create thread");
      break;
    case CAM_ERR_INVALID_PARAM:
      Serial.println("Invalid parameter");
      break;
    case CAM_ERR_NO_MEMORY:
      Serial.println("No memory");
      break;
    case CAM_ERR_USR_INUSED:
      Serial.println("Buffer already in use");
      break;
    case CAM_ERR_NOT_PERMITTED:
      Serial.println("Operation not permitted");
      break;
    default:
      break;
  }
}

/**
   Callback from Camera library when video frame is captured.
*/

void CamCB(CamImage img)
{
  CamErr err;


  /* Check the img instance is available or not. */

  if (img.isAvailable())
  {


    err = img.convertPixFormat(CAM_IMAGE_PIX_FMT_RGB565);
    if (err != CAM_ERR_SUCCESS)
    {
      printError(err);
    }
    const int iWidth = 96, iHeight = 96;
    int i, iMCUCount, rc, iDataSize, iSize;
    JPEGENCODE jpe;
    uint8_t *pBuffer;

    //
    uint8_t* img_buffer = img.getImgBuff();

    // tensorflow inference code



//
//
//


    for (int i = 0; i < 96 * 96 * 2; ++i) {
      input->data.f[i] = (float)(img_buffer[i]);
    }
//
//    Serial.println("Do inference");
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Invoke failed");
      return;
    }
    int maxIndex = 0; // Variable to store the index of the highest value
    float maxValue = output->data.f[0]; // Assume the first value is the highest
//
    for (int n = 0; n < 4; ++n) {
      float value = output->data.f[n];


      if (value > maxValue) {
        maxValue = value;
        maxIndex = n;
      }
      
    }
    Serial.println("output:" + String(maxIndex));

//








    iSize = 65536; // test with an output buffer
    pBuffer = (uint8_t *)malloc(iSize);
    rc = jpg.open(pBuffer, iSize);
    rc = jpg.encodeBegin(&jpe, iWidth, iHeight, JPEG_PIXEL_RGB565, JPEG_SUBSAMPLE_420, JPEG_Q_LOW);
    if (rc == JPEG_SUCCESS) {

      iMCUCount = ((iWidth + jpe.cx - 1) / jpe.cx) * ((iHeight + jpe.cy - 1) / jpe.cy);
      for (i = 0; i < iMCUCount && rc == JPEG_SUCCESS; i++) {
        // Send the same data for all MCUs (a simple diagonal line)
        rc = jpg.addMCU(&jpe, &img_buffer[jpe.x * 2 + jpe.y * iWidth * 2], iWidth * 2);
      }
      iDataSize = jpg.close();
      Serial.println("SIZE:" + String(iDataSize));
      Serial.println("START");
    }
    Serial.write(pBuffer, iDataSize);

    free(pBuffer);
    pBuffer = NULL;

  }
  else
  {
    Serial.println("Failed to get video stream image");
  }
}

/**
   @brief Initialize camera
*/
void setup()
{
  CamErr err;
  Serial.begin(BAUDRATE);
  while (!Serial)
  {
    ; /* wait for serial port to connect. Needed for native USB port only */
  }
  tflite::InitializeTarget();
  memset(tensor_arena, 0, kTensorArenaSize * sizeof(uint8_t));

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure..
  model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version "
                   + String(model->version()) + " not equal "
                   + "to supported version "
                   + String(TFLITE_SCHEMA_VERSION));
    return;
  } else {
    Serial.println("Model version: " + String(model->version()));
  }

  // This pulls in all the operation implementations we need.
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  } else {
    Serial.println("AllocateTensor() Success");
  }

  size_t used_size = interpreter->arena_used_bytes();
  Serial.println("Area used bytes: " + String(used_size));
  input = interpreter->input(0);
  output = interpreter->output(0);


























  
  Serial.println("Prepare camera");
  err = theCamera.begin(1, CAM_VIDEO_FPS_30, 96, 96, CAM_IMAGE_PIX_FMT_YUV422, 7);
  //    err = theCamera.begin(1, CAM_VIDEO_FPS_60, CAM_IMGSIZE_QQVGA_H, CAM_IMGSIZE_QQVGA_V, CAM_IMAGE_PIX_FMT_YUV422, 7);
  if (err != CAM_ERR_SUCCESS)
  {
    printError(err);
  }

  Serial.println("Start streaming");
  err = theCamera.startStreaming(true, CamCB);
  if (err != CAM_ERR_SUCCESS)
  {
    printError(err);
  }

  /* Auto white balance configuration */

  Serial.println("Set Auto white balance parameter");
  err = theCamera.setAutoWhiteBalanceMode(CAM_WHITE_BALANCE_DAYLIGHT);
  if (err != CAM_ERR_SUCCESS)
  {
    printError(err);
  }

  /* Set parameters about still picture.
     In the following case, QUADVGA and JPEG.
  */

  Serial.println("Set still picture format");
  err = theCamera.setStillPictureImageFormat(
          96,
          96,
          CAM_IMAGE_PIX_FMT_JPG);
  if (err != CAM_ERR_SUCCESS)
  {
    printError(err);
  }
}

/**
   @brief Take picture with format JPEG per second
*/

void loop()
{

}
