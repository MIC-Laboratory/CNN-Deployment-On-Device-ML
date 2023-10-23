import torch
import tensorflow as tf
import onnx
import numpy as np
import os
import binascii
from onnxsim import simplify
from onnx_tf.backend import prepare
from mobilenetv1 import MobileNetV1


def representative_dataset():
    """
    Prepare representive data for quantization activation layer
    """
    data = np.load("representive_data.npy",allow_pickle=True)#("representive_data.npy",allow_pickle=True)#("Representive_data.npy") ??????????????????
    for i in range(100):
        temp_data = data[i]
        temp_data = temp_data.reshape(1,2,96,96)#temp_data = temp_data.reshape(1,globalSize,globalSize,3)
        yield [temp_data.astype(np.float32)]

def convert_to_c_array(bytes) -> str:
    """
    Convert TFlite model to C array
    """
    hexstr = binascii.hexlify(bytes).decode("UTF-8")
    hexstr = hexstr.upper()
    array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
    array = [array[i:i+10] for i in range(0, len(array), 10)]
    return ",\n  ".join([", ".join(e) for e in array])

"""
Prepare model
"""
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = MobileNetV1(2,4,0.15)
model.load_state_dict(torch.load("Weights/MobilenetV1_Best.pt")["state_dict"])
model.to(device)
dummy_input = torch.randn(1,2, 96, 96, device=device)

"""
Convert Pytorch Model to Onnx format
"""
torch.onnx.export(model, dummy_input,
                  "MobilenetV1.onnx", verbose=True,input_names=["input"],output_names=["output"])

onnx_model = onnx.load("MobilenetV1.onnx")

"""
Convert Onnx file to TF model
"""
tf_rep = prepare(onnx_model)
tf_rep.export_graph("TF_MobilenetV1")

"""
Convert TF model to TF-lite file for deployment on microcontroller
"""
converter = tf.lite.TFLiteConverter.from_saved_model("TF_MobilenetV1")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.float32

tflite_model = converter.convert()
tflite_model_size = len(tflite_model) / 1024
print('Quantized model size = %dKBs.' % tflite_model_size)
# Save the model
with open("mobilenetv1.tflite", 'wb') as f:
    f.write(tflite_model)

tflite_binary = open('mobilenetv1.tflite', 'rb').read()
ascii_bytes = convert_to_c_array(tflite_binary)
header_file = "const unsigned char model_tflite[] = {\n  " + ascii_bytes + "\n};\nunsigned int model_tflite_len = " + str(len(tflite_binary)) + ";"
with open("model.h", "w") as f:
    f.write(header_file)