import onnxruntime
from onnxruntime.quantization import quantize_dynamic, QuantType
import shutil
import os

# Load the original model
original_model_path = "model.onnx"
quantized_model_path = "quantized_model.onnx"

# Quantize the model dynamically
quantize_dynamic(
    model_input=original_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # You can also use QuantType.QUInt8 if needed
)

print("Model quantization complete.")

# Move the quantized model to the desktop
desktop_path = os.path.expanduser("~/Desktop/")
shutil.move(quantized_model_path, os.path.join(desktop_path, quantized_model_path))

print("Quantized model moved to Desktop.")
