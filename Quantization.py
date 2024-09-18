from onnxruntime.quantization import quantize_dynamic, QuantType

model_input = "C:\\Users\\depan\\Desktop\\Study\\Python\\modell.onnx"

model_output = "C:\\Users\\depan\\Desktop\\Study\\Python\\modell_quantized.onnx"

# Apply full quantization to reduce model size aggressively
quantize_dynamic(model_input, model_output, weight_type=QuantType.QInt8, per_channel=True)
