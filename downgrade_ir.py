import onnx

model_path = "/home/ameen/ameen/projects/grad/hazeclue-ai/onnx_models/lstm_workload.onnx"
try:
    model = onnx.load(model_path)
    print(f"Original IR version: {model.ir_version}")
    
    # Downgrade IR version to 9 to be compatible with Flutter onnxruntime 1.4.1
    model.ir_version = 9
    
    onnx.save(model, model_path)
    print(f"Successfully updated IR version to: {model.ir_version}")
except Exception as e:
    print(f"Error: {e}")
