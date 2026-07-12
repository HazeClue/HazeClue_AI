import onnxruntime as ort
import numpy as np

def test_onnx_model():
    model_path = "onnx_models/lstm_workload.onnx"
    print(f"Loading ONNX model from {model_path}...")
    
    # Initialize the ONNX Runtime session
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get the input name and shape
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Model Input Name: {input_name}")
    print(f"Expected Input Shape: {input_shape}")
    
    # Create dummy data matching the expected input shape (Batch=1, TimeSteps=256, Channels=14)
    # The shape from the script was (1, 256, 14) and batch_size is dynamic
    dummy_input = np.random.randn(1, 256, 14).astype(np.float32)
    print(f"Generated Dummy Input Shape: {dummy_input.shape}")

    # Run inference
    print("Running inference...")
    outputs = session.run(None, {input_name: dummy_input})
    
    # Get the output
    predictions = outputs[0]
    print(f"Inference Successful!")
    print(f"Output Shape: {predictions.shape}")
    print(f"Output Values (Logits): {predictions}")
    
    # Get the predicted class (assuming it's a classification model)
    predicted_class = np.argmax(predictions, axis=1)[0]
    print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    test_onnx_model()
