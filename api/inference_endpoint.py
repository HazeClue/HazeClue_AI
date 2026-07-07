import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import onnxruntime as ort
import uvicorn

# Ensure the parent directory is in the path to import preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.eeg_processor import EEGProcessor

app = FastAPI(title="HazeClue AI Inference API - LSTM Edition")

# Initialize Preprocessor
preprocessor = EEGProcessor(sfreq=128)

# Load ONNX Model
onnx_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "onnx_models", "lstm_workload.onnx")
if os.path.exists(onnx_path):
    ort_session = ort.InferenceSession(onnx_path)
else:
    ort_session = None
    print(f"WARNING: ONNX model not found at {onnx_path}. Inference will fail until model is trained.")

class EEGWindow(BaseModel):
    # Expecting 14 channels x 256 samples (2 seconds at 128 Hz)
    data: List[List[float]]

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model_loaded": ort_session is not None,
        "model_type": "Bi-LSTM ONNX"
    }

@app.post("/api/inference")
def run_inference(window: EEGWindow):
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train and export the ONNX model first.")
        
    try:
        # data should be shape (14, 256)
        data_np = np.array(window.data, dtype=np.float32)
        
        if data_np.shape[0] != 14:
            raise ValueError(f"Expected 14 channels, got {data_np.shape[0]}")
        if data_np.shape[1] != 256:
            raise ValueError(f"Expected 256 samples (2s @ 128Hz), got {data_np.shape[1]}")
            
        # 1. Preprocess: Bandpass -> CAR -> MAD Clip -> Z-Score
        # Expects shape (256, 14) for filtering algorithms, so transpose
        data_transposed = data_np.transpose() # shape (256, 14)
        processed_data = preprocessor.preprocess_window(data_transposed)
        
        # 2. Prepare for LSTM model: (Batch, TimeSteps, Channels) -> (1, 256, 14)
        input_tensor = np.expand_dims(processed_data, axis=0)
        
        # 3. Inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Output shape is (1, 2) logits
        logits = ort_outs[0][0]
        prediction = int(np.argmax(logits))
        
        # Convert logits to probability via softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        probability = float(probs[prediction])
        
        mode = "WORKLOAD" if prediction == 1 else "REST"
        
        return {
            "prediction": prediction,
            "probability": probability,
            "mode": mode,
            "accepted": True
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
