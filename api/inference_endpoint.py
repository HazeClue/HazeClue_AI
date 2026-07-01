from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import uvicorn
from typing import List

from inference.engine import HazeClueInferenceEngine

app = FastAPI(title="HazeClue AI Inference API")

# Initialize the engine
engine = HazeClueInferenceEngine()
engine.load_models('trained_models')

# Optional: Initialize with identity if calibration is not provided yet
# In a real scenario, you'd have a /calibrate endpoint, but for now we set a default
if not engine.is_calibrated:
    engine.P_ref = np.eye(14)
    engine.is_calibrated = True

class EEGWindow(BaseModel):
    # Expecting 14 channels x 512 samples
    data: List[List[float]]

@app.get("/health")
def health_check():
    stats = engine.get_stats()
    return {"status": "healthy", "engine_stats": stats}

@app.post("/api/inference")
def run_inference(window: EEGWindow):
    try:
        data_np = np.array(window.data, dtype=np.float32)
        
        if data_np.shape[0] != 14:
            raise ValueError(f"Expected 14 channels, got {data_np.shape[0]}")
            
        result = engine.infer(data_np)
        
        return {
            "prediction": result.prediction,
            "probability": result.probability,
            "smoothed_output": result.smoothed_output,
            "mode": result.mode.name,
            "sqi_mean": result.sqi_mean,
            "accepted": result.accepted
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
