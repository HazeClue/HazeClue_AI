"""
ONNX Model Export
==================
Exports trained RARD and MVES classifiers to ONNX format (.onnx)
for edge deployment on Flutter via onnxruntime_flutter.

Target latency: < 35 ms on mobile CPU.
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARN] skl2onnx not installed. ONNX export unavailable.")


def export_classifier_to_onnx(
    classifier,
    scaler,
    n_features: int,
    output_path: str,
    model_name: str = "hazeclue_classifier"
) -> bool:
    """
    Export a scikit-learn classifier + scaler to ONNX.
    
    The exported model includes the scaler as part of the pipeline
    so the ONNX model accepts raw (unscaled) features.
    
    Args:
        classifier: Trained sklearn classifier
        scaler: Fitted StandardScaler
        n_features: Number of input features
        output_path: Path to save .onnx file
        model_name: Name identifier for the model
    
    Returns:
        bool: Success status
    """
    if not ONNX_AVAILABLE:
        print("[ERROR] skl2onnx required for ONNX export")
        return False
    
    from sklearn.pipeline import Pipeline
    
    # Build pipeline: scaler → classifier
    if scaler is not None:
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', classifier)
        ])
    else:
        pipeline = Pipeline([
            ('classifier', classifier)
        ])
    
    # Define input type
    initial_type = [('input', FloatTensorType([None, n_features]))]
    
    try:
        # Convert to ONNX
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=12,
            options={id(classifier): {'zipmap': False}}  # Return probabilities directly
        )
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"[ONNX] Exported {model_name} → {output_file}")
        print(f"       Input: ({n_features},) float32")
        print(f"       Size: {output_file.stat().st_size / 1024:.1f} KB")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ONNX export failed: {e}")
        return False


def verify_onnx_model(
    onnx_path: str,
    test_input: np.ndarray,
    expected_output: np.ndarray = None
) -> bool:
    """
    Verify ONNX model produces valid output.
    
    Args:
        onnx_path: Path to .onnx file
        test_input: Shape (1, n_features)
        expected_output: Optional expected prediction
    
    Returns:
        bool: Verification passed
    """
    try:
        import onnxruntime as ort
        
        sess = ort.InferenceSession(onnx_path)
        
        input_name = sess.get_inputs()[0].name
        result = sess.run(None, {input_name: test_input.astype(np.float32)})
        
        prediction = result[0]
        probabilities = result[1] if len(result) > 1 else None
        
        print(f"[ONNX Verify] Input shape: {test_input.shape}")
        print(f"              Prediction: {prediction}")
        if probabilities is not None:
            print(f"              Probabilities: {probabilities}")
        
        if expected_output is not None:
            match = np.array_equal(prediction.flatten(), expected_output.flatten())
            print(f"              Match expected: {match}")
            return match
        
        return True
        
    except Exception as e:
        print(f"[ERROR] ONNX verification failed: {e}")
        return False


def export_all_models(model_dir: str, export_dir: str):
    """
    Export all trained models to ONNX format.
    
    Expected files in model_dir:
      - rard_classifier.joblib
      - mves_classifier.joblib
      - rard_scaler.joblib
      - mves_scaler.joblib
    """
    model_path = Path(model_dir)
    export_path = Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    
    # Export RARD classifier (105 features)
    rard_clf_path = model_path / 'rard_classifier.joblib'
    rard_scaler_path = model_path / 'rard_scaler.joblib'
    
    if rard_clf_path.exists():
        clf = joblib.load(rard_clf_path)
        scaler = joblib.load(rard_scaler_path) if rard_scaler_path.exists() else None
        
        export_classifier_to_onnx(
            classifier=clf,
            scaler=scaler,
            n_features=105,
            output_path=str(export_path / 'rard_classifier.onnx'),
            model_name='RARD (Riemannian)'
        )
    
    # Export MVES classifier (203 features)
    mves_clf_path = model_path / 'mves_classifier.joblib'
    mves_scaler_path = model_path / 'mves_scaler.joblib'
    
    if mves_clf_path.exists():
        clf = joblib.load(mves_clf_path)
        scaler = joblib.load(mves_scaler_path) if mves_scaler_path.exists() else None
        
        export_classifier_to_onnx(
            classifier=clf,
            scaler=scaler,
            n_features=203,
            output_path=str(export_path / 'mves_classifier.onnx'),
            model_name='MVES (Statistical)'
        )
    
    print(f"\n[ONNX] All models exported to {export_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export HazeClue models to ONNX")
    parser.add_argument("--model-dir", default="trained_models", help="Directory with joblib models")
    parser.add_argument("--export-dir", default="onnx_models", help="Output directory for ONNX files")
    args = parser.parse_args()
    
    export_all_models(args.model_dir, args.export_dir)
