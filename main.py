"""
HazeClue AI — Main Entry Point
=================================
RARD–MVES v2.2: Hybrid Riemannian–Statistical EEG Inference System

Usage:
  python main.py --stew-dir data/raw/stew --mendeley-dir data/raw/mendeley
  python main.py --stew-dir data/raw/stew --folds 5 --augment
"""

import argparse
import numpy as np
from pathlib import Path

from data.dataset import HazeClueDataset
from training.train import train_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="HazeClue AI — RARD–MVES v2.2 Training Pipeline"
    )
    parser.add_argument("--stew-dir", type=str, default=None,
                        help="Path to STEW dataset directory")
    parser.add_argument("--mendeley-dir", type=str, default=None,
                        help="Path to Mendeley dataset directory")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of GroupKFold splits")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Enable noise robustness training (30% corruption)")
    parser.add_argument("--no-augment", action="store_false", dest="augment",
                        help="Disable noise augmentation")
    parser.add_argument("--output-dir", type=str, default="trained_models",
                        help="Directory to save trained models")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export models to ONNX after training")
    
    args = parser.parse_args()
    
    if args.stew_dir is None and args.mendeley_dir is None:
        parser.error("Must provide at least one dataset: --stew-dir or --mendeley-dir")
    
    # Load datasets
    print("=" * 60)
    print("  HazeClue AI — RARD–MVES v2.2")
    print("  Hybrid Riemannian–Statistical EEG Inference")
    print("=" * 60)
    print()
    
    dataset = HazeClueDataset()
    dataset.load(
        stew_dir=args.stew_dir,
        mendeley_dir=args.mendeley_dir
    )
    
    print(f"\nDataset: {dataset}")
    
    # Train
    results = train_pipeline(
        X=dataset.X,
        y=dataset.y,
        subject_ids=dataset.subject_ids,
        n_splits=args.folds,
        output_dir=args.output_dir,
        augment=args.augment
    )
    
    # Export
    if args.export_onnx:
        from export.export_onnx import export_all_models
        export_all_models(args.output_dir, "onnx_models")
    
    print("\n✅ Pipeline complete!")
    return results


if __name__ == "__main__":
    main()
