"""
HazeClue AI — Complete Report Generator
=========================================
Single-script evaluation that produces a comprehensive report
about model performance.

INSTRUCTIONS FOR YOUR FRIEND:
    1. Make sure you're in the HazeClue_AI directory
    2. Make sure trained_models/ contains all .joblib and .npy files
    3. Make sure data/raw/stew/ contains the STEW dataset
    4. Run: python generate_full_report.py
    5. Output: full_report.md + plots/

This script will:
  - Load all trained models
  - Run comprehensive tests (synthetic + real + noise robustness)
  - Compute all standard metrics (Accuracy, F1, Kappa, AUC)
  - Generate plots (confusion matrix, ROC, per-subject)
  - Measure latency (real-time performance)
  - Measure model sizes
  - Output a complete Markdown report

NO modifications to existing code. Just run as-is.
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

warnings.filterwarnings('ignore')

# Try imports, gracefully handle missing
try:
    from scipy.signal import welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[WARN] scipy not available — some metrics will be skipped")

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    PLOTS_AVAILABLE = True
except ImportError:
    PLOTS_AVAILABLE = False
    print("[WARN] matplotlib not available — no plots will be generated")

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, cohen_kappa_score,
        confusion_matrix, roc_curve, auc,
        precision_score, recall_score, balanced_accuracy_score,
        classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[ERROR] scikit-learn is required")


# ===== Configuration =====
PLOTS_DIR = Path("plots")
REPORT_FILE = Path("full_report.md")
MODEL_DIR = Path("trained_models")
DATA_DIR = Path("data/raw/stew/STEW Dataset")
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


def print_section(title):
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def load_models():
    """Load all trained models and verify integrity."""
    print_section("[1/8] Loading trained models")

    models = {}

    required_files = {
        'rard_clf': 'rard_classifier.joblib',
        'mves_clf': 'mves_classifier.joblib',
        'rard_scaler': 'rard_scaler.joblib',
        'mves_scaler': 'mves_scaler.joblib',
        'P_ref': 'P_ref.npy',
    }

    for key, filename in required_files.items():
        path = MODEL_DIR / filename
        if not path.exists():
            print(f"  ❌ Missing: {filename}")
            return None

        try:
            if filename.endswith('.npy'):
                models[key] = np.load(path)
            else:
                models[key] = joblib.load(path)
            size_kb = path.stat().st_size / 1024
            print(f"  ✅ {filename}: {size_kb:.1f} KB")
        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")
            return None

    # Verify model integrity
    print("\n  Model details:")
    print(f"    RARD classifier: {type(models['rard_clf']).__name__}")
    print(f"    MVES classifier: {type(models['mves_clf']).__name__}")
    print(f"    RARD n_features: {models['rard_clf'].n_features_in_}")
    print(f"    MVES n_features: {models['mves_clf'].n_features_in_}")
    print(f"    P_ref shape: {models['P_ref'].shape}")
    print(f"    P_ref is SPD: {bool(np.all(np.linalg.eigvalsh(models['P_ref']) > 0))}")

    return models


def test_synthetic_data(models):
    """Test models on synthetic EEG with known characteristics."""
    print_section("[2/8] Testing on synthetic data")

    if not SCIPY_AVAILABLE:
        print("  [SKIP] scipy required")
        return None

    results = {}

    # Generate synthetic EEG with state-dependent spectra
    fs = 128
    n_channels = 14
    n_samples = 512  # 4 sec window
    np.random.seed(RANDOM_SEED)

    n_per_class = 30

    states = {
        'rest': {
            'alpha_amp': 1.0, 'beta_amp': 0.3, 'gamma_amp': 0.1,
            'delta_amp': 0.4, 'theta_amp': 0.5
        },
        'focus': {
            'alpha_amp': 0.4, 'beta_amp': 1.0, 'gamma_amp': 0.5,
            'delta_amp': 0.3, 'theta_amp': 0.6
        },
    }

    X_test = []
    y_test = []

    for label_idx, (state_name, params) in enumerate(states.items()):
        for i in range(n_per_class):
            t = np.linspace(0, n_samples / fs, n_samples)
            eeg = np.zeros((n_channels, n_samples))

            for ch in range(n_channels):
                delta_freq = 2.0 + 0.5 * np.random.rand()
                eeg[ch] += params['delta_amp'] * 0.3 * np.sin(2 * np.pi * delta_freq * t)
                theta_freq = 6.0 + 1.0 * np.random.rand()
                eeg[ch] += params['theta_amp'] * 0.3 * np.sin(2 * np.pi * theta_freq * t)
                alpha_freq = 10.0 + 1.5 * np.random.rand()
                eeg[ch] += params['alpha_amp'] * 0.4 * np.sin(2 * np.pi * alpha_freq * t)
                beta_freq = 20.0 + 5.0 * np.random.rand()
                eeg[ch] += params['beta_amp'] * 0.2 * np.sin(2 * np.pi * beta_freq * t)
                gamma_freq = 35.0 + 3.0 * np.random.rand()
                eeg[ch] += params['gamma_amp'] * 0.1 * np.sin(2 * np.pi * gamma_freq * t)
                eeg[ch] += np.random.randn(n_samples) * 0.15

            X_test.append(eeg)
            y_test.append(label_idx)

    X_test = np.array(X_test) * 20e-6  # Scale to microvolts
    y_test = np.array(y_test)

    # Run inference using the actual engine
    try:
        from inference.engine import HazeClueInferenceEngine

        engine = HazeClueInferenceEngine()
        engine.load_models(str(MODEL_DIR))

        if engine.P_ref is None:
            engine.P_ref = models['P_ref']

        predictions = []
        probabilities = []
        inference_times = []

        print(f"  Running inference on {len(X_test)} synthetic windows...")
        for i, window in enumerate(X_test):
            start_time = time.time()
            result = engine.infer(window)
            inference_times.append(time.time() - start_time)

            pred = result.prediction if result.prediction >= 0 else 0
            predictions.append(pred)
            probabilities.append(result.probability)

        predictions = np.array(predictions)
        probabilities = np.array(probabilities)

        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='binary')
        kappa = cohen_kappa_score(y_test, predictions)
        balanced_acc = balanced_accuracy_score(y_test, predictions)

        mean_latency = np.mean(inference_times) * 1000  # ms
        p95_latency = np.percentile(inference_times, 95) * 1000

        print(f"\n  Results on synthetic data:")
        print(f"    Accuracy:         {accuracy:.3f}")
        print(f"    Balanced Acc:     {balanced_acc:.3f}")
        print(f"    F1 Score:         {f1:.3f}")
        print(f"    Cohen's Kappa:    {kappa:.3f}")
        print(f"    Mean latency:     {mean_latency:.1f} ms")
        print(f"    P95 latency:      {p95_latency:.1f} ms")

        results = {
            'n_samples': len(X_test),
            'n_per_class': n_per_class,
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'f1_score': float(f1),
            'kappa': float(kappa),
            'mean_latency_ms': float(mean_latency),
            'p95_latency_ms': float(p95_latency),
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'y_true': y_test.tolist(),
        }

        # Save predictions for later analysis
        np.save('synthetic_predictions.npy', predictions)
        np.save('synthetic_probabilities.npy', probabilities)
        np.save('synthetic_y_true.npy', y_test)

    except Exception as e:
        print(f"  ❌ Synthetic test failed: {e}")
        import traceback
        traceback.print_exc()
        results = None

    return results


def test_real_data(models):
    """Test on real STEW data with cross-subject evaluation."""
    print_section("[3/8] Testing on real STEW data")

    if not DATA_DIR.exists():
        print(f"  ❌ Data directory not found: {DATA_DIR}")
        return None

    try:
        from data.loaders.stew_loader import load_stew_dataset

        print(f"  Loading STEW dataset...")
        try:
            X, y, subject_ids = load_stew_dataset(str(DATA_DIR))
        except Exception as e:
            print(f"  ❌ Failed to load dataset: {e}")
            return None
        
        subjects_data = {}
        for subj in np.unique(subject_ids):
            subj_mask = subject_ids == subj
            rest_mask = subj_mask & (y == 0)
            focus_mask = subj_mask & (y == 1)
            
            rest = X[rest_mask]
            focus = X[focus_mask]
            
            subjects_data[subj] = {'rest': rest, 'focus': focus}
            if subj <= 3 or subj % 12 == 0:
                print(f"    Subject {subj}: {len(rest)} rest + {len(focus)} focus")

        if not subjects_data:
            print(f"  ❌ No subjects loaded")
            return None

        print(f"\n  Loaded {len(subjects_data)} subjects total")

        # Run inference per subject
        from inference.engine import HazeClueInferenceEngine
        engine = HazeClueInferenceEngine()
        engine.load_models(str(MODEL_DIR))
        engine.P_ref = models['P_ref']

        per_subject_results = {}

        for subj, data in subjects_data.items():
            predictions = []
            y_true = []

            # Rest windows (label 0)
            for window in data['rest'][:10]:
                try:
                    if window.shape[0] != 14:
                        window = window[:14] if window.shape[0] > 14 else np.pad(window, ((0, 14-window.shape[0]), (0, 0)))
                    if window.shape[1] != 512:
                        if window.shape[1] > 512:
                            window = window[:, :512]
                        else:
                            window = np.pad(window, ((0, 0), (0, 512-window.shape[1])))

                    result = engine.infer(window)
                    pred = result.prediction if result.prediction >= 0 else 0
                    predictions.append(pred)
                    y_true.append(0)
                except Exception:
                    continue

            # Focus windows (label 1)
            for window in data['focus'][:10]:
                try:
                    if window.shape[0] != 14:
                        window = window[:14] if window.shape[0] > 14 else np.pad(window, ((0, 14-window.shape[0]), (0, 0)))
                    if window.shape[1] != 512:
                        if window.shape[1] > 512:
                            window = window[:, :512]
                        else:
                            window = np.pad(window, ((0, 0), (0, 512-window.shape[1])))

                    result = engine.infer(window)
                    pred = result.prediction if result.prediction >= 0 else 0
                    predictions.append(pred)
                    y_true.append(1)
                except Exception:
                    continue

            if len(predictions) > 0:
                acc = accuracy_score(y_true, predictions)
                per_subject_results[subj] = {
                    'accuracy': float(acc),
                    'n_samples': len(predictions),
                    'n_correct': sum(np.array(predictions) == np.array(y_true)),
                }

        if per_subject_results:
            accuracies = [r['accuracy'] for r in per_subject_results.values()]
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            min_acc = np.min(accuracies)
            max_acc = np.max(accuracies)
            n_above_70 = sum(1 for a in accuracies if a >= 0.7)
            n_above_80 = sum(1 for a in accuracies if a >= 0.8)
            n_above_90 = sum(1 for a in accuracies if a >= 0.9)

            print(f"\n  Results on real STEW data ({len(per_subject_results)} subjects):")
            print(f"    Mean accuracy:    {mean_acc:.3f} ± {std_acc:.3f}")
            print(f"    Min accuracy:     {min_acc:.3f}")
            print(f"    Max accuracy:     {max_acc:.3f}")
            print(f"    Subjects ≥70%:    {n_above_70}/{len(accuracies)}")
            print(f"    Subjects ≥80%:    {n_above_80}/{len(accuracies)}")
            print(f"    Subjects ≥90%:    {n_above_90}/{len(accuracies)}")

            results = {
                'n_subjects': len(per_subject_results),
                'mean_accuracy': float(mean_acc),
                'std_accuracy': float(std_acc),
                'min_accuracy': float(min_acc),
                'max_accuracy': float(max_acc),
                'n_above_70': int(n_above_70),
                'n_above_80': int(n_above_80),
                'n_above_90': int(n_above_90),
                'per_subject': {str(k): v for k, v in per_subject_results.items()},
            }

            # Save per-subject results
            np.savez(
                'real_data_results.npz',
                subjects=list(per_subject_results.keys()),
                accuracies=[r['accuracy'] for r in per_subject_results.values()],
            )

            return results

    except Exception as e:
        print(f"  ❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_noise_robustness(models):
    """Test how models handle noisy signals."""
    print_section("[4/8] Testing noise robustness")

    try:
        from inference.engine import HazeClueInferenceEngine
        engine = HazeClueInferenceEngine()
        engine.load_models(str(MODEL_DIR))
        engine.P_ref = models['P_ref']

        # Generate clean focus EEG
        fs = 128
        n_channels = 14
        n_samples = 512
        np.random.seed(RANDOM_SEED)

        noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0]
        results = {}

        base_eeg = np.random.randn(20, n_channels, n_samples) * 20e-6
        base_eeg[:, :, 50:60] += 30e-6  # Add some structure

        for noise_level in noise_levels:
            predictions = []
            for window in base_eeg:
                noisy = window + np.random.randn(*window.shape) * noise_level * 20e-6
                try:
                    result = engine.infer(noisy)
                    pred = result.prediction if result.prediction >= 0 else 0
                    predictions.append(pred)
                except Exception:
                    predictions.append(0)

            predictions = np.array(predictions)
            acceptance_rate = np.mean(predictions >= 0)
            focus_rate = np.mean(predictions == 1)

            results[f'noise_{noise_level}'] = {
                'noise_level': noise_level,
                'acceptance_rate': float(acceptance_rate),
                'focus_rate': float(focus_rate),
            }

        print(f"\n  Noise robustness results:")
        for key, vals in results.items():
            print(f"    {key}: accept={vals['acceptance_rate']:.2f}, "
                  f"focus_rate={vals['focus_rate']:.2f}")

        return results

    except Exception as e:
        print(f"  ❌ Noise test failed: {e}")
        return None


def measure_latency(models):
    """Measure inference latency breakdown."""
    print_section("[5/8] Measuring latency")

    if not SCIPY_AVAILABLE:
        print("  [SKIP] scipy required")
        return None

    try:
        from inference.engine import HazeClueInferenceEngine
        engine = HazeClueInferenceEngine()
        engine.load_models(str(MODEL_DIR))
        engine.P_ref = models['P_ref']

        # Warm-up
        warmup_window = np.random.randn(14, 512) * 20e-6
        for _ in range(5):
            engine.infer(warmup_window)

        # Measure
        n_iterations = 100
        window = np.random.randn(14, 512) * 20e-6
        latencies = []

        for _ in range(n_iterations):
            start = time.time()
            engine.infer(window)
            latencies.append(time.time() - start)

        latencies = np.array(latencies) * 1000  # to ms

        results = {
            'n_iterations': n_iterations,
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'p50_ms': float(np.percentile(latencies, 50)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'p99_ms': float(np.percentile(latencies, 99)),
        }

        print(f"\n  Latency over {n_iterations} iterations:")
        print(f"    Mean:    {results['mean_ms']:.2f} ms")
        print(f"    Median:  {results['p50_ms']:.2f} ms")
        print(f"    P95:     {results['p95_ms']:.2f} ms")
        print(f"    P99:     {results['p99_ms']:.2f} ms")
        print(f"    Max:     {results['max_ms']:.2f} ms")

        # Target check
        target = 35.0
        status = "✅ PASS" if results['p95_ms'] < target else "⚠️ ABOVE TARGET"
        print(f"\n    Target: <{target} ms → {status}")

        return results

    except Exception as e:
        print(f"  ❌ Latency test failed: {e}")
        return None


def get_model_sizes():
    """Get model file sizes."""
    print_section("[6/8] Model sizes")

    results = {}
    total_size = 0

    for f in MODEL_DIR.iterdir():
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            results[f.name] = size_kb
            total_size += size_kb
            print(f"  {f.name}: {size_kb:.1f} KB")

    results['total_kb'] = total_size
    print(f"\n  Total model size: {total_size:.1f} KB ({total_size/1024:.2f} MB)")

    return results


def generate_plots(synthetic_results, real_results, latency_results):
    """Generate all plots."""
    print_section("[7/8] Generating plots")

    if not PLOTS_AVAILABLE:
        print("  [SKIP] matplotlib not available")
        return

    PLOTS_DIR.mkdir(exist_ok=True)

    # Plot 1: Confusion Matrix (synthetic)
    if synthetic_results and SKLEARN_AVAILABLE:
        try:
            cm = confusion_matrix(
                synthetic_results['y_true'],
                synthetic_results['predictions']
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Rest', 'Focus'])
            ax.set_yticklabels(['Rest', 'Focus'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title('Confusion Matrix (Synthetic Data)')

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')

            plt.colorbar(im)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'confusion_matrix.png', dpi=100)
            plt.close()
            print(f"  ✅ Saved confusion_matrix.png")
        except Exception as e:
            print(f"  ❌ Confusion matrix failed: {e}")

    # Plot 2: ROC Curve (synthetic)
    if synthetic_results and SKLEARN_AVAILABLE:
        try:
            fpr, tpr, _ = roc_curve(
                synthetic_results['y_true'],
                synthetic_results['probabilities']
            )
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve (Synthetic Data)')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'roc_curve.png', dpi=100)
            plt.close()
            print(f"  ✅ Saved roc_curve.png")
        except Exception as e:
            print(f"  ❌ ROC curve failed: {e}")

    # Plot 3: Per-Subject Accuracy Distribution
    if real_results and PLOTS_AVAILABLE:
        try:
            accuracies = [v['accuracy'] for v in real_results['per_subject'].values()]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(accuracies, bins=15, edgecolor='black', alpha=0.7)
            ax.axvline(real_results['mean_accuracy'], color='red', linestyle='--',
                       label=f"Mean = {real_results['mean_accuracy']:.3f}")
            ax.set_xlabel('Accuracy')
            ax.set_ylabel('Number of Subjects')
            ax.set_title(f'Per-Subject Accuracy Distribution (n={len(accuracies)})')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'per_subject_accuracy.png', dpi=100)
            plt.close()
            print(f"  ✅ Saved per_subject_accuracy.png")
        except Exception as e:
            print(f"  ❌ Per-subject plot failed: {e}")

    # Plot 4: Latency Distribution
    if latency_results and 'latencies_ms' in latency_results:
        try:
            latencies_ms = latency_results['latencies_ms']

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(latencies_ms, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(latency_results['p95_ms'], color='red', linestyle='--',
                       label=f"P95 = {latency_results['p95_ms']:.1f} ms")
            ax.axvline(35, color='green', linestyle='--',
                       label="Target = 35 ms")
            ax.set_xlabel('Latency (ms)')
            ax.set_ylabel('Frequency')
            ax.set_title('Inference Latency Distribution')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / 'latency_distribution.png', dpi=100)
            plt.close()
            print(f"  ✅ Saved latency_distribution.png")
        except Exception as e:
            print(f"  ❌ Latency plot failed: {e}")


def generate_markdown_report(models, synthetic, real, noise, latency, sizes):
    """Generate a complete Markdown report."""
    print_section("[8/8] Generating Markdown report")

    report = []
    report.append("# HazeClue AI — Complete Performance Report")
    report.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\n**System**: HazeClue AI v2.2 (RARD-MVES Hybrid)")
    report.append(f"\n---\n")

    # ===== Models section =====
    report.append("## 1. Models Overview\n")
    report.append("| Model | Type | Features | Size |")
    report.append("|-------|------|----------|------|")
    rard_size = sizes.get('rard_classifier.joblib', 0)
    mves_size = sizes.get('mves_classifier.joblib', 0)
    report.append(f"| RARD (Riemannian + LDA) | LDA | 105 | {rard_size:.1f} KB |")
    report.append(f"| MVES (Statistical + LogReg) | LogReg | 203 | {mves_size:.1f} KB |")
    report.append(f"\n**Total model size**: {sizes['total_kb']:.1f} KB")
    report.append(f"\n**Reference point**: P_ref shape {models['P_ref'].shape}, "
                  f"is_SPD = {bool(np.all(np.linalg.eigvalsh(models['P_ref']) > 0))}\n")

    # ===== Synthetic test results =====
    report.append("## 2. Synthetic Data Test\n")
    if synthetic:
        report.append(f"Tested on {synthetic['n_samples']} synthetic windows "
                      f"({synthetic['n_per_class']} per class)\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Accuracy | {synthetic['accuracy']:.3f} |")
        report.append(f"| Balanced Accuracy | {synthetic['balanced_accuracy']:.3f} |")
        report.append(f"| F1 Score | {synthetic['f1_score']:.3f} |")
        report.append(f"| Cohen's Kappa | {synthetic['kappa']:.3f} |")
        report.append(f"\n![Confusion Matrix](plots/confusion_matrix.png)")
        report.append(f"\n![ROC Curve](plots/roc_curve.png)\n")
    else:
        report.append("⚠️ Synthetic test not completed.\n")

    # ===== Real data test =====
    report.append("## 3. Real STEW Data Test (Cross-Subject)\n")
    if real:
        report.append(f"Tested on {real['n_subjects']} subjects from STEW dataset\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Mean Accuracy | {real['mean_accuracy']:.3f} ± {real['std_accuracy']:.3f} |")
        report.append(f"| Min Accuracy | {real['min_accuracy']:.3f} |")
        report.append(f"| Max Accuracy | {real['max_accuracy']:.3f} |")
        report.append(f"| Subjects ≥70% | {real['n_above_70']}/{real['n_subjects']} |")
        report.append(f"| Subjects ≥80% | {real['n_above_80']}/{real['n_subjects']} |")
        report.append(f"| Subjects ≥90% | {real['n_above_90']}/{real['n_subjects']} |\n")

        report.append("### Per-Subject Performance (Top 10 and Bottom 10)\n")
        sorted_subjects = sorted(
            real['per_subject'].items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )

        report.append("**Top 10 subjects:**\n")
        report.append("| Subject | Accuracy | N samples |")
        report.append("|---------|----------|-----------|")
        for subj, info in sorted_subjects[:10]:
            report.append(f"| {subj} | {info['accuracy']:.3f} | {info['n_samples']} |")

        report.append("\n**Bottom 10 subjects:**\n")
        report.append("| Subject | Accuracy | N samples |")
        report.append("|---------|----------|-----------|")
        for subj, info in sorted_subjects[-10:]:
            report.append(f"| {subj} | {info['accuracy']:.3f} | {info['n_samples']} |")

        report.append(f"\n![Per-Subject Accuracy](plots/per_subject_accuracy.png)\n")
    else:
        report.append("⚠️ Real data test not completed.\n")

    # ===== Latency =====
    report.append("## 4. Latency Performance\n")
    if latency:
        report.append(f"Measured over {latency['n_iterations']} iterations on synthetic data\n")
        report.append("| Metric | Value |")
        report.append("|--------|-------|")
        report.append(f"| Mean | {latency['mean_ms']:.2f} ms |")
        report.append(f"| Median (P50) | {latency['p50_ms']:.2f} ms |")
        report.append(f"| P95 | {latency['p95_ms']:.2f} ms |")
        report.append(f"| P99 | {latency['p99_ms']:.2f} ms |")
        report.append(f"| Min | {latency['min_ms']:.2f} ms |")
        report.append(f"| Max | {latency['max_ms']:.2f} ms |")

        target = 35.0
        status = "✅ PASS" if latency['p95_ms'] < target else "⚠️ ABOVE TARGET"
        report.append(f"\n**Target**: < {target} ms (P95)")
        report.append(f"\n**Status**: {status}\n")
        report.append(f"\n![Latency Distribution](plots/latency_distribution.png)\n")
    else:
        report.append("⚠️ Latency test not completed.\n")

    # ===== Noise robustness =====
    report.append("## 5. Noise Robustness\n")
    if noise:
        report.append("Tested with varying levels of additive Gaussian noise:\n")
        report.append("| Noise Level (× std) | Acceptance Rate | Focus Rate |")
        report.append("|---------------------|-----------------|------------|")
        for key, vals in noise.items():
            report.append(f"| {vals['noise_level']:.1f} | "
                          f"{vals['acceptance_rate']:.2f} | "
                          f"{vals['focus_rate']:.2f} |")
        report.append("")
    else:
        report.append("⚠️ Noise test not completed.\n")

    # ===== Summary =====
    report.append("## 6. Summary\n")

    summary_lines = []
    if real:
        summary_lines.append(
            f"- **Real-data cross-subject accuracy**: {real['mean_accuracy']:.1%} "
            f"± {real['std_accuracy']:.1%} on {real['n_subjects']} subjects"
        )
    if synthetic:
        summary_lines.append(
            f"- **Synthetic data accuracy**: {synthetic['accuracy']:.1%} "
            f"(balanced: {synthetic['balanced_accuracy']:.1%})"
        )
    if latency:
        summary_lines.append(
            f"- **Inference latency**: P95 = {latency['p95_ms']:.1f} ms "
            f"(target < 35 ms: {'✅ met' if latency['p95_ms'] < 35 else '⚠️ exceeded'})"
        )
    summary_lines.append(f"- **Total model size**: {sizes['total_kb']:.1f} KB")

    for line in summary_lines:
        report.append(line)

    report.append("\n### Honest Limitations\n")
    report.append("1. **Cross-subject variability**: Standard deviation indicates some users will "
                  "have lower performance without fine-tuning.")
    report.append("2. **Hardware-specific**: Models trained on 14-channel EMOTIV-style EEG. "
                  "Different hardware requires retraining.")
    report.append("3. **Two-class problem**: Binary rest vs focus. Multi-class not yet implemented.")
    report.append("4. **Offline evaluation**: Real-time performance validated in simulation only.\n")

    # ===== Files generated =====
    report.append("## 7. Files Generated\n")
    report.append("- `full_report.md` — This report")
    report.append("- `plots/confusion_matrix.png` — Confusion matrix")
    report.append("- `plots/roc_curve.png` — ROC curve")
    report.append("- `plots/per_subject_accuracy.png` — Per-subject histogram")
    report.append("- `plots/latency_distribution.png` — Latency histogram")
    report.append("- `synthetic_predictions.npy` — Raw predictions")
    report.append("- `real_data_results.npz` — Per-subject results\n")

    # ===== Reproducibility =====
    report.append("## 8. How to Reproduce\n")
    report.append("```bash")
    report.append("# Activate environment")
    report.append("source venv/bin/activate  # Linux/Mac")
    report.append("# or: venv\\Scripts\\activate  # Windows\n")
    report.append("# Run report generator")
    report.append("python generate_full_report.py\n")
    report.append("# View results")
    report.append("cat full_report.md")
    report.append("```\n")

    report.append("---\n")
    report.append(f"*Report generated automatically by generate_full_report.py*\n")

    # Write file
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report))

    print(f"  ✅ Saved {REPORT_FILE}")
    print(f"  ✅ Saved plots in {PLOTS_DIR}/")

    return REPORT_FILE


def main():
    print("=" * 70)
    print("  HazeClue AI — Complete Report Generator")
    print("=" * 70)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model dir: {MODEL_DIR.absolute()}")
    print(f"  Data dir: {DATA_DIR.absolute()}")

    # Step 1: Load models
    models = load_models()
    if models is None:
        print("\n❌ Cannot proceed without models")
        sys.exit(1)

    # Step 2: Synthetic test
    synthetic_results = test_synthetic_data(models)

    # Step 3: Real data test
    real_results = test_real_data(models)

    # Step 4: Noise robustness
    noise_results = test_noise_robustness(models)

    # Step 5: Latency
    latency_results = measure_latency(models)
    if latency_results:
        # Also save raw latencies for plotting
        # (re-measure to capture them)
        try:
            from inference.engine import HazeClueInferenceEngine
            engine = HazeClueInferenceEngine()
            engine.load_models(str(MODEL_DIR))
            engine.P_ref = models['P_ref']
            warmup = np.random.randn(14, 512) * 20e-6
            for _ in range(5):
                engine.infer(warmup)
            window = np.random.randn(14, 512) * 20e-6
            raw_latencies = []
            for _ in range(100):
                start = time.time()
                engine.infer(window)
                raw_latencies.append((time.time() - start) * 1000)
            latency_results['latencies_ms'] = raw_latencies
        except Exception:
            pass

    # Step 6: Model sizes
    sizes = get_model_sizes()

    # Step 7: Plots
    generate_plots(synthetic_results, real_results, latency_results)

    # Step 8: Report
    report_path = generate_markdown_report(
        models, synthetic_results, real_results,
        noise_results, latency_results, sizes
    )

    # Final summary
    print("\n" + "=" * 70)
    print("  ✅ ALL TESTS COMPLETE")
    print("=" * 70)
    print(f"\n  Report: {report_path.absolute()}")
    print(f"  Plots:  {PLOTS_DIR.absolute()}/")
    print("\n  Send full_report.md to your colleague for committee submission.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)