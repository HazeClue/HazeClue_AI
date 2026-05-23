"""
Optimized Training Script — RARD–MVES v2.2
=============================================
Key fix: Train BOTH classifiers on ALL clean data.
Mode routing is for INFERENCE only — during training,
we maximize data available to each classifier.
"""

import numpy as np
import warnings
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import GroupKFold
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.bandpass import preprocess_batch
from preprocessing.sqi import compute_sqi
from preprocessing.covariance import compute_stabilized_covariance
from features.rard_features import extract_rard_features, compute_frechet_mean
from features.mves_features import extract_mves_features

warnings.filterwarnings('ignore', category=RuntimeWarning)


def extract_all_features(X, P_ref):
    """
    Extract BOTH RARD and MVES features for all windows.
    No routing — both classifiers see all data during training.
    """
    N, C, T = X.shape
    
    rard_list = []
    mves_list = []
    valid_indices = []
    
    for i in range(N):
        window = X[i]
        sigma = np.array([1.0] * C)  # Skip SQI for clean training data
        
        # Covariance for RARD
        P_spd, kappa, _ = compute_stabilized_covariance(window, sigma)
        
        # RARD features (105-d)
        try:
            feat_rard = extract_rard_features(P_spd, P_ref)
            if not np.all(np.isfinite(feat_rard)):
                feat_rard = np.zeros(105)
        except Exception:
            feat_rard = np.zeros(105)
        
        # MVES features (~203-d)
        feat_mves = extract_mves_features(window, sigma)
        feat_mves = np.nan_to_num(feat_mves, nan=0.0, posinf=0.0, neginf=0.0)
        
        rard_list.append(feat_rard)
        mves_list.append(feat_mves)
        valid_indices.append(i)
        
        if (i + 1) % 500 == 0:
            print(f"      Features extracted: {i+1}/{N}")
    
    return np.array(rard_list), np.array(mves_list), np.array(valid_indices)


def train_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    n_splits: int = 5,
    output_dir: str = "trained_models",
    augment: bool = True
):
    """
    Optimized training: both classifiers trained on ALL data.
    Ensemble prediction combines both paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("RARD–MVES v2.2 Training Pipeline (Optimized)")
    print("=" * 60)
    print(f"Total windows: {len(y)}")
    print(f"Subjects: {len(np.unique(subject_ids))}")
    print(f"Class balance: 0={int((y==0).sum())}, 1={int((y==1).sum())}")
    print()
    
    # Step 1: Bandpass filter
    print("[1/4] Bandpass filtering...")
    X_filtered = preprocess_batch(X)
    
    # Step 2: Compute global Fréchet reference
    print("[2/4] Computing Fréchet reference from stable windows...")
    ref_covs = []
    n_ref = min(300, len(X_filtered))
    ref_idx = np.random.RandomState(42).choice(len(X_filtered), n_ref, replace=False)
    
    for idx in ref_idx:
        sigma = np.ones(14)
        P_spd, kappa, _ = compute_stabilized_covariance(X_filtered[idx], sigma)
        if kappa < 200:
            ref_covs.append(P_spd)
    
    print(f"   Using {len(ref_covs)} stable covariances for reference")
    ref_covs_arr = np.array(ref_covs[:150])
    P_ref = compute_frechet_mean(ref_covs_arr, max_iter=30)
    
    # Save reference
    np.save(output_path / 'P_ref.npy', P_ref)
    
    # Step 3: Extract ALL features
    print("[3/4] Extracting features for all windows...")
    feat_rard, feat_mves, valid_idx = extract_all_features(X_filtered, P_ref)
    y_valid = y[valid_idx]
    subj_valid = subject_ids[valid_idx]
    
    # Combine features: RARD(105) + MVES(203) = 308 total
    feat_combined = np.hstack([feat_rard, feat_mves])
    
    print(f"   RARD features: {feat_rard.shape}")
    print(f"   MVES features: {feat_mves.shape}")
    print(f"   Combined: {feat_combined.shape}")
    
    # Step 4: Cross-validation
    print("[4/4] GroupKFold Cross-Validation...")
    gkf = GroupKFold(n_splits=n_splits)
    
    fold_results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(feat_combined, y_valid, groups=subj_valid)):
        print(f"\n  --- FOLD {fold+1}/{n_splits} ---")
        
        # RARD classifier (LDA on 105 tangent features)
        scaler_r = StandardScaler()
        X_r_train = scaler_r.fit_transform(feat_rard[train_idx])
        X_r_test = scaler_r.transform(feat_rard[test_idx])
        
        clf_rard = LinearDiscriminantAnalysis()
        clf_rard.fit(X_r_train, y_valid[train_idx])
        pred_rard = clf_rard.predict(X_r_test)
        prob_rard = clf_rard.predict_proba(X_r_test)[:, 1]
        acc_rard = accuracy_score(y_valid[test_idx], pred_rard)
        
        # MVES classifier (LogReg on 203 statistical features)
        scaler_m = StandardScaler()
        X_m_train = scaler_m.fit_transform(feat_mves[train_idx])
        X_m_test = scaler_m.transform(feat_mves[test_idx])
        
        clf_mves = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf_mves.fit(X_m_train, y_valid[train_idx])
        pred_mves = clf_mves.predict(X_m_test)
        prob_mves = clf_mves.predict_proba(X_m_test)[:, 1]
        acc_mves = accuracy_score(y_valid[test_idx], pred_mves)
        
        # Combined classifier (LogReg on 308 features)
        scaler_c = StandardScaler()
        X_c_train = scaler_c.fit_transform(feat_combined[train_idx])
        X_c_test = scaler_c.transform(feat_combined[test_idx])
        
        clf_combined = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf_combined.fit(X_c_train, y_valid[train_idx])
        pred_combined = clf_combined.predict(X_c_test)
        prob_combined = clf_combined.predict_proba(X_c_test)[:, 1]
        acc_combined = accuracy_score(y_valid[test_idx], pred_combined)
        
        # Ensemble: average probabilities
        prob_ensemble = 0.5 * prob_rard + 0.3 * prob_mves + 0.2 * prob_combined
        pred_ensemble = (prob_ensemble > 0.5).astype(int)
        acc_ensemble = accuracy_score(y_valid[test_idx], pred_ensemble)
        f1_ensemble = f1_score(y_valid[test_idx], pred_ensemble, average='weighted')
        kappa_ensemble = cohen_kappa_score(y_valid[test_idx], pred_ensemble)
        
        print(f"    RARD (LDA):     {acc_rard:.4f}")
        print(f"    MVES (LogReg):  {acc_mves:.4f}")
        print(f"    Combined:       {acc_combined:.4f}")
        print(f"    Ensemble:       {acc_ensemble:.4f} | F1={f1_ensemble:.4f} | κ={kappa_ensemble:.4f}")
        
        fold_results.append({
            'fold': fold + 1,
            'rard_acc': acc_rard,
            'mves_acc': acc_mves,
            'combined_acc': acc_combined,
            'ensemble_acc': acc_ensemble,
            'f1': f1_ensemble,
            'kappa': kappa_ensemble,
        })
    
    # Summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    
    for key, label in [('rard_acc', 'RARD (LDA)'), ('mves_acc', 'MVES (LogReg)'), 
                        ('combined_acc', 'Combined'), ('ensemble_acc', 'Ensemble')]:
        vals = [r[key] for r in fold_results]
        print(f"  {label:20s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    
    f1s = [r['f1'] for r in fold_results]
    kappas = [r['kappa'] for r in fold_results]
    print(f"  {'Ensemble F1':20s}: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  {'Ensemble Kappa':20s}: {np.mean(kappas):.4f} ± {np.std(kappas):.4f}")
    
    # Train final models on all data
    print("\n[Final] Training on all data...")
    
    scaler_rard_final = StandardScaler()
    X_r_all = scaler_rard_final.fit_transform(feat_rard)
    clf_rard_final = LinearDiscriminantAnalysis()
    clf_rard_final.fit(X_r_all, y_valid)
    
    scaler_mves_final = StandardScaler()
    X_m_all = scaler_mves_final.fit_transform(feat_mves)
    clf_mves_final = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf_mves_final.fit(X_m_all, y_valid)
    
    # Save models
    joblib.dump(clf_rard_final, output_path / 'rard_classifier.joblib')
    joblib.dump(clf_mves_final, output_path / 'mves_classifier.joblib')
    joblib.dump(scaler_rard_final, output_path / 'rard_scaler.joblib')
    joblib.dump(scaler_mves_final, output_path / 'mves_scaler.joblib')
    
    print(f"  Models saved to {output_path}/")
    
    return fold_results


if __name__ == "__main__":
    print("Usage: python -m training.train")
