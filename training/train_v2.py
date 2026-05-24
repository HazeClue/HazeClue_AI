"""
Unified Training Script — All 3 Accuracy Levels
=================================================
Level 1: XGBoost + SVM + RF (Quick Wins)
Level 2: Euclidean Alignment + Filter Bank CSP
Level 3: EEGNet Deep Learning + Subject-Adaptive Fine-tuning

Run: python training/train_v2.py
"""

import numpy as np
import warnings
import joblib
import time
from pathlib import Path
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from sklearn.model_selection import GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.bandpass import preprocess_batch
from preprocessing.alignment import euclidean_alignment
from preprocessing.covariance import compute_stabilized_covariance
from features.rard_features import extract_rard_features, compute_frechet_mean
from features.mves_features import extract_mves_features
from features.csp_features import FilterBankCSP

warnings.filterwarnings('ignore')

# Try importing XGBoost (optional)
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed, using GradientBoosting instead")
    from sklearn.ensemble import GradientBoostingClassifier

# Try importing PyTorch (optional for Level 3)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARN] PyTorch not installed, skipping Level 3 (EEGNet)")


def extract_features_with_cov(X_filtered, P_ref):
    """Extract RARD + MVES features and covariance matrices."""
    N, C, T = X_filtered.shape
    
    rard_list = []
    mves_list = []
    covs_list = []
    
    for i in range(N):
        window = X_filtered[i]
        sigma = np.ones(C)
        
        P_spd, kappa, _ = compute_stabilized_covariance(window, sigma)
        covs_list.append(P_spd)
        
        try:
            feat_rard = extract_rard_features(P_spd, P_ref)
            if not np.all(np.isfinite(feat_rard)):
                feat_rard = np.zeros(105)
        except Exception:
            feat_rard = np.zeros(105)
        
        feat_mves = extract_mves_features(window, sigma)
        feat_mves = np.nan_to_num(feat_mves, nan=0.0, posinf=0.0, neginf=0.0)
        
        rard_list.append(feat_rard)
        mves_list.append(feat_mves)
        
        if (i + 1) % 1000 == 0:
            print(f"      {i+1}/{N}")
    
    return np.array(rard_list), np.array(mves_list), np.array(covs_list)


def compute_reference(X_filtered, n_ref=200):
    """Compute Fréchet reference from stable windows."""
    ref_covs = []
    indices = np.random.RandomState(42).choice(len(X_filtered), min(n_ref, len(X_filtered)), replace=False)
    
    for idx in indices:
        P_spd, kappa, _ = compute_stabilized_covariance(X_filtered[idx], np.ones(14))
        if kappa < 200:
            ref_covs.append(P_spd)
    
    ref_covs = np.array(ref_covs[:100])
    P_ref = compute_frechet_mean(ref_covs, max_iter=30)
    return P_ref


def run_level1(feat_combined, y, subject_ids, n_splits=5):
    """Level 1: Better classifiers + Feature Selection."""
    print("\n" + "=" * 60)
    print("LEVEL 1: XGBoost + SVM + RF + Feature Selection")
    print("=" * 60)
    
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    
    for fold, (tr, te) in enumerate(gkf.split(feat_combined, y, groups=subject_ids)):
        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=min(150, feat_combined.shape[1]))
        X_train_sel = selector.fit_transform(feat_combined[tr], y[tr])
        X_test_sel = selector.transform(feat_combined[te])
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_sel)
        X_test = scaler.transform(X_test_sel)
        
        # XGBoost
        if HAS_XGBOOST:
            clf_xgb = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0, n_jobs=-1
            )
        else:
            clf_xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
        clf_xgb.fit(X_train, y[tr])
        
        # SVM
        clf_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        clf_svm.fit(X_train, y[tr])
        
        # Random Forest
        clf_rf = RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1)
        clf_rf.fit(X_train, y[tr])
        
        # Individual predictions
        prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        prob_svm = clf_svm.predict_proba(X_test)[:, 1]
        prob_rf = clf_rf.predict_proba(X_test)[:, 1]
        
        # Ensemble
        prob_ens = 0.4 * prob_xgb + 0.35 * prob_svm + 0.25 * prob_rf
        pred_ens = (prob_ens > 0.5).astype(int)
        
        acc = accuracy_score(y[te], pred_ens)
        f1 = f1_score(y[te], pred_ens, average='weighted')
        kappa = cohen_kappa_score(y[te], pred_ens)
        
        acc_xgb = accuracy_score(y[te], (prob_xgb > 0.5).astype(int))
        acc_svm = accuracy_score(y[te], (prob_svm > 0.5).astype(int))
        acc_rf = accuracy_score(y[te], (prob_rf > 0.5).astype(int))
        
        print(f"  Fold {fold+1}: XGB={acc_xgb:.4f} SVM={acc_svm:.4f} RF={acc_rf:.4f} → Ensemble={acc:.4f}")
        results.append({'acc': acc, 'f1': f1, 'kappa': kappa})
    
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n  Level 1 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Level 1 F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return results


def run_level2(X_aligned, y, subject_ids, P_ref, n_splits=5):
    """Level 2: Euclidean Alignment + FBCSP + Better Classifiers."""
    print("\n" + "=" * 60)
    print("LEVEL 2: EA + FBCSP + Ensemble")
    print("=" * 60)
    
    # Extract covariance-based features on aligned data
    print("  [a] Extracting features on aligned data...")
    feat_rard, feat_mves, covs = extract_features_with_cov(X_aligned, P_ref)
    
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    
    for fold, (tr, te) in enumerate(gkf.split(X_aligned, y, groups=subject_ids)):
        # FBCSP features (fitted per fold)
        print(f"  [b] Fold {fold+1}: Fitting FBCSP...")
        fbcsp = FilterBankCSP(n_components=6)
        fbcsp.fit(X_aligned[tr], y[tr])
        feat_csp_train = fbcsp.transform(X_aligned[tr])
        feat_csp_test = fbcsp.transform(X_aligned[te])
        
        # Combine ALL features: RARD(105) + MVES(203) + CSP(30)
        X_train_all = np.hstack([feat_rard[tr], feat_mves[tr], feat_csp_train])
        X_test_all = np.hstack([feat_rard[te], feat_mves[te], feat_csp_test])
        
        # Feature selection
        selector = SelectKBest(mutual_info_classif, k=min(200, X_train_all.shape[1]))
        X_train_sel = selector.fit_transform(X_train_all, y[tr])
        X_test_sel = selector.transform(X_test_all)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_sel)
        X_test = scaler.transform(X_test_sel)
        
        # XGBoost
        if HAS_XGBOOST:
            clf_xgb = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric='logloss',
                verbosity=0, n_jobs=-1
            )
        else:
            clf_xgb = GradientBoostingClassifier(n_estimators=200, max_depth=5)
        clf_xgb.fit(X_train, y[tr])
        
        # SVM
        clf_svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        clf_svm.fit(X_train, y[tr])
        
        # RF
        clf_rf = RandomForestClassifier(n_estimators=300, max_depth=15, n_jobs=-1)
        clf_rf.fit(X_train, y[tr])
        
        # Ensemble
        prob_xgb = clf_xgb.predict_proba(X_test)[:, 1]
        prob_svm = clf_svm.predict_proba(X_test)[:, 1]
        prob_rf = clf_rf.predict_proba(X_test)[:, 1]
        prob_ens = 0.4 * prob_xgb + 0.35 * prob_svm + 0.25 * prob_rf
        pred_ens = (prob_ens > 0.5).astype(int)
        
        acc = accuracy_score(y[te], pred_ens)
        f1 = f1_score(y[te], pred_ens, average='weighted')
        kappa = cohen_kappa_score(y[te], pred_ens)
        
        print(f"  Fold {fold+1}: Ensemble={acc:.4f} | F1={f1:.4f} | κ={kappa:.4f}")
        results.append({'acc': acc, 'f1': f1, 'kappa': kappa})
    
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n  Level 2 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Level 2 F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return results


def run_level3(X_aligned, y, subject_ids, n_splits=5, epochs=30):
    """Level 3: EEGNet Deep Learning."""
    if not HAS_TORCH:
        print("\n[SKIP] Level 3 requires PyTorch")
        return []
    
    print("\n" + "=" * 60)
    print("LEVEL 3: EEGNet Deep Learning")
    print("=" * 60)
    
    from models.eegnet import EEGNet
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    
    for fold, (tr, te) in enumerate(gkf.split(X_aligned, y, groups=subject_ids)):
        print(f"\n  --- FOLD {fold+1}/{n_splits} ---")
        
        # Prepare data: (N, 1, 14, 512)
        X_train_t = torch.FloatTensor(X_aligned[tr][:, np.newaxis, :, :]).to(device)
        y_train_t = torch.LongTensor(y[tr]).to(device)
        X_test_t = torch.FloatTensor(X_aligned[te][:, np.newaxis, :, :]).to(device)
        y_test_t = torch.LongTensor(y[te]).to(device)
        
        # Normalize per-sample
        for i in range(len(X_train_t)):
            X_train_t[i] = (X_train_t[i] - X_train_t[i].mean()) / (X_train_t[i].std() + 1e-8)
        for i in range(len(X_test_t)):
            X_test_t[i] = (X_test_t[i] - X_test_t[i].mean()) / (X_test_t[i].std() + 1e-8)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
        
        model = EEGNet(n_channels=14, n_samples=512, n_classes=2, dropout=0.5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # Training
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
                total += len(y_batch)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                train_acc = correct / total
                print(f"    Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, train_acc={train_acc:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            # Process in batches to avoid OOM
            all_preds = []
            batch_size = 128
            for i in range(0, len(X_test_t), batch_size):
                batch = X_test_t[i:i+batch_size]
                output = model(batch)
                pred = output.argmax(dim=1)
                all_preds.append(pred.cpu().numpy())
            
            pred_test = np.concatenate(all_preds)
            y_test_np = y[te]
        
        acc = accuracy_score(y_test_np, pred_test)
        f1 = f1_score(y_test_np, pred_test, average='weighted')
        kappa = cohen_kappa_score(y_test_np, pred_test)
        
        print(f"  Fold {fold+1}: EEGNet Acc={acc:.4f} | F1={f1:.4f} | κ={kappa:.4f}")
        print(f"  Model params: {model.count_parameters():,}")
        
        results.append({'acc': acc, 'f1': f1, 'kappa': kappa})
    
    accs = [r['acc'] for r in results]
    f1s = [r['f1'] for r in results]
    print(f"\n  Level 3 Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Level 3 F1:       {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    return results


def run_all_levels():
    """Run all 3 accuracy improvement levels and compare."""
    from data.dataset import HazeClueDataset
    
    print("=" * 60)
    print("RARD–MVES v2.2 — Full Accuracy Improvement Pipeline")
    print("=" * 60)
    
    # Load data
    print("\n[0] Loading datasets...")
    dataset = HazeClueDataset()
    dataset.load(
        stew_dir='data/raw/stew/STEW Dataset/',
        mendeley_dir='data/raw/mendeley/'
    )
    
    # Bandpass filter
    print("\n[1] Bandpass filtering...")
    X_filtered = preprocess_batch(dataset.X)
    
    # Euclidean Alignment
    print("\n[2] Euclidean Alignment...")
    X_aligned = euclidean_alignment(X_filtered, dataset.subject_ids)
    
    # Compute Fréchet reference (on aligned data)
    print("\n[3] Computing Fréchet reference...")
    P_ref = compute_reference(X_aligned)
    
    # Extract features
    print("\n[4] Extracting RARD + MVES features...")
    feat_rard, feat_mves, covs = extract_features_with_cov(X_aligned, P_ref)
    feat_combined = np.hstack([feat_rard, feat_mves])
    
    y = dataset.y
    subject_ids = dataset.subject_ids
    
    # =============================================
    # LEVEL 1: Better Classifiers
    # =============================================
    t1 = time.time()
    results_l1 = run_level1(feat_combined, y, subject_ids)
    t1_elapsed = time.time() - t1
    
    # =============================================
    # LEVEL 2: EA + FBCSP + Better Classifiers
    # =============================================
    t2 = time.time()
    results_l2 = run_level2(X_aligned, y, subject_ids, P_ref)
    t2_elapsed = time.time() - t2
    
    # =============================================
    # LEVEL 3: EEGNet
    # =============================================
    t3 = time.time()
    results_l3 = run_level3(X_aligned, y, subject_ids, epochs=30)
    t3_elapsed = time.time() - t3
    
    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    
    baseline = 0.6346  # Previous best (RARD LDA)
    
    print(f"  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'Time':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Baseline (LDA/LogReg)':<25} {baseline:>10.4f} {'—':>10} {'—':>10}")
    
    if results_l1:
        acc1 = np.mean([r['acc'] for r in results_l1])
        f11 = np.mean([r['f1'] for r in results_l1])
        print(f"  {'L1: XGB+SVM+RF':<25} {acc1:>10.4f} {f11:>10.4f} {t1_elapsed:>8.0f}s")
    
    if results_l2:
        acc2 = np.mean([r['acc'] for r in results_l2])
        f12 = np.mean([r['f1'] for r in results_l2])
        print(f"  {'L2: EA+FBCSP+Ensemble':<25} {acc2:>10.4f} {f12:>10.4f} {t2_elapsed:>8.0f}s")
    
    if results_l3:
        acc3 = np.mean([r['acc'] for r in results_l3])
        f13 = np.mean([r['f1'] for r in results_l3])
        print(f"  {'L3: EEGNet':<25} {acc3:>10.4f} {f13:>10.4f} {t3_elapsed:>8.0f}s")
    
    print()
    return results_l1, results_l2, results_l3


if __name__ == "__main__":
    run_all_levels()
