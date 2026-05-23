"""
Unified Dataset
================
Combines STEW and Mendeley datasets into a single unified interface.
Provides GroupKFold splitting by subject_id to prevent data leakage.
"""

import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import GroupKFold

from data.loaders.stew_loader import load_stew_dataset
from data.loaders.mendeley_loader import load_mendeley_dataset


class HazeClueDataset:
    """
    Unified EEG dataset combining STEW and Mendeley sources.
    
    Attributes:
        X: (N, 14, 512) EEG windows
        y: (N,) binary labels (0=rest/low, 1=workload/concentration)
        subject_ids: (N,) subject identifiers (unique across datasets)
        source: (N,) dataset source identifier ('stew' or 'mendeley')
    """
    
    def __init__(self):
        self.X = None
        self.y = None
        self.subject_ids = None
        self.source = None
    
    def load(
        self,
        stew_dir: Optional[str] = None,
        mendeley_dir: Optional[str] = None,
        stew_subjects: Optional[List[int]] = None,
        mendeley_subjects: Optional[List[int]] = None,
    ) -> 'HazeClueDataset':
        """
        Load one or both datasets.
        
        Args:
            stew_dir: Path to STEW data directory
            mendeley_dir: Path to Mendeley data directory
            stew_subjects: Optional subset of STEW subjects
            mendeley_subjects: Optional subset of Mendeley subjects
        """
        all_X, all_y, all_subj, all_src = [], [], [], []
        
        if stew_dir:
            X_s, y_s, subj_s = load_stew_dataset(stew_dir, stew_subjects)
            all_X.append(X_s)
            all_y.append(y_s)
            all_subj.append(subj_s)
            all_src.append(np.full(len(y_s), 'stew'))
        
        if mendeley_dir:
            X_m, y_m, subj_m = load_mendeley_dataset(mendeley_dir, mendeley_subjects)
            all_X.append(X_m)
            all_y.append(y_m)
            all_subj.append(subj_m)
            all_src.append(np.full(len(y_m), 'mendeley'))
        
        if not all_X:
            raise ValueError("Must provide at least one dataset directory")
        
        self.X = np.concatenate(all_X, axis=0)
        self.y = np.concatenate(all_y, axis=0)
        self.subject_ids = np.concatenate(all_subj, axis=0)
        self.source = np.concatenate(all_src, axis=0)
        
        print(f"\n[Dataset] Total: {self.X.shape[0]} windows, "
              f"{len(np.unique(self.subject_ids))} subjects")
        print(f"          Shape: {self.X.shape}")
        print(f"          Class balance: 0={int((self.y==0).sum())}, 1={int((self.y==1).sum())}")
        
        return self
    
    def get_group_kfold_splits(self, n_splits: int = 5):
        """
        Generate subject-level GroupKFold splits.
        
        CRITICAL: Windows from the same subject NEVER appear in both 
        train and test within the same fold. This prevents temporal leakage.
        
        Yields:
            (train_idx, test_idx) arrays for each fold
        """
        gkf = GroupKFold(n_splits=n_splits)
        for train_idx, test_idx in gkf.split(self.X, self.y, groups=self.subject_ids):
            # Verify no subject overlap
            train_subjects = set(self.subject_ids[train_idx])
            test_subjects = set(self.subject_ids[test_idx])
            assert len(train_subjects & test_subjects) == 0, \
                "LEAKAGE DETECTED: Subject overlap between train and test!"
            yield train_idx, test_idx
    
    def __len__(self):
        return len(self.y) if self.y is not None else 0
    
    def __repr__(self):
        if self.X is None:
            return "HazeClueDataset(empty)"
        return (f"HazeClueDataset(windows={len(self)}, "
                f"subjects={len(np.unique(self.subject_ids))}, "
                f"shape={self.X.shape})")
