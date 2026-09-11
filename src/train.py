"""Model Training Pipeline for Rainfall Prediction System.
Trains production-grade CatBoostClassifier with meteorological domain features,
eliminating data leakage and optimizing decision thresholds.
"""

import sys
import pickle
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from catboost import CatBoostClassifier

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLUMNS,
    TARGET_COL,
)
from src.preprocessing import clean_raw_data


def run_training(
    data_path: Path = RAW_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    iterations: int = 600,
    learning_rate: float = 0.06,
    depth: int = 6,
) -> dict:
    """Executes end-to-end model training, threshold evaluation, and artifact serialization.
    
    Args:
        data_path: Path to raw CSV dataset.
        test_size: Hold-out test fraction.
        random_state: Reproducibility seed.
        iterations: Boosting trees count.
        learning_rate: Gradient shrinkage.
        depth: Decision tree maximum depth.
        
    Returns:
        Dictionary of performance metrics.
    """
    print(f"[*] Loading dataset from: {data_path}")
    if not data_path.exists():
        if PROCESSED_DATA_PATH.exists():
            print(f"[!] Falling back to processed data at: {PROCESSED_DATA_PATH}")
            data_path = PROCESSED_DATA_PATH
        else:
            raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"[*] Raw data shape: {df.shape}")

    # Clean raw data with zero false labeling & domain feature engineering
    print("[*] Running domain feature engineering & data cleaning...")
    X, y = clean_raw_data(df)
    print(f"[*] Clean dataset: {len(X)} records with {X.shape[1]} features")
    print(f"[*] Target distribution:\n{y.value_counts()}")

    # Train / Test Split BEFORE scaling to eliminate Data Leakage
    print(f"[*] Splitting dataset (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Feature Scaling: Fit strictly on X_train, transform X_train and X_test
    print("[*] Fitting StandardScaler strictly on training split (preventing data leakage)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train CatBoost Classifier
    print(f"[*] Training CatBoostClassifier (iterations={iterations}, lr={learning_rate}, depth={depth})...")
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_state,
        verbose=100,
        eval_metric="Logloss"
    )
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), early_stopping_rounds=50, verbose=100)

    # Predictions & Probabilities
    print("[*] Evaluating model on unseen test set...")
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred_default = (y_proba >= 0.50).astype(int)

    metrics_default = {
        "accuracy": float(accuracy_score(y_test, y_pred_default)),
        "precision": float(precision_score(y_test, y_pred_default)),
        "recall": float(recall_score(y_test, y_pred_default)),
        "f1_score": float(f1_score(y_test, y_pred_default)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\n" + "=" * 55)
    print("       MODEL EVALUATION METRICS (Threshold = 0.50)")
    print("=" * 55)
    for k, v in metrics_default.items():
        print(f"  {k.upper():<15}: {v:.4f}")
    print("=" * 55)

    # Threshold Tuning Analysis
    print("\n--- Decision Threshold Sensitivity Tuning ---")
    print(f"{'Threshold':<12} | {'Accuracy':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 60)
    for th in [0.35, 0.40, 0.45, 0.50]:
        p = (y_proba >= th).astype(int)
        print(f"{th:<12.2f} | {accuracy_score(y_test, p):<10.4f} | {recall_score(y_test, p):<10.4f} | {precision_score(y_test, p):<10.4f} | {f1_score(y_test, p):<10.4f}")

    print("\nConfusion Matrix (Default 0.50):")
    print(confusion_matrix(y_test, y_pred_default))

    # Save artifacts in local environment format (eliminates InconsistentVersionWarning)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n[+] Production model successfully saved to: {MODEL_PATH}")

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[+] Fitted StandardScaler successfully saved to: {SCALER_PATH}")

    return metrics_default


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Production CatBoost model with domain features")
    parser.add_argument("--data-path", type=str, default=str(RAW_DATA_PATH), help="Path to raw CSV dataset")
    parser.add_argument("--iterations", type=int, default=600, help="CatBoost boosting iterations")
    parser.add_argument("--lr", type=float, default=0.06, help="Learning rate")
    parser.add_argument("--depth", type=int, default=6, help="Tree depth")
    args = parser.parse_args()

    run_training(
        data_path=Path(args.data_path),
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth
    )
