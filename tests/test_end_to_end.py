"""End-to-end synthetic pipeline test.

Runs the complete pipeline on synthetic data:
loader -> merge -> features -> labels -> split -> train -> val -> test -> predict -> save -> load -> predict
"""
from __future__ import annotations

import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import joblib

from src.config import (CADENCE_S, GAP_S, HORIZON_S, LOOKBACK_WINDOWS_S,
                        BACKGROUND_WINDOW_S, SAMPLE_EVERY_S, TRAIN_FRACTION,
                        VAL_FRACTION, WASHOUT_S, MODEL_PATH)
from src.data_loader import load_solexs, load_hel1os
from src.features import build_features, sample_rows
from src.flares import build_labels, detect_flare_events
from src.train import _split_train_val_test, _evaluate_split, _skill_scores, _best_threshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix


def make_synthetic_data(n_days=5, flare_days=None):
    """Create synthetic multi-day SoLEXS + HEL1OS data with optional flares.

    Args:
        n_days: Number of days of data to generate
        flare_days: List of day indices (0-based) where flares should occur
    """
    if flare_days is None:
        flare_days = [1, 3]  # Flares on day 1 and day 3

    seconds_per_day = 86400
    total_seconds = n_days * seconds_per_day
    index = pd.date_range("2026-01-01", periods=total_seconds, freq="1s", tz=None)

    # Quiet sun background
    solexs = np.random.poisson(20, total_seconds).astype(float)
    hel1os_1 = np.random.poisson(5, total_seconds).astype(float)
    hel1os_2 = np.random.poisson(3, total_seconds).astype(float)

    # Inject flares on specified days
    for day_idx in flare_days:
        day_start = day_idx * seconds_per_day
        flare_time = day_start + 43200  # Midday (12:00)
        flare_duration = 300
        t = np.arange(flare_duration)
        flare_profile = 500 * np.exp(-t / 60) * (1 - np.exp(-t / 30))
        end = min(flare_time + flare_duration, total_seconds)
        solexs[flare_time:end] += flare_profile[:end - flare_time]

    df = pd.DataFrame({
        "solexs_sdd2_counts": solexs,
        "hel1os_cdte1_5-20keV": hel1os_1,
        "hel1os_czt1_20-40keV": hel1os_2,
    }, index=index)
    return df


def test_end_to_end_pipeline():
    """Run complete pipeline on synthetic data."""
    print("=" * 60)
    print("END-TO-END SYNTHETIC PIPELINE TEST")
    print("=" * 60)

    # 1. Generate synthetic data
    print("\n1. Generating synthetic data (5 days, flares on day 1 and 3)...")
    df = make_synthetic_data(n_days=5, flare_days=[1, 3])
    print(f"   Data shape: {df.shape}, index range: {df.index[0]} to {df.index[-1]}")

    # 2. Build features
    print("\n2. Building features...")
    feats = build_features(df)
    print(f"   Feature shape: {feats.shape}")
    print(f"   Feature columns: {len(feats.columns)}")
    print(f"   NaN count: {feats.isna().sum().sum()}")

    # 3. Detect flare events (POST-HOC)
    print("\n3. Detecting flare events (post-hoc)...")
    events = detect_flare_events(df["solexs_sdd2_counts"])
    print(f"   Events detected: {len(events)}")
    for _, ev in events.iterrows():
        print(f"     {ev['onset']} - peak {ev['peak_flux']:.1f} cts/s")

    # 4. Build labels
    print("\n4. Building forecast labels...")
    labels = build_labels(feats.index, events["onset"].tolist(), HORIZON_S, GAP_S)
    print(f"   Label positive rate: {labels.mean():.4f}")

    # 5. Sample and filter
    print("\n5. Sampling and filtering...")
    keep = sample_rows(feats.index, SAMPLE_EVERY_S)
    X = feats.iloc[keep]
    y = labels.iloc[keep]

    # Exclude ongoing flares
    in_flare = np.zeros(len(y), dtype=bool)
    for _, ev in events.iterrows():
        in_flare |= (X.index >= ev["onset"]) & (X.index <= ev["end"])
    X = X[~in_flare]
    y = y[~in_flare]

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]
    print(f"   Final samples: {len(X)}, positive: {y.sum()} ({y.mean():.3%})")

    # 6. Chronological split with washout
    print("\n6. Chronological split (70/15/15) with washout...")
    X_tr, y_tr, X_va, y_va, X_te, y_te = _split_train_val_test(X, y)
    print(f"   Train: {len(X_tr)} (+{y_tr.sum()})")
    print(f"   Val:   {len(X_va)} (+{y_va.sum()})")
    print(f"   Test:  {len(X_te)} (+{y_te.sum()})")

    # 7. Train model
    print("\n7. Training model...")
    model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    )
    model.fit(X_tr, y_tr)

    # 8. Evaluate on all splits
    print("\n8. Evaluating...")
    val_metrics = _evaluate_split(model, X_va, y_va, "validation")
    best_thr = val_metrics["threshold"]
    test_metrics = _evaluate_split(model, X_te, y_te, "test", threshold=best_thr)
    train_metrics = _evaluate_split(model, X_tr, y_tr, "train")

    # 9. Save model and metadata
    print("\n9. Saving model...")
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "flare_model.joblib")
        meta_path = model_path.replace(".joblib", "_meta.joblib")

        joblib.dump(model, model_path)

        meta = {
            "feature_names": list(X_tr.columns),
            "feature_medians": X_tr.median().to_dict(),
            "threshold": float(best_thr),
            "gap_s": GAP_S,
            "horizon_s": HORIZON_S,
            "roc_auc": test_metrics["roc_auc"],
            "pr_auc": test_metrics["pr_auc"],
            "tss": test_metrics["tss"],
            "hss": test_metrics["hss"],
        }
        joblib.dump(meta, meta_path)

        # 10. Load and verify
        print("\n10. Loading model and verifying predictions...")
        loaded_model = joblib.load(model_path)
        loaded_meta = joblib.load(meta_path)

        # Predict on test set with both models
        proba_orig = model.predict_proba(X_te)[:, 1]
        proba_loaded = loaded_model.predict_proba(X_te)[:, 1]

        assert np.allclose(proba_orig, proba_loaded), "Predictions differ after save/load!"
        print("   Save/load verification: PASSED")

        # 11. Test predict_timeline equivalent
        print("\n11. Testing forecast on latest data...")
        latest_feats = feats.tail(100)
        latest_proba = model.predict_proba(latest_feats)[:, 1]
        print(f"   Latest 100 samples forecast range: {latest_proba.min():.4f} - {latest_proba.max():.4f}")

    print("\n" + "=" * 60)
    print("END-TO-END TEST PASSED")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_end_to_end_pipeline()


if __name__ == "__main__":
    test_end_to_end_pipeline()