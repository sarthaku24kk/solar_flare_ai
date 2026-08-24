"""Train + evaluate the flare-forecast classifier (scikit-learn).

Metrics include ROC-AUC, PR-AUC, True Skill Statistic (TSS) and Heidke
Skill Score (HSS). The time-ordered split has washout buffers so no
rolling-window features can leak across train/val/test boundaries.

Split: 70% train | 15% validation | 15% test (chronological)
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix,
                              precision_recall_curve, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (GAP_S, HORIZON_S, MODEL_PATH, SAMPLE_EVERY_S,
                        TEST_PLOTS, TRAIN_FRACTION, VAL_FRACTION, WASHOUT_S,
                        CADENCE_S, LOOKBACK_WINDOWS_S, BACKGROUND_WINDOW_S)
from src.data_loader import load
from src.features import build_features, sample_rows, _GAP_FFILL_LIMIT
from src.flares import build_labels, detect_flare_events


def _skill_scores(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    tss = tp / (tp + fn) - fp / (fp + tn) if (tp + fn) and (fp + tn) else 0.0
    pcs = (tp + tn) / (tp + fp + fn + tn)
    expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (tp + fp + fn + tn) ** 2
    hss = (pcs - expected) / (1 - expected) if expected != 1 else 0.0
    return tss, hss


def _best_threshold(y, proba):
    prec, rec, th = precision_recall_curve(y, proba)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    i = int(np.argmax(f1))
    return th[i] if i < len(th) else 0.5


def _split_train_val_test(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Chronological 70/15/15 split with washout buffers between splits."""
    n = len(X)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    # n_test = n - n_train - n_val (remainder)

    washout_ns = WASHOUT_S * 1_000_000_000

    # Train: [0, n_train)
    train_end_idx = n_train
    train_end_time = X.index[train_end_idx - 1]

    # Validation starts after washout from train end
    val_start_time = train_end_time + pd.Timedelta(seconds=WASHOUT_S)
    val_start_idx = X.index.searchsorted(val_start_time, side="left")

    # Validation ends before next washout
    val_end_idx = min(val_start_idx + n_val, n)
    val_end_time = X.index[val_end_idx - 1]

    # Test starts after washout from val end
    test_start_time = val_end_time + pd.Timedelta(seconds=WASHOUT_S)
    test_start_idx = X.index.searchsorted(test_start_time, side="left")

    train_mask = np.zeros(n, dtype=bool)
    train_mask[:train_end_idx] = True

    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_start_idx:val_end_idx] = True

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_start_idx:] = True

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    print(f"train n={len(X_tr)} (+{y_tr.sum()}) | val n={len(X_va)} (+{y_va.sum()}) | test n={len(X_te)} (+{y_te.sum()})")
    print(f"washout buffer: {WASHOUT_S}s between each split")
    print(f"train period: {X_tr.index[0]} -> {X_tr.index[-1]}")
    print(f"val period:   {X_va.index[0]} -> {X_va.index[-1]}")
    print(f"test period:  {X_te.index[0]} -> {X_te.index[-1]}")

    return X_tr, y_tr, X_va, y_va, X_te, y_te


def _evaluate_split(model, X, y, split_name, threshold=None):
    """Evaluate model on a split, return metrics dict.

    Handles edge cases where a split has only one class.
    """
    proba = model.predict_proba(X)[:, 1]
    pos_rate = float(y.mean())

    if threshold is None:
        threshold = _best_threshold(y, proba)
    pred = (proba >= threshold).astype(int)
    tss, hss = _skill_scores(y, pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    # Compute metrics safely (handle single-class cases)
    try:
        roc_auc = float(roc_auc_score(y, proba))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y, proba))
    except ValueError:
        pr_auc = float("nan")

    precision = float(tp/(tp+fp)) if (tp+fp) else 0.0
    recall = float(tp/(tp+fn)) if (tp+fn) else 0.0

    print(f"\n=== {split_name.upper()} (threshold={threshold:.3f}) ===")
    print(f"positive rate: {pos_rate:.3%}")
    print(f"ROC-AUC: {roc_auc:.3f}" if not np.isnan(roc_auc) else "ROC-AUC: N/A (single class)")
    print(f"PR-AUC:  {pr_auc:.3f}" if not np.isnan(pr_auc) else "PR-AUC:  N/A (single class)")
    print(f"TSS: {tss:.3f}   HSS: {hss:.3f}")
    print(f"precision: {precision:.3f}  recall: {recall:.3f}" if (tp+fn) else f"precision: {precision:.3f}  recall: N/A")
    print(f"confusion matrix (rows=true, cols=pred):\n[[{tn} {fp}]\n [{fn} {tp}]]")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tss": float(tss),
        "hss": float(hss),
        "threshold": float(threshold),
        "positive_rate": pos_rate,
        "precision": precision,
        "recall": recall,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def make_dataset() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    events = detect_flare_events(df["solexs_sdd2_counts"])
    feats = build_features(df)
    idx = feats.index
    labels = build_labels(idx, events["onset"], HORIZON_S, GAP_S)
    keep = sample_rows(idx, SAMPLE_EVERY_S)
    X = feats.iloc[keep]
    y = labels.iloc[keep]

    # exclude samples during an ongoing flare
    in_flare = np.zeros(len(y), dtype=bool)
    for _, ev in events.iterrows():
        in_flare |= (X.index >= ev["onset"]) & (X.index <= ev["end"])
    X = X[~in_flare]
    y = y[~in_flare]

    mask = X.notna().all(axis=1)
    return X[mask], y[mask]


def main() -> None:
    X, y = make_dataset()
    print(f"samples={len(X)}  positive={y.sum()}  ({y.mean():.3%})")

    # Chronological 70/15/15 split with washout
    X_tr, y_tr, X_va, y_va, X_te, y_te = _split_train_val_test(X, y)

    model = make_pipeline(
        StandardScaler(),
        GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
    )
    model.fit(X_tr, y_tr)

    # Evaluate on validation (tune threshold)
    val_metrics = _evaluate_split(model, X_va, y_va, "validation")
    best_thr = val_metrics["threshold"]

    # Evaluate on test with validation-chosen threshold
    test_metrics = _evaluate_split(model, X_te, y_te, "test", threshold=best_thr)

    # Also evaluate on train for reference
    train_metrics = _evaluate_split(model, X_tr, y_tr, "train")

    imp = pd.Series(model.named_steps["gradientboostingclassifier"].feature_importances_,
                    index=X_tr.columns).sort_values(ascending=False)
    print("\ntop-15 features:\n", imp.head(15).to_string())

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # save metadata needed for inference on new readings
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
        "val_roc_auc": val_metrics["roc_auc"],
        "val_pr_auc": val_metrics["pr_auc"],
        "val_tss": val_metrics["tss"],
        "val_hss": val_metrics["hss"],
    }
    joblib.dump(meta, MODEL_PATH.replace(".joblib", "_meta.joblib"))
    print(f"\nsaved model -> {MODEL_PATH} (+ _meta.joblib)")

    # Save comprehensive training summary
    summary = {
        "total_days": int((X.index[-1] - X.index[0]).total_seconds() / 86400) + 1,
        "split": {
            "train_days": int((X_tr.index[-1] - X_tr.index[0]).total_seconds() / 86400) + 1,
            "val_days": int((X_va.index[-1] - X_va.index[0]).total_seconds() / 86400) + 1 if len(X_va) > 0 else 0,
            "test_days": int((X_te.index[-1] - X_te.index[0]).total_seconds() / 86400) + 1 if len(X_te) > 0 else 0,
            "train_samples": int(len(X_tr)),
            "val_samples": int(len(X_va)),
            "test_samples": int(len(X_te)),
        },
        "flare_counts": {
            "train": int(y_tr.sum()),
            "val": int(y_va.sum()),
            "test": int(y_te.sum()),
        },
        "positive_rates": {
            "train": float(y_tr.mean()),
            "val": float(y_va.mean()) if len(y_va) > 0 else 0.0,
            "test": float(y_te.mean()) if len(y_te) > 0 else 0.0,
        },
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        "features": list(X_tr.columns),
        "causal_pipeline": True,
        "leakage_free": True,
        "configuration": {
            "resample_frequency": f"{CADENCE_S}s",
            "gap_ffill_limit_steps": _GAP_FFILL_LIMIT,
            "gap_ffill_limit_seconds": _GAP_FFILL_LIMIT * CADENCE_S,
            "forecast_horizon_1h_steps": HORIZON_S // CADENCE_S,
            "forecast_horizon_2h_steps": HORIZON_S // CADENCE_S,
            "washout_seconds": WASHOUT_S,
            "background_window_seconds": BACKGROUND_WINDOW_S,
        },
    }
    summary_path = os.path.join(os.path.dirname(MODEL_PATH), "training_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    import json
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"training summary -> {summary_path}")

    os.makedirs(TEST_PLOTS, exist_ok=True)
    pd.DataFrame({
        "time": X_te.index,
        "y_true": y_te.values,
        "score": model.predict_proba(X_te)[:, 1],
    }).to_csv(os.path.join(TEST_PLOTS, "test_scores.csv"), index=False)
    print("test scores ->", os.path.join(TEST_PLOTS, "test_scores.csv"))


if __name__ == "__main__":
    main()
