import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
    confusion_matrix, average_precision_score, accuracy_score, f1_score
)

FEATURE_COLUMNS = [
    'solexs_counts', 'solexs_smooth', 'solexs_baseline', 'solexs_excess',
    'solexs_mean_5m', 'solexs_std_5m', 'solexs_rise_rate_15m',
    'hel1os_czt_total', 'hel1os_smooth', 'hel1os_baseline', 'hel1os_excess',
    'hel1os_mean_5m', 'hel1os_std_5m',
    'hardness_ratio', 'hardness_ratio_excess',
    'd_solexs_dt', 'd2_solexs_dt2', 'd_hel1os_dt',
    'hel1os_10_20', 'hel1os_20_40', 'hel1os_40_60', 'hel1os_60_80', 'hel1os_80_150',
    'solexs_gap', 'hel1os_gap'
]

def extract_features_and_targets(df, horizon_1h_steps=360, horizon_2h_steps=720, dt_sec=10):
    """
    Builds tabular feature matrix and forward-looking forecast labels.

    Labels are constructed from STRICTLY FUTURE data [t+1, t+horizon].
    The FixedForwardWindowIndexer includes the current row, so we shift(-1)
    to exclude time t from the label window.

    Returns:
        X: Feature DataFrame
        y_flare_1h: Binary 1h flare label
        y_flare_2h: Binary 2h flare label
        future_max_flux_1h: Regression target (max flux in next 1h)
        y_class_1h: Multiclass flare intensity label (0=Quiet, 1=B, 2=C, 3=M, 4=X)
        valid_mask: Boolean mask — True where full forecast horizon is available
    """
    df = df.copy()

    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURE_COLUMNS].copy()

    # Forward-looking labels: max excess/flux in the STRICTLY FUTURE window [t+1, t+horizon]
    # Step 1: FixedForwardWindowIndexer covers [t, t+window-1]
    # Step 2: shift(-1) moves to [t+1, t+window], excluding current timestep
    # Step 3: Rows near the end where the full horizon isn't available get NaN
    indexer_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_1h_steps)
    indexer_2h = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_2h_steps)

    future_max_excess_1h = df['solexs_excess'].rolling(window=indexer_1h, min_periods=1).max().shift(-1)
    future_max_excess_2h = df['solexs_excess'].rolling(window=indexer_2h, min_periods=1).max().shift(-1)
    future_max_flux_1h = df['solexs_counts'].rolling(window=indexer_1h, min_periods=1).max().shift(-1)

    # Mark rows where the full forecast horizon is unavailable (end of day)
    # These rows must be EXCLUDED from training (incomplete labels)
    valid_mask = future_max_excess_1h.notna() & future_max_excess_2h.notna()

    future_max_excess_1h = future_max_excess_1h.fillna(0)
    future_max_excess_2h = future_max_excess_2h.fillna(0)
    future_max_flux_1h = future_max_flux_1h.fillna(0)

    # Flare occurrence label: active flare (excess >= 15.0) — matches physics engine B-class threshold
    y_flare_1h = (future_max_excess_1h >= 15.0).astype(int)
    y_flare_2h = (future_max_excess_2h >= 15.0).astype(int)

    # Multiclass target aligned with physics engine classify_flare_intensity:
    # 0: Quiet/Micro (<15), 1: B-Class (15-40), 2: C-Class (40-150), 3: M-Class (150-800), 4: X-Class (>=800)
    y_class_1h = np.zeros(len(df), dtype=int)
    y_class_1h[future_max_flux_1h >= 15.0] = 1   # B-Class
    y_class_1h[future_max_flux_1h >= 40.0] = 2   # C-Class
    y_class_1h[future_max_flux_1h >= 150.0] = 3  # M-Class
    y_class_1h[future_max_flux_1h >= 800.0] = 4  # X-Class

    return X, y_flare_1h, y_flare_2h, future_max_flux_1h, y_class_1h, valid_mask


def compute_space_weather_metrics(y_true, y_prob, threshold=0.50):
    """
    Computes the full suite of operational space weather forecast metrics.

    Returns a dictionary containing: ROC-AUC, PR-AUC, TSS, HSS, Brier, FAR,
    and the confusion matrix components (TP, TN, FP, FN).
    """
    y_pred = (y_prob >= threshold).astype(int)

    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    TSS = TPR - FPR

    HSS_num = 2 * (TP * TN - FP * FN)
    HSS_den = (TP + FN) * (FN + TN) + (TP + FP) * (FP + TN)
    HSS = HSS_num / HSS_den if HSS_den > 0 else 0.0

    FAR = FP / (TP + FP) if (TP + FP) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0

    brier = brier_score_loss(y_true, y_prob)

    return {
        'ROC-AUC': round(float(roc_auc), 4),
        'PR-AUC': round(float(pr_auc), 4),
        'TSS': round(float(TSS), 4),
        'HSS': round(float(HSS), 4),
        'Brier': round(float(brier), 4),
        'FAR': round(float(FAR), 4),
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'TPR (POD)': round(float(TPR), 4),
        'FPR': round(float(FPR), 4),
    }


def compute_event_level_metrics(y_true, y_prob, timestamps, threshold=0.50, min_event_gap_min=10):
    """
    Computes event-level metrics for solar flare forecasting.
    Groups consecutive positive predictions into events and matches with ground truth events.

    Returns dict with: event_precision, event_recall, event_f1, mean_lead_time, false_alarm_events
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Find predicted events (contiguous runs of 1s)
    pred_events = []
    in_event = False
    start_idx = 0
    for i, val in enumerate(y_pred):
        if val == 1 and not in_event:
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            in_event = False
            pred_events.append((start_idx, i - 1))
    if in_event:
        pred_events.append((start_idx, len(y_pred) - 1))

    # Find true events
    true_events = []
    in_event = False
    start_idx = 0
    for i, val in enumerate(y_true):
        if val == 1 and not in_event:
            in_event = True
            start_idx = i
        elif val == 0 and in_event:
            in_event = False
            true_events.append((start_idx, i - 1))
    if in_event:
        true_events.append((start_idx, len(y_true) - 1))

    # Match predicted events to true events (within min_event_gap_min * 6 steps at 10s)
    max_gap_steps = min_event_gap_min * 6
    matched_true = set()
    matched_pred = set()
    lead_times = []

    for p_idx, (p_start, p_end) in enumerate(pred_events):
        best_match = None
        best_overlap = 0
        for t_idx, (t_start, t_end) in enumerate(true_events):
            if t_idx in matched_true:
                continue
            # Check overlap or proximity
            overlap = max(0, min(p_end, t_end) - max(p_start, t_start) + 1)
            proximity = max(p_start - t_end, t_start - p_end)
            if overlap > 0 or proximity <= max_gap_steps:
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = t_idx
        if best_match is not None:
            matched_true.add(best_match)
            matched_pred.add(p_idx)
            # Lead time: first prediction to first true positive
            t_start = true_events[best_match][0]
            lead_time_steps = max(0, t_start - p_start)
            lead_times.append(lead_time_steps * 10 / 60)  # minutes

    event_precision = len(matched_pred) / len(pred_events) if pred_events else 0.0
    event_recall = len(matched_true) / len(true_events) if true_events else 0.0
    event_f1 = 2 * event_precision * event_recall / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0.0
    mean_lead_time = np.mean(lead_times) if lead_times else 0.0
    false_alarm_events = len(pred_events) - len(matched_pred)
    missed_events = len(true_events) - len(matched_true)

    return {
        'event_precision': round(float(event_precision), 4),
        'event_recall': round(float(event_recall), 4),
        'event_f1': round(float(event_f1), 4),
        'mean_lead_time_min': round(float(mean_lead_time), 2),
        'false_alarm_events': int(false_alarm_events),
        'missed_events': int(missed_events),
        'predicted_events': len(pred_events),
        'true_events': len(true_events)
    }


class SolarFlareAI:
    def __init__(self):
        # Fast, state-of-the-art histogram gradient boosted classifiers
        # HistGradientBoosting handles raw features natively — no scaling needed
        self.clf_1h = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight='balanced',
            random_state=42
        )
        self.clf_2h = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight='balanced',
            random_state=42
        )
        # Multiclass flare intensity classifier (0=Quiet, 1=B, 2=C, 3=M, 4=X)
        self.clf_class = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            class_weight='balanced',
            random_state=42
        )
        self.reg_peak_flux = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            random_state=42
        )
        self.is_trained = False
        self.metrics = {}

    def fit(self, X, y_1h, y_2h, y_flux, y_class):
        """Fits the AI models on historical feature matrices."""
        print("Fitting AI Flare Forecast Models...")
        X_clean = X.fillna(0.0).values

        # Fit 1-Hour Flare Predictor
        self.clf_1h.fit(X_clean, y_1h)

        # Fit 2-Hour Flare Predictor
        self.clf_2h.fit(X_clean, y_2h)

        # Fit Multiclass Flare Intensity Classifier (B/C/M/X)
        self.clf_class.fit(X_clean, y_class)

        # Fit Expected Peak Flux Regressor
        self.reg_peak_flux.fit(X_clean, y_flux)

        self.is_trained = True

        # Compute training skill scores
        prob_1h = self.clf_1h.predict_proba(X_clean)[:, 1]
        train_metrics = compute_space_weather_metrics(y_1h, prob_1h)

        # Multiclass metrics
        class_pred = self.clf_class.predict(X_clean)
        class_probs = self.clf_class.predict_proba(X_clean)
        from sklearn.metrics import accuracy_score, f1_score
        class_acc = accuracy_score(y_class, class_pred)
        class_f1_macro = f1_score(y_class, class_pred, average='macro')

        self.metrics = {
            'train_samples': len(X),
            'train_positive_1h': int(np.sum(y_1h)),
            'train_class_accuracy': round(float(class_acc), 4),
            'train_class_f1_macro': round(float(class_f1_macro), 4),
            **{f'train_{k}': v for k, v in train_metrics.items()}
        }
        print(f"Training Complete! Samples: {len(X)} | Positive 1h events: {np.sum(y_1h)} | ROC-AUC: {train_metrics['ROC-AUC']:.3f} | Class Acc: {class_acc:.3f} | Class F1-macro: {class_f1_macro:.3f}")

    def evaluate(self, X_test, y_1h_test, y_2h_test, y_flux_test, y_class_test, label="TEST"):
        """
        Evaluates the trained model on a held-out chronological test set.
        Reports full space weather metrics: ROC-AUC, PR-AUC, TSS, HSS, Brier, FAR.
        Also evaluates multiclass flare intensity classification and event-level metrics.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not yet trained.")

        X_clean = X_test.fillna(0.0).values

        prob_1h = self.clf_1h.predict_proba(X_clean)[:, 1]
        prob_2h = self.clf_2h.predict_proba(X_clean)[:, 1]
        class_probs = self.clf_class.predict_proba(X_clean)
        class_pred = self.clf_class.predict(X_clean)

        metrics_1h = compute_space_weather_metrics(y_1h_test, prob_1h)
        metrics_2h = compute_space_weather_metrics(y_2h_test, prob_2h)

        # Multiclass evaluation
        class_acc = accuracy_score(y_class_test, class_pred)
        class_f1_macro = f1_score(y_class_test, class_pred, average='macro')
        class_f1_weighted = f1_score(y_class_test, class_pred, average='weighted')

        # Per-class ROC-AUC (one-vs-rest)
        class_roc_auc = {}
        try:
            for i in range(class_probs.shape[1]):
                if len(np.unique(y_class_test)) > 1:
                    class_roc_auc[i] = round(float(roc_auc_score((y_class_test == i).astype(int), class_probs[:, i])), 4)
        except ValueError:
            pass

        # Event-level metrics (requires timestamps - use index as proxy if not available)
        event_metrics_1h = compute_event_level_metrics(y_1h_test, prob_1h, np.arange(len(y_1h_test)))
        event_metrics_2h = compute_event_level_metrics(y_2h_test, prob_2h, np.arange(len(y_2h_test)))

        print(f"\n=== {label} SET EVALUATION (1-Hour Horizon) ===")
        print(f"  Samples: {len(X_test):,} | Positives: {np.sum(y_1h_test):,} ({np.mean(y_1h_test)*100:.1f}%)")
        for k, v in metrics_1h.items():
            print(f"  {k}: {v}")

        print(f"\n=== {label} SET EVALUATION (2-Hour Horizon) ===")
        print(f"  Samples: {len(X_test):,} | Positives: {np.sum(y_2h_test):,} ({np.mean(y_2h_test)*100:.1f}%)")
        for k, v in metrics_2h.items():
            print(f"  {k}: {v}")

        print(f"\n=== {label} SET EVALUATION (Multiclass Flare Intensity) ===")
        print(f"  Samples: {len(X_test):,} | Classes: {np.unique(y_class_test)}")
        print(f"  Accuracy: {class_acc:.4f} | F1-macro: {class_f1_macro:.4f} | F1-weighted: {class_f1_weighted:.4f}")
        for cls_idx, auc in class_roc_auc.items():
            class_name = {0: 'Quiet', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}.get(cls_idx, str(cls_idx))
            print(f"  ROC-AUC ({class_name}): {auc}")

        print(f"\n=== {label} SET EVALUATION (Event-Level 1-Hour) ===")
        for k, v in event_metrics_1h.items():
            print(f"  {k}: {v}")

        print(f"\n=== {label} SET EVALUATION (Event-Level 2-Hour) ===")
        for k, v in event_metrics_2h.items():
            print(f"  {k}: {v}")

        self.metrics[f'{label.lower()}_1h'] = metrics_1h
        self.metrics[f'{label.lower()}_2h'] = metrics_2h
        self.metrics[f'{label.lower()}_class'] = {
            'accuracy': round(float(class_acc), 4),
            'f1_macro': round(float(class_f1_macro), 4),
            'f1_weighted': round(float(class_f1_weighted), 4),
            'roc_auc_per_class': class_roc_auc
        }
        self.metrics[f'{label.lower()}_event_1h'] = event_metrics_1h
        self.metrics[f'{label.lower()}_event_2h'] = event_metrics_2h
        return metrics_1h, metrics_2h, self.metrics[f'{label.lower()}_class'], event_metrics_1h, event_metrics_2h

    def predict_timeline(self, df):
        """
        Generates continuous real-time nowcast & forecast predictions across a 24-hour timeline.
        All smoothing is strictly causal (trailing window, no centered smoothing).
        """
        if not self.is_trained:
            raise RuntimeError("Model is not yet trained. Call fit() or load_model() first.")

        X = df[FEATURE_COLUMNS].fillna(0.0).values

        # Forecast probabilities
        prob_1h = self.clf_1h.predict_proba(X)[:, 1]
        prob_2h = self.clf_2h.predict_proba(X)[:, 1]
        class_probs = self.clf_class.predict_proba(X)
        class_pred = self.clf_class.predict(X)
        pred_flux = np.maximum(df['solexs_counts'].values, self.reg_peak_flux.predict(X))

        # Causal trailing-window smoothing (no future data, center=False)
        prob_1h_smooth = pd.Series(prob_1h).rolling(6, min_periods=1, center=False).mean().values
        prob_2h_smooth = pd.Series(prob_2h).rolling(6, min_periods=1, center=False).mean().values

        # Risk level string
        risk_levels = []
        for p in prob_1h_smooth:
            if p >= 0.75:
                risk_levels.append('CRITICAL / SEVERE')
            elif p >= 0.50:
                risk_levels.append('HIGH')
            elif p >= 0.25:
                risk_levels.append('ELEVATED')
            elif p >= 0.10:
                risk_levels.append('GUARDED')
            else:
                risk_levels.append('LOW')

        # Class name mapping
        class_names = {0: 'Quiet', 1: 'B', 2: 'C', 3: 'M', 4: 'X'}

        df_out = df.copy()
        df_out['prob_flare_1h'] = np.clip(prob_1h_smooth * 100.0, 0.0, 100.0)
        df_out['prob_flare_2h'] = np.clip(prob_2h_smooth * 100.0, 0.0, 100.0)
        df_out['forecast_expected_flux_1h'] = pred_flux
        df_out['forecast_risk_level'] = risk_levels
        df_out['forecast_class'] = [class_names.get(c, str(c)) for c in class_pred]
        for i, name in class_names.items():
            if i < class_probs.shape[1]:
                df_out[f'prob_class_{name}'] = np.clip(class_probs[:, i] * 100.0, 0.0, 100.0)

        return df_out

    def save(self, filepath):
        """Serializes the trained AI model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'clf_1h': self.clf_1h,
            'clf_2h': self.clf_2h,
            'clf_class': self.clf_class,
            'reg_peak_flux': self.reg_peak_flux,
            'is_trained': self.is_trained,
            'metrics': self.metrics
        }, filepath)
        print(f"Model successfully saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads a serialized trained model."""
        data = joblib.load(filepath)
        instance = cls()
        instance.clf_1h = data['clf_1h']
        instance.clf_2h = data['clf_2h']
        instance.clf_class = data['clf_class']
        instance.reg_peak_flux = data['reg_peak_flux']
        instance.is_trained = data['is_trained']
        instance.metrics = data.get('metrics', {})
        return instance
