import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import classification_report, roc_auc_score, brier_score_loss, confusion_matrix
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    'solexs_counts', 'solexs_smooth', 'solexs_baseline', 'solexs_excess',
    'solexs_mean_5m', 'solexs_std_5m', 'solexs_rise_rate_15m',
    'hel1os_czt_total', 'hel1os_smooth', 'hel1os_baseline', 'hel1os_excess',
    'hel1os_mean_5m', 'hel1os_std_5m',
    'hardness_ratio', 'hardness_ratio_excess',
    'd_solexs_dt', 'd2_solexs_dt2', 'd_hel1os_dt',
    'hel1os_10_20', 'hel1os_20_40', 'hel1os_40_60', 'hel1os_60_80', 'hel1os_80_150'
]

def extract_features_and_targets(df, horizon_1h_steps=360, horizon_2h_steps=720, dt_sec=10):
    """
    Builds tabular feature matrix and forward-looking forecast labels.
    - horizon_1h_steps: 360 steps of 10s = 60 mins (1 hour)
    - horizon_2h_steps: 720 steps of 10s = 120 mins (2 hours)
    """
    df = df.copy()
    
    # Ensure all feature columns exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
            
    X = df[FEATURE_COLUMNS].copy()
    
    # Forward rolling maximum of solexs_excess to define future flare occurrence
    # We shift backward so at time t, forward_max looks at [t+1 to t+horizon]
    indexer_1h = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_1h_steps)
    indexer_2h = pd.api.indexers.FixedForwardWindowIndexer(window_size=horizon_2h_steps)
    
    future_max_excess_1h = df['solexs_excess'].rolling(window=indexer_1h, min_periods=1).max().shift(-1).fillna(0)
    future_max_excess_2h = df['solexs_excess'].rolling(window=indexer_2h, min_periods=1).max().shift(-1).fillna(0)
    
    future_max_flux_1h = df['solexs_counts'].rolling(window=indexer_1h, min_periods=1).max().shift(-1).fillna(0)
    
    # Flare occurrence label: active flare (excess >= 15.0 or peak >= 30.0)
    y_flare_1h = (future_max_excess_1h >= 15.0).astype(int)
    y_flare_2h = (future_max_excess_2h >= 15.0).astype(int)
    
    # Multiclass target: 0: Quiet/B, 1: C-Class (>=40), 2: M/X-Class (>=150)
    y_class_1h = np.zeros(len(df), dtype=int)
    y_class_1h[future_max_flux_1h >= 30.0] = 1 # B/C class
    y_class_1h[future_max_flux_1h >= 75.0] = 2 # Moderate/C-high
    y_class_1h[future_max_flux_1h >= 150.0] = 3 # M/X class
    
    return X, y_flare_1h, y_flare_2h, future_max_flux_1h, y_class_1h

class SolarFlareAI:
    def __init__(self):
        self.scaler = StandardScaler()
        # Fast, state-of-the-art histogram gradient boosted classifiers
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
        self.reg_peak_flux = HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.08,
            random_state=42
        )
        self.is_trained = False
        self.metrics = {}

    def fit(self, X, y_1h, y_2h, y_flux):
        """Fits the AI models on historical feature matrices."""
        print("Fitting AI Flare Forecast Models...")
        X_clean = X.fillna(0.0).values
        
        # Fit 1-Hour Flare Predictor
        self.clf_1h.fit(X_clean, y_1h)
        
        # Fit 2-Hour Flare Predictor
        self.clf_2h.fit(X_clean, y_2h)
        
        # Fit Expected Peak Flux Regressor
        self.reg_peak_flux.fit(X_clean, y_flux)
        
        self.is_trained = True
        
        # Compute training skill scores
        prob_1h = self.clf_1h.predict_proba(X_clean)[:, 1]
        try:
            auc = roc_auc_score(y_1h, prob_1h)
        except Exception:
            auc = 0.5
        
        self.metrics = {
            'train_samples': len(X),
            'train_positive_1h': int(np.sum(y_1h)),
            'roc_auc_1h': round(float(auc), 4)
        }
        print(f"Training Complete! Samples: {len(X)} | Positive 1h events: {np.sum(y_1h)} | ROC-AUC: {auc:.3f}")

    def predict_timeline(self, df):
        """
        Generates continuous real-time nowcast & forecast predictions across a 24-hour timeline.
        """
        if not self.is_trained:
            raise RuntimeError("Model is not yet trained. Call fit() or load_model() first.")
        
        X = df[FEATURE_COLUMNS].fillna(0.0).values
        
        # Forecast probabilities
        prob_1h = self.clf_1h.predict_proba(X)[:, 1]
        prob_2h = self.clf_2h.predict_proba(X)[:, 1]
        pred_flux = np.maximum(df['solexs_counts'].values, self.reg_peak_flux.predict(X))
        
        # Smooth forecast probabilities slightly for clean visualization
        prob_1h_smooth = pd.Series(prob_1h).rolling(6, min_periods=1, center=True).mean().values
        prob_2h_smooth = pd.Series(prob_2h).rolling(6, min_periods=1, center=True).mean().values
        
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
                
        df_out = df.copy()
        df_out['prob_flare_1h'] = np.clip(prob_1h_smooth * 100.0, 0.0, 100.0)
        df_out['prob_flare_2h'] = np.clip(prob_2h_smooth * 100.0, 0.0, 100.0)
        df_out['forecast_expected_flux_1h'] = pred_flux
        df_out['forecast_risk_level'] = risk_levels
        
        return df_out

    def save(self, filepath):
        """Serializes the trained AI model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'clf_1h': self.clf_1h,
            'clf_2h': self.clf_2h,
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
        instance.reg_peak_flux = data['reg_peak_flux']
        instance.is_trained = data['is_trained']
        instance.metrics = data.get('metrics', {})
        return instance
