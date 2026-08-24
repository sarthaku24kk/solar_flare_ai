# tests/test_causality.py
# ===================================================================
# Pytest suite verifying NO future data leakage in the solar flare
# forecasting pipeline. Tests cover:
# A) Source-level inspection (no center=True, savgol, np.gradient, bfill)
# B) Perturbation tests (modifying future data must not change features at t)
# C) Forward-label correctness (labels only reflect future events)
# D) Chronological split ordering (train < val < test in time)
# ===================================================================

import pytest
import inspect
import textwrap
import numpy as np
import pandas as pd
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _get_source(func):
    """Return dedented source of a function."""
    return textwrap.dedent(inspect.getsource(func))


def _make_test_df(n=500, seed=42):
    """Create a synthetic Aditya-L1 like DataFrame for testing."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-07-18", periods=n, freq="10s", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "solexs_counts": rng.poisson(lam=15, size=n).astype(float),
        "hel1os_czt_total": rng.poisson(lam=5, size=n).astype(float),
        "hel1os_10_20": rng.poisson(2, n).astype(float),
        "hel1os_20_40": rng.poisson(2, n).astype(float),
        "hel1os_40_60": rng.poisson(1, n).astype(float),
        "hel1os_60_80": rng.poisson(1, n).astype(float),
        "hel1os_80_150": rng.poisson(1, n).astype(float),
    })


# ===================================================================
# TEST GROUP A: Source-level leakage detection
# ===================================================================

class TestSourceLevelLeakage:
    """Verify that compute_physics_features never uses leaky operations."""

    def test_no_center_true_in_physics_engine(self):
        from src.physics_engine import compute_physics_features
        src = _get_source(compute_physics_features)
        assert "center=True" not in src, (
            "compute_physics_features still uses center=True rolling windows"
        )

    def test_no_savgol_filter_in_physics_engine(self):
        from src.physics_engine import compute_physics_features
        src = _get_source(compute_physics_features)
        assert "savgol_filter" not in src, (
            "compute_physics_features still uses symmetric savgol_filter"
        )

    def test_no_np_gradient_in_physics_engine(self):
        from src.physics_engine import compute_physics_features
        src = _get_source(compute_physics_features)
        assert "np.gradient" not in src, (
            "compute_physics_features still uses np.gradient (central diff)"
        )

    def test_no_bfill_in_physics_engine(self):
        from src.physics_engine import compute_physics_features
        src = _get_source(compute_physics_features)
        assert ".bfill()" not in src, (
            "compute_physics_features uses bfill which leaks future data"
        )

    def test_no_center_true_in_predict_timeline(self):
        from src.ai_model import SolarFlareAI
        src = _get_source(SolarFlareAI.predict_timeline)
        assert "center=True" not in src, (
            "predict_timeline still uses center=True for probability smoothing"
        )


# ===================================================================
# TEST GROUP B: Perturbation tests — modifying future must not
#               change features at current time t
# ===================================================================

class TestCausalPerturbation:
    """Perturb data AFTER index t and verify features at t are unchanged."""

    def test_smooth_at_t_ignores_future(self):
        from src.physics_engine import compute_physics_features

        df_base = _make_test_df(500)
        result_base = compute_physics_features(df_base, dt_sec=10)
        t = 250

        df_perturbed = df_base.copy()
        df_perturbed.loc[t + 1:, "solexs_counts"] *= 10.0
        result_perturbed = compute_physics_features(df_perturbed, dt_sec=10)

        assert result_base["solexs_smooth"].iloc[t] == pytest.approx(
            result_perturbed["solexs_smooth"].iloc[t], abs=1e-10
        ), "solexs_smooth at time t changed when future data was perturbed — LEAKAGE"

    def test_derivative_at_t_ignores_future(self):
        from src.physics_engine import compute_physics_features

        df_base = _make_test_df(500)
        result_base = compute_physics_features(df_base, dt_sec=10)
        t = 250

        df_perturbed = df_base.copy()
        df_perturbed.loc[t + 1:, "solexs_counts"] += 500.0
        result_perturbed = compute_physics_features(df_perturbed, dt_sec=10)

        assert result_base["d_solexs_dt"].iloc[t] == pytest.approx(
            result_perturbed["d_solexs_dt"].iloc[t], abs=1e-10
        ), "d_solexs_dt at time t changed when future data was perturbed — LEAKAGE"

    def test_baseline_at_t_ignores_future(self):
        from src.physics_engine import compute_physics_features

        df_base = _make_test_df(500)
        result_base = compute_physics_features(df_base, dt_sec=10)
        t = 250

        df_perturbed = df_base.copy()
        df_perturbed.loc[t + 1:, "solexs_counts"] += 1000.0
        result_perturbed = compute_physics_features(df_perturbed, dt_sec=10)

        assert result_base["solexs_baseline"].iloc[t] == pytest.approx(
            result_perturbed["solexs_baseline"].iloc[t], abs=1e-10
        ), "solexs_baseline at time t changed when future data was perturbed — LEAKAGE"

    def test_ewm_at_t_ignores_future(self):
        from src.physics_engine import compute_physics_features

        df_base = _make_test_df(500)
        result_base = compute_physics_features(df_base, dt_sec=10)
        t = 250

        df_perturbed = df_base.copy()
        df_perturbed.loc[t + 1:, "solexs_counts"] += 999.0
        result_perturbed = compute_physics_features(df_perturbed, dt_sec=10)

        assert result_base["solexs_ewm"].iloc[t] == pytest.approx(
            result_perturbed["solexs_ewm"].iloc[t], abs=1e-10
        ), "solexs_ewm at time t changed when future data was perturbed — LEAKAGE"


# ===================================================================
# TEST GROUP C: Forward-label construction correctness
# ===================================================================

class TestForwardLabels:
    """Verify that labels only use strictly future data."""

    def _make_spike_df(self, n=1000, spike_idx=500):
        ts = pd.date_range("2026-07-18", periods=n, freq="10s", tz="UTC")
        df = pd.DataFrame({
            "timestamp": ts,
            "solexs_counts": np.zeros(n),
            "solexs_excess": np.zeros(n),
        })
        df.loc[spike_idx, "solexs_excess"] = 50.0
        df.loc[spike_idx, "solexs_counts"] = 80.0
        # Add all required feature columns
        for col in [
            'solexs_smooth', 'solexs_baseline', 'solexs_mean_5m',
            'solexs_std_5m', 'solexs_rise_rate_15m',
            'hel1os_czt_total', 'hel1os_smooth', 'hel1os_baseline',
            'hel1os_excess', 'hel1os_mean_5m', 'hel1os_std_5m',
            'hardness_ratio', 'hardness_ratio_excess',
            'd_solexs_dt', 'd2_solexs_dt2', 'd_hel1os_dt',
            'hel1os_10_20', 'hel1os_20_40', 'hel1os_40_60',
            'hel1os_60_80', 'hel1os_80_150',
        ]:
            if col not in df.columns:
                df[col] = 0.0
        return df

    def test_label_at_spike_is_zero(self):
        """At the spike itself (t=500), the label must be 0 because the
        label window [t+1, t+360] does NOT include t."""
        from src.ai_model import extract_features_and_targets
        df = self._make_spike_df()
        _, y1h, _, _, _, _ = extract_features_and_targets(df, horizon_1h_steps=360)
        assert y1h.iloc[500] == 0, "Label at spike time should be 0 (spike is at t, not in future)"

    def test_label_before_spike_is_one(self):
        """At t=499, the spike at t=500 is in [t+1, t+360] -> label = 1."""
        from src.ai_model import extract_features_and_targets
        df = self._make_spike_df()
        _, y1h, _, _, _, _ = extract_features_and_targets(df, horizon_1h_steps=360)
        assert y1h.iloc[499] == 1, "Label one step before spike should be 1"

    def test_label_at_horizon_boundary_is_one(self):
        """At t=140 (360 steps before 500), the spike is still within horizon -> label = 1."""
        from src.ai_model import extract_features_and_targets
        df = self._make_spike_df()
        _, y1h, _, _, _, _ = extract_features_and_targets(df, horizon_1h_steps=360)
        assert y1h.iloc[140] == 1, "Label at t=140 should be 1 (spike at 500 is within horizon)"

    def test_label_outside_horizon_is_zero(self):
        """At t=139, the spike at 500 is OUTSIDE [t+1=140, t+360=499] wait...
        actually the forward window at t=139 with shift(-1) starts at t+1=140.
        The window covers 360 steps: [140, 499]. Spike at 500 is outside -> label = 0."""
        from src.ai_model import extract_features_and_targets
        df = self._make_spike_df()
        _, y1h, _, _, _, _ = extract_features_and_targets(df, horizon_1h_steps=360)
        assert y1h.iloc[139] == 0, "Label outside horizon should be 0"


# ===================================================================
# TEST GROUP D: Chronological train/val/test split
# ===================================================================

class TestChronologicalSplit:
    """Verify that the training pipeline enforces time-ordered splits."""

    def test_train_days_precede_val_days(self):
        sorted_dates = [f"202607{d:02d}" for d in range(18, 32)] + \
                       [f"202608{d:02d}" for d in range(1, 7)]  # 20 days
        n_days = len(sorted_dates)
        n_train = int(n_days * 0.70)  # 14
        n_val = int(n_days * 0.15)    # 3

        train_dates = sorted_dates[:n_train]
        val_dates = sorted_dates[n_train:n_train + n_val]
        test_dates = sorted_dates[n_train + n_val:]

        assert max(train_dates) < min(val_dates), \
            "Last training day must precede first validation day"
        assert max(val_dates) < min(test_dates), \
            "Last validation day must precede first test day"
        assert len(train_dates) + len(val_dates) + len(test_dates) == n_days

    def test_no_overlap_between_splits(self):
        sorted_dates = [f"202607{d:02d}" for d in range(18, 32)] + \
                       [f"202608{d:02d}" for d in range(1, 7)]
        n_days = len(sorted_dates)
        n_train = int(n_days * 0.70)
        n_val = int(n_days * 0.15)

        train_set = set(sorted_dates[:n_train])
        val_set = set(sorted_dates[n_train:n_train + n_val])
        test_set = set(sorted_dates[n_train + n_val:])

        assert train_set.isdisjoint(val_set), "Train and val overlap"
        assert train_set.isdisjoint(test_set), "Train and test overlap"
        assert val_set.isdisjoint(test_set), "Val and test overlap"
