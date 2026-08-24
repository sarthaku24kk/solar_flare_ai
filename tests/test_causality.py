"""Causality regression tests for the solar flare forecasting pipeline.

These tests verify that no future information leaks into features or labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import CADENCE_S, GAP_S, HORIZON_S
from src.features import build_features, _GAP_FFILL_LIMIT
from src.flares import build_labels, detect_flare_events


class TestCausality:
    """Tests that verify no future data leaks into features."""

    def make_synthetic_data(self, n_seconds=5000, flare_at=None):
        """Create synthetic 1-second SoLEXS + HEL1OS data with optional flare."""
        index = pd.date_range("2026-01-01", periods=n_seconds, freq="1s", tz=None)
        solexs = np.random.poisson(20, n_seconds).astype(float)
        hel1os_1 = np.random.poisson(5, n_seconds).astype(float)
        hel1os_2 = np.random.poisson(3, n_seconds).astype(float)

        if flare_at is not None:
            flare_duration = 300
            t = np.arange(flare_duration)
            flare_profile = 500 * np.exp(-t / 60) * (1 - np.exp(-t / 30))
            end = min(flare_at + flare_duration, n_seconds)
            solexs[flare_at:end] += flare_profile[:end - flare_at]

        df = pd.DataFrame({
            "solexs_sdd2_counts": solexs,
            "hel1os_cdte1_5-20keV": hel1os_1,
            "hel1os_czt1_20-40keV": hel1os_2,
        }, index=index)
        return df

    def _get_feat_idx(self, feats, t_seconds):
        """Convert 1-second timestamp to feature row index (10s cadence)."""
        return t_seconds // CADENCE_S

    def test_features_do_not_see_future(self):
        """Mutate future data and verify past features don't change."""
        df = self.make_synthetic_data(n_seconds=5000, flare_at=3000)
        feats_orig = build_features(df)

        # Mutate data far in the future (after t=4000s)
        df_mutated = df.copy()
        df_mutated.loc[df_mutated.index[4000:], "solexs_sdd2_counts"] *= 100

        feats_mutated = build_features(df_mutated)

        # Check features at t=2000s (feature index 200)
        check_idx = self._get_feat_idx(feats_orig, 2000)
        for col in feats_orig.columns:
            orig_val = feats_orig.iloc[check_idx][col]
            mut_val = feats_mutated.iloc[check_idx][col]
            if np.isnan(orig_val) and np.isnan(mut_val):
                continue
            assert np.isclose(orig_val, mut_val, rtol=1e-10, atol=1e-12), \
                f"Feature {col} at t=2000s changed due to future mutation: {orig_val} -> {mut_val}"

    def test_features_at_flare_onset_not_influenced_by_flare_peak(self):
        """Features at flare onset should not see the flare peak."""
        df = self.make_synthetic_data(n_seconds=5000, flare_at=3000)
        feats_orig = build_features(df)

        # Mutate the flare peak region (t=3060 to 3120) to be much larger
        df_mutated = df.copy()
        df_mutated.loc[df_mutated.index[3060:3120], "solexs_sdd2_counts"] *= 10

        feats_mutated = build_features(df_mutated)

        # Check features at flare onset t=3000s (feature index 300)
        check_idx = self._get_feat_idx(feats_orig, 3000)
        for col in feats_orig.columns:
            orig_val = feats_orig.iloc[check_idx][col]
            mut_val = feats_mutated.iloc[check_idx][col]
            if np.isnan(orig_val) and np.isnan(mut_val):
                continue
            assert np.isclose(orig_val, mut_val, rtol=1e-10, atol=1e-12), \
                f"Feature {col} at flare onset changed due to future peak mutation: {orig_val} -> {mut_val}"

    def test_gap_indicators_causal(self):
        """Gap indicators should only depend on past/current data availability."""
        df = self.make_synthetic_data(n_seconds=5000)
        df_with_gap = df.copy()
        df_with_gap.loc[df_with_gap.index[2000:2050], "solexs_sdd2_counts"] = np.nan

        feats = build_features(df_with_gap)

        # Gap at 2000s -> feature bin 200
        gap_bin = 2000 // CADENCE_S  # 200
        pre_gap_bin = gap_bin - 1   # 199
        post_gap_bin = gap_bin + 5  # 205 (after 50s gap = 5 bins)

        assert feats.iloc[gap_bin]["solexs_sdd2_counts_gap"] == 1, "Gap bin should have gap=1"
        assert feats.iloc[pre_gap_bin]["solexs_sdd2_counts_gap"] == 0, "Pre-gap bin should have gap=0"
        assert feats.iloc[post_gap_bin]["solexs_sdd2_counts_gap"] == 0, "Post-gap bin should have gap=0"

    def test_rolling_windows_are_trailing(self):
        """Verify all rolling windows use center=False (trailing only)."""
        df = self.make_synthetic_data(n_seconds=5000)
        feats_orig = build_features(df)

        # Mutate future data at t>=2000s
        df_mutated = df.copy()
        df_mutated.loc[df_mutated.index[2000:], "solexs_sdd2_counts"] *= 100

        feats_mutated = build_features(df_mutated)

        # w60 window at t=1000s (feat idx 100) uses [940, 1000] - should not be affected by t>=2000
        check_idx = 1000 // CADENCE_S  # 100
        for col in ["solexs_sdd2_counts_mean_w60", "solexs_sdd2_counts_max_w60"]:
            orig = feats_orig.iloc[check_idx][col]
            mut = feats_mutated.iloc[check_idx][col]
            assert np.isclose(orig, mut, rtol=1e-10), \
                f"{col} at t=1000s changed due to future data at t>=2000s"

    def test_background_normalized_is_causal(self):
        """Background-normalized feature should be causal (trailing median)."""
        df = self.make_synthetic_data(n_seconds=5000)
        feats_orig = build_features(df)

        df_mutated = df.copy()
        df_mutated.loc[df_mutated.index[2000:], "solexs_sdd2_counts"] *= 100

        feats_mutated = build_features(df_mutated)

        # bglog at t=1000s (feat idx 100) should not change
        check_idx = 1000 // CADENCE_S  # 100
        orig = feats_orig.iloc[check_idx]["solexs_sdd2_counts_bglog"]
        mut = feats_mutated.iloc[check_idx]["solexs_sdd2_counts_bglog"]
        assert np.isclose(orig, mut, rtol=1e-10), \
            "Background-normalized feature at t=1000s changed due to future data"


class TestLabelCausality:
    """Tests for label construction causality."""

    def test_label_uses_future_not_past(self):
        """Labels at time t should depend on flares in (t+gap, t+gap+horizon]."""
        index = pd.date_range("2026-01-01", periods=10000, freq="1s", tz=None)
        flux = pd.Series(np.full(10000, 20.0, dtype=float), index=index)
        flux.iloc[5000:5300] = 500.0  # flare from 5000-5300

        events = detect_flare_events(flux)
        assert len(events) == 1
        onset = events.iloc[0]["onset"]

        labels = build_labels(index, [onset], HORIZON_S, GAP_S)

        # With GAP_S=3600, HORIZON_S=3600:
        # Label should be 1 for t in [onset - GAP_S - HORIZON_S, onset - GAP_S)
        # = [5000 - 7200, 5000 - 3600) = [-2200, 1400) relative to onset
        onset_idx = index.get_loc(onset)
        label_end = onset_idx - GAP_S  # 5000 - 3600 = 1400

        pos_indices = np.where(labels.values == 1)[0]
        if len(pos_indices) > 0:
            assert pos_indices[0] == 0, "Labels should start positive from beginning"
            assert pos_indices[-1] == label_end - 1, f"Last positive should be at {label_end-1}, got {pos_indices[-1]}"

        # At flare onset (t=5000), label should be 0
        assert labels.iloc[onset_idx] == 0, "Label at flare onset should be 0 (flare already started)"

        # At t=onset - GAP_S - 1 (just before label window), label should be 0
        before_label = onset_idx - GAP_S - HORIZON_S - 1
        if before_label >= 0:
            assert labels.iloc[before_label] == 0, "Label before window should be 0"


class TestGapHandling:
    """Tests for gap imputation policy."""

    def test_short_gap_forward_filled(self):
        """Gaps <= 2 minutes (12 samples at 10s) should be forward-filled."""
        index = pd.date_range("2026-01-01", periods=1000, freq="1s", tz=None)
        solexs = np.full(1000, 20.0, dtype=float)
        solexs[400:460] = np.nan  # 60s gap
        df = pd.DataFrame({"solexs_sdd2_counts": solexs}, index=index)

        feats = build_features(df)

        # At 10s cadence, 1 minute = 6 bins. Gap is 6 bins < 12, should be ffilled
        # Check middle of gap (bin 42 = 420s)
        gap_bin = 420 // CADENCE_S  # 42
        val = feats.iloc[gap_bin]["solexs_sdd2_counts"]
        assert val == 20.0, f"Short gap not forward-filled at {gap_bin}: {val}"

    def test_long_gap_median_filled(self):
        """Gaps > 2 minutes should NOT be forward-filled beyond 12 bins (filled with median)."""
        index = pd.date_range("2026-01-01", periods=4000, freq="1s", tz=None)
        solexs = np.full(4000, 50.0, dtype=float)
        solexs[1500:1800] = np.nan  # 300s gap
        solexs[1800:] = 10.0  # post-gap = 10
        df = pd.DataFrame({"solexs_sdd2_counts": solexs}, index=index)

        feats = build_features(df)

        # Gap at bins 150-180 (30 bins). First 12 bins ffilled, rest median-filled.
        # Median should be ~10 (post-gap dominates)
        median_val = feats["solexs_sdd2_counts"].median()

        # Check bin 165 (15 bins into gap, beyond 12-bin ffill limit)
        val = feats.iloc[165]["solexs_sdd2_counts"]
        assert np.isclose(val, median_val), \
            f"Long gap bin 165 not median-filled: {val} vs median {median_val}"

        # Gap indicator should be 1 for entire gap
        assert feats.iloc[165]["solexs_sdd2_counts_gap"] == 1
        assert feats.iloc[149]["solexs_sdd2_counts_gap"] == 0
        assert feats.iloc[181]["solexs_sdd2_counts_gap"] == 0


class TestNoCenteredOperations:
    """Verify no centered/symmetric operations are used."""

    def test_no_centered_rolling(self):
        """All rolling operations should use center=False."""
        import inspect
        from src.features import _rolling_features, _background_normalized

        source = inspect.getsource(_rolling_features)
        assert "center=False" in source, "_rolling_features should use center=False"

        source = inspect.getsource(_background_normalized)
        assert "center=False" in source, "_background_normalized should use center=False"

    def test_no_bfill(self):
        """No backward fill should be used."""
        import inspect
        from src.features import build_features
        source = inspect.getsource(build_features)
        assert ".bfill(" not in source, "build_features should not use bfill"

    def test_no_interpolate(self):
        """No interpolation (which can use future data) should be used."""
        import inspect
        from src.features import build_features
        source = inspect.getsource(build_features)
        assert ".interpolate(" not in source, "build_features should not use interpolate"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])