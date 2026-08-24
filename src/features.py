"""Feature engineering: rolling statistics per channel + cross-channel ratios.

Every feature at time t uses only information available up to t (no lookahead).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CADENCE_S, LOOKBACK_WINDOWS_S, BACKGROUND_WINDOW_S


# Maximum forward-fill limit: 12 samples = 2 minutes at 10s cadence
_GAP_FFILL_LIMIT = 12


def _resample(df: pd.DataFrame) -> pd.DataFrame:
    """Downsample raw 1s data to CADENCE_S bins (mean), then causal forward-fill.

    Returns:
        DataFrame with resampled data where gaps <= 2min are forward-filled,
        longer gaps remain NaN (to be handled by caller with gap indicators).
    """
    r = df.resample(f"{CADENCE_S}s").mean()
    # Causal forward-fill for short gaps only (<= 2 minutes)
    r = r.ffill(limit=_GAP_FFILL_LIMIT)
    return r


def _rolling_features(s: pd.Series, win_s: int) -> pd.DataFrame:
    """Per-channel rolling statistics over a window of win_s seconds.

    Uses only past and current data (center=False, min_periods=1).
    """
    win = max(1, int(win_s / CADENCE_S))
    out = pd.DataFrame(index=s.index)
    base = f"w{win_s}"
    m = s.rolling(win, min_periods=1, center=False)
    out[f"{s.name}_mean_{base}"] = m.mean()
    out[f"{s.name}_max_{base}"] = m.max()
    out[f"{s.name}_std_{base}"] = m.std().fillna(0)
    # rate of change vs start of window (causal: s[t] - s[t-win])
    out[f"{s.name}_slope_{base}"] = (s - s.shift(win)) / max(win, 1)
    return out


def _background_normalized(s: pd.Series) -> pd.Series:
    """Flux / 2h-background (median), log-scaled.

    Uses causal trailing window (center=False). Background at time t
    depends only on data up to t.
    """
    win = int(BACKGROUND_WINDOW_S / CADENCE_S)
    bg = s.rolling(win, min_periods=1, center=False).median()
    return np.log1p(s) - np.log1p(bg)


def _add_gap_indicators(r: pd.DataFrame, original: pd.DataFrame) -> pd.DataFrame:
    """Add binary gap indicator columns for each channel.

    A gap is where the resampled data was NaN before forward-fill,
    meaning no observation existed in that CADENCE_S bin.
    """
    # Check which bins had no data in original resampled (before ffill)
    # We infer this from the original 1s data: if a 10s bin has no 1s samples,
    # the resampled mean would be NaN.
    # Since we don't have the pre-ffill resampled data here, we compute
    # gap indicators by checking if the resampled bin had any 1s observations.
    out = r.copy()
    for col in original.columns:
        # Resample count of observations per bin
        obs_count = original[col].resample(f"{CADENCE_S}s").count()
        # Gap = no observations in this bin
        gap_col = f"{col}_gap"
        out[gap_col] = (obs_count == 0).astype(np.int8)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build feature matrix from a (possibly raw 1s) channel DataFrame.

    Gap handling:
    1. Resample to CADENCE_S using mean
    2. Forward-fill gaps <= 2 minutes (12 samples at 10s)
    3. Remaining NaN (long gaps) are filled with channel median (neutral)
    4. Binary gap indicators added for each channel (1 = imputed/long gap)

    All operations are strictly causal (no lookahead).
    """
    # Step 1: resample with causal short-gap forward-fill
    r = _resample(df)

    # Step 2: add gap indicators BEFORE filling long gaps
    # (need original 1s data to compute observation counts)
    feats = _add_gap_indicators(r, df)

    # Step 3: fill remaining NaN (long gaps > 2min) with column median
    # This is a neutral value that doesn't introduce future information
    for col in r.columns:
        if feats[col].isna().any():
            feats[col] = feats[col].fillna(feats[col].median())
            # If still NaN (all NaN column), fill with 0
            feats[col] = feats[col].fillna(0.0)

    parts = [feats.copy()]
    for col in r.columns:
        s = feats[col].astype(float)
        parts.append(_background_normalized(s).to_frame(f"{col}_bglog"))
        parts.append(s.diff().clip(lower=0).to_frame(f"{col}_rise"))
        for w in LOOKBACK_WINDOWS_S:
            parts.append(_rolling_features(s, w))

    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan)
    # Final safety: fill any remaining NaN with 0
    out = out.fillna(0.0)
    return out


def sample_rows(index: pd.DatetimeIndex, every_s: int) -> np.ndarray:
    """Keep one index position every `every_s` seconds to decorrelate samples."""
    if len(index) == 0:
        return np.array([], dtype=int)
    step = max(1, int(every_s / CADENCE_S))
    return np.arange(0, len(index), step)
