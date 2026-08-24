"""Flare event detection and forecast-label construction.

Flare detection uses SoLEXS soft-X-ray total counts (background-subtracted
adaptive threshold). Onsets feed a binary forecast label: does a flare start
within the next HORIZON seconds?
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (FLARE_WINDOW_S, FLARE_MULT, FLARE_MIN_PEAK,
                        FLARE_MIN_DUR_S, BACKGROUND_WINDOW_S)


def detect_flare_events(flux: pd.Series, dt_s: float = 1.0) -> pd.DataFrame:
    """Find flare events in a 1-second flux series (POST-HOC ONLY).

    Returns DataFrame with columns: onset, peak_time, peak_flux, end, duration_s.

    Uses a 2-hour trailing median background to avoid flare contamination.
    This function examines the FULL time series and is NOT suitable for
    real-time feature generation. It is used only for post-hoc event
    cataloging and label construction.
    """
    flux = flux.astype(float)
    # Use 2-hour trailing median background (same as feature background)
    # to avoid flare contamination of the background estimate.
    bg_win = int(BACKGROUND_WINDOW_S)  # 7200s = 2 hours
    background = flux.rolling(bg_win, min_periods=1, center=False).median()
    # Only require ratio threshold when background is non-trivial
    active = (flux >= FLARE_MULT * background) | (flux >= FLARE_MIN_PEAK)
    active = active.fillna(False)

    diff = active.astype(int).diff().fillna(0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    events = []
    for s in starts:
        e = ends[ends > s][0] if np.any(ends > s) else len(flux) - 1
        seg = flux.iloc[s:e + 1]
        peak_i = seg.idxmax()
        if seg.max() < FLARE_MIN_PEAK:
            continue
        duration = len(seg) * dt_s
        if duration < FLARE_MIN_DUR_S:
            continue
        events.append({
            "onset": seg.index[0],
            "peak_time": peak_i,
            "peak_flux": float(seg.max()),
            "end": seg.index[-1],
            "duration_s": duration,
        })
    return pd.DataFrame(events)


def build_labels(index: pd.DatetimeIndex, onsets: list, horizon_s: int,
                 gap_s: int = 0) -> pd.Series:
    """Binary label: 1 if a flare onset falls within (t+gap, t+gap+horizon].

    gap_s enforces a minimum lead time so the model must predict from
    quiet pre-flare data, not while the flare is already rising.

    For a flare onset at time o, the label is 1 for times t where:
        t in [o - gap - horizon, o - gap)

    This means the model sees data up to time t and must predict a flare
    onset in the future window (t+gap, t+gap+horizon].
    """
    onset_arr = np.asarray([t.value for t in pd.to_datetime(onsets)])
    idx_ns = index.values.astype("datetime64[ns]").astype(np.int64)
    gap_ns = int(gap_s) * 1_000_000_000
    horizon_ns = int(horizon_s) * 1_000_000_000
    labels = np.zeros(len(index), dtype=np.int8)
    for o in onset_arr:
        # Label is 1 for t in [o - gap - horizon, o - gap)
        # i.e., flare onset falls in (t+gap, t+gap+horizon]
        before = idx_ns < o - gap_ns
        within = idx_ns >= o - gap_ns - horizon_ns
        labels[before & within] = 1
    return pd.Series(labels, index=index)
