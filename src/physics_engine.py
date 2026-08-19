import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

def compute_physics_features(df, dt_sec=10):
    """
    Computes solar physics indicators from synchronized SoLEXS and HEL1OS time series:
    - Dynamic baseline (quiescent Sun level)
    - Excess flux above baseline
    - Hardness Ratio (Hard X-ray / Soft X-ray)
    - First and second time derivatives (Neupert Effect)
    - Multi-scale rolling volatility and rate of rise
    """
    df = df.copy()
    
    # 1. Smooth signals with rolling median to reject single-bin detector spikes
    df['solexs_smooth'] = df['solexs_counts'].rolling(window=7, center=True, min_periods=1).median()
    df['hel1os_smooth'] = df['hel1os_czt_total'].rolling(window=7, center=True, min_periods=1).median()
    
    # Optional Savitzky-Golay filtering for smooth continuous derivatives
    try:
        df['solexs_savgol'] = savgol_filter(df['solexs_smooth'], window_length=15, polyorder=2)
        df['hel1os_savgol'] = savgol_filter(df['hel1os_smooth'], window_length=15, polyorder=2)
    except Exception:
        df['solexs_savgol'] = df['solexs_smooth']
        df['hel1os_savgol'] = df['hel1os_smooth']
    
    # 2. Dynamic Baseline Estimation (30-min rolling 10th percentile / minimum)
    baseline_window = int(1800 / dt_sec) # 30 mins
    df['solexs_baseline'] = (
        df['solexs_smooth']
        .rolling(window=baseline_window, min_periods=max(5, baseline_window // 4), center=True)
        .quantile(0.10)
        .bfill()
        .ffill()
    )
    df['hel1os_baseline'] = (
        df['hel1os_smooth']
        .rolling(window=baseline_window, min_periods=max(5, baseline_window // 4), center=True)
        .quantile(0.10)
        .bfill()
        .ffill()
    )
    
    # 3. Excess Flux
    df['solexs_excess'] = np.maximum(0.0, df['solexs_smooth'] - df['solexs_baseline'])
    df['hel1os_excess'] = np.maximum(0.0, df['hel1os_smooth'] - df['hel1os_baseline'])
    
    # 4. Hardness Ratio (HR)
    # HR rises during impulsive non-thermal heating phase
    df['hardness_ratio'] = (df['hel1os_smooth'] + 0.1) / (df['solexs_smooth'] + 1.0)
    df['hardness_ratio_excess'] = (df['hel1os_excess'] + 0.01) / (df['solexs_excess'] + 0.1)
    
    # 5. Temporal Derivatives (Neupert Effect indicator: d(SoLEXS)/dt ~ HEL1OS)
    df['d_solexs_dt'] = np.gradient(df['solexs_savgol'], dt_sec)
    df['d2_solexs_dt2'] = np.gradient(df['d_solexs_dt'], dt_sec)
    df['d_hel1os_dt'] = np.gradient(df['hel1os_savgol'], dt_sec)
    
    # 6. Multi-scale rolling dynamics (5-min, 15-min, 30-min windows)
    w_5m = max(3, int(300 / dt_sec))
    w_15m = max(5, int(900 / dt_sec))
    
    df['solexs_mean_5m'] = df['solexs_smooth'].rolling(w_5m, min_periods=1).mean()
    df['solexs_std_5m'] = df['solexs_smooth'].rolling(w_5m, min_periods=1).std().fillna(0)
    df['solexs_rise_rate_15m'] = (df['solexs_smooth'] - df['solexs_smooth'].shift(w_15m)).fillna(0)
    
    df['hel1os_mean_5m'] = df['hel1os_smooth'].rolling(w_5m, min_periods=1).mean()
    df['hel1os_std_5m'] = df['hel1os_smooth'].rolling(w_5m, min_periods=1).std().fillna(0)
    
    # Flare indicator score (Physics-guided preliminary score)
    noise_std = df['solexs_counts'].std() if df['solexs_counts'].std() > 0 else 1.0
    df['flare_signal_sigma'] = df['solexs_excess'] / noise_std
    
    return df

def classify_flare_intensity(peak_flux):
    """
    Classifies flare intensity based on Aditya-L1 SoLEXS peak count rate:
    - B-Class: 15 to 40 counts/s (Minor solar flare)
    - C-Class: 40 to 150 counts/s (Common solar flare)
    - M-Class: 150 to 800 counts/s (Strong flare - space weather risk)
    - X-Class: > 800 counts/s (Major/Extreme flare - severe radio blackout)
    """
    if peak_flux >= 800:
        return 'X-Class', 'CRITICAL (Severe HF Radio Blackout & Grid Threat)', '#FF1744'
    elif peak_flux >= 150:
        return 'M-Class', 'HIGH (Moderate Radio Blackout & Sat Warning)', '#FF9100'
    elif peak_flux >= 40:
        return 'C-Class', 'MODERATE (Minor Space Weather Disruption)', '#FFD600'
    elif peak_flux >= 15:
        return 'B-Class', 'LOW (Background Active Region Activity)', '#00E676'
    else:
        return 'Quiet/Micro', 'NOMINAL (Quiet Solar Conditions)', '#00B0FF'

def detect_flare_events(df, min_prominence=8.0, min_distance_mins=10, dt_sec=10):
    """
    Detects solar flare events across the timeline:
    Returns a list of structured flare event dictionaries.
    """
    if 'solexs_excess' not in df.columns:
        df = compute_physics_features(df, dt_sec=dt_sec)
        
    signal = df['solexs_excess'].values
    timestamps = df['timestamp'].values
    min_dist_samples = max(3, int((min_distance_mins * 60) / dt_sec))
    
    # Peak detection on excess flux
    peaks, properties = find_peaks(
        signal,
        prominence=min_prominence,
        distance=min_dist_samples,
        width=int(60 / dt_sec) # at least 1 min width
    )
    
    events = []
    n_points = len(signal)
    
    for i, p_idx in enumerate(peaks):
        peak_time = pd.to_datetime(timestamps[p_idx])
        peak_flux = float(df['solexs_counts'].iloc[p_idx])
        peak_excess = float(signal[p_idx])
        
        # Determine Start Time (step back until excess <= 10% of peak excess or <= baseline threshold)
        start_idx = p_idx
        thresh_start = max(1.0, 0.15 * peak_excess)
        while start_idx > 0 and signal[start_idx] > thresh_start and (p_idx - start_idx) < (7200 / dt_sec):
            start_idx -= 1
        start_time = pd.to_datetime(timestamps[start_idx])
        
        # Determine End Time (step forward until excess <= 20% of peak or baseline reached)
        end_idx = p_idx
        thresh_end = max(1.0, 0.20 * peak_excess)
        while end_idx < n_points - 1 and signal[end_idx] > thresh_end and (end_idx - p_idx) < (14400 / dt_sec):
            end_idx += 1
        end_time = pd.to_datetime(timestamps[end_idx])
        
        duration_mins = max(1.0, (end_time - start_time).total_seconds() / 60.0)
        rise_time_mins = max(0.5, (peak_time - start_time).total_seconds() / 60.0)
        decay_time_mins = max(0.5, (end_time - peak_time).total_seconds() / 60.0)
        
        # Peak Hardness Ratio
        event_slice = df.iloc[start_idx:end_idx+1]
        max_hr = float(event_slice['hardness_ratio'].max()) if len(event_slice) > 0 else 0.0
        max_hls_flux = float(event_slice['hel1os_czt_total'].max()) if len(event_slice) > 0 else 0.0
        
        flare_cls, impact_desc, color_code = classify_flare_intensity(peak_flux)
        
        # Total integrated counts (fluence proxy)
        fluence = float(event_slice['solexs_counts'].sum() * dt_sec)
        
        events.append({
            'event_id': f"FLARE-{peak_time.strftime('%Y%m%d-%H%M')}",
            'start_time': start_time,
            'peak_time': peak_time,
            'end_time': end_time,
            'duration_mins': round(duration_mins, 1),
            'rise_time_mins': round(rise_time_mins, 1),
            'decay_time_mins': round(decay_time_mins, 1),
            'peak_solexs_counts': round(peak_flux, 1),
            'peak_hel1os_counts': round(max_hls_flux, 1),
            'peak_hardness_ratio': round(max_hr, 4),
            'total_fluence': round(fluence, 0),
            'flare_class': flare_cls,
            'space_weather_alert': impact_desc,
            'color': color_code
        })
        
    return events
