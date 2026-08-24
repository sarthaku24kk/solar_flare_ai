import os
import sys
import glob
import re
import json
import time
from collections import defaultdict
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_solexs_from_zip, load_hel1os_from_zip, merge_and_synchronize
from src.physics_engine import compute_physics_features, detect_flare_events
from src.ai_model import SolarFlareAI, extract_features_and_targets

def run_training_pipeline(data_dir=r'C:\Users\sarth\OneDrive\Pictures\New folder', output_dir=r'C:\Users\sarth\.gemini\antigravity\scratch\solar_flare_ai\models'):
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== SOLAR FLARE AI TRAINING PIPELINE (CAUSAL, NO LEAKAGE) ===")
    print(f"Dataset Directory: {data_dir}")
    print(f"Output Directory: {output_dir}\n")

    # 1. Match all SoLEXS and HEL1OS files by date
    all_files = os.listdir(data_dir)
    slx_files = [f for f in all_files if f.startswith('AL1_SLX') and f.endswith('.zip')]
    hls_files = [f for f in all_files if f.startswith('HLS') and f.endswith('.zip')]

    date_map = defaultdict(lambda: {'slx': [], 'hls': []})
    for f in slx_files:
        m = re.search(r'(\d{8})', f)
        if m:
            date_map[m.group(1)]['slx'].append(f)
    for f in hls_files:
        m = re.search(r'(\d{8})', f)
        if m:
            date_map[m.group(1)]['hls'].append(f)

    sorted_dates = sorted([d for d in date_map.keys() if date_map[d]['slx']])
    print(f"Discovered {len(sorted_dates)} observation days.\n")

    all_X_list = []
    all_y1_list = []
    all_y2_list = []
    all_yflux_list = []
    all_yclass_list = []
    all_events_summary = []
    processed_dates = []  # Track only successfully processed dates

    total_start = time.time()

    for i, date_str in enumerate(sorted_dates, 1):
        slx_file = os.path.join(data_dir, date_map[date_str]['slx'][0])
        hls_files_list = [os.path.join(data_dir, hf) for hf in sorted(date_map[date_str]['hls'])]

        print(f"[{i:02d}/{len(sorted_dates):02d}] Processing Date: {date_str} (SoLEXS: 1, HEL1OS: {len(hls_files_list)})...", end='', flush=True)
        t0 = time.time()

        try:
            # Ingest SoLEXS
            slx_df = load_solexs_from_zip(slx_file)

            # Ingest HEL1OS
            hls_dfs = []
            for hf in hls_files_list:
                try:
                    hdf = load_hel1os_from_zip(hf)
                    hls_dfs.append(hdf)
                except Exception as e:
                    print(f" [Warning HLS: {e}]", end='')

            if not hls_dfs:
                print(" -> Skipped (no valid HEL1OS data).")
                continue

            # Synchronize 24h timeline
            synced_df = merge_and_synchronize(slx_df, hls_dfs, resample_freq='10s')

            # Physics features (CAUSAL — no future data leakage)
            physics_df = compute_physics_features(synced_df, dt_sec=10)

            # Event detection (post-hoc cataloging only, not for training labels)
            events = detect_flare_events(physics_df, min_prominence=8.0, dt_sec=10)

            # Feature extraction for AI training (labels from strictly future windows)
            X, y1, y2, yflux, yclass, valid_mask = extract_features_and_targets(physics_df, dt_sec=10)

            # Only keep rows with complete forecast horizons
            X_valid = X[valid_mask].reset_index(drop=True)
            y1_valid = y1[valid_mask].reset_index(drop=True)
            y2_valid = y2[valid_mask].reset_index(drop=True)
            yflux_valid = yflux[valid_mask].reset_index(drop=True)
            yclass_valid = yclass[valid_mask].reset_index(drop=True)

            all_X_list.append(X_valid)
            all_y1_list.append(y1_valid)
            all_y2_list.append(y2_valid)
            all_yflux_list.append(yflux_valid)
            all_yclass_list.append(yclass_valid)
            processed_dates.append(date_str)  # Track successfully processed date

            for ev in events:
                all_events_summary.append({
                    'date': date_str,
                    'event_id': ev['event_id'],
                    'peak_time': str(ev['peak_time']),
                    'flare_class': ev['flare_class'],
                    'peak_flux': ev['peak_solexs_counts'],
                    'duration_mins': ev['duration_mins'],
                    'hardness_ratio': ev['peak_hardness_ratio']
                })

            elapsed = time.time() - t0
            print(f" Done ({elapsed:.1f}s) | Points: {len(X_valid)} (valid) | Flares: {len(events)}")

        except Exception as e:
            print(f" -> Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCompleted Data Ingestion in {time.time() - total_start:.1f}s.")

    if not all_X_list:
        raise RuntimeError("No training data could be processed.")

    # ── CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT ──
    # Use processed_dates (only successfully processed days) for correct alignment
    # Split: 70% train | 15% validation | 15% test
    n_days = len(all_X_list)
    n_train = int(n_days * 0.70)   # First 70%
    n_val   = int(n_days * 0.15)   # Next 15%
    # n_test = remainder             # Last 15%

    train_dates = processed_dates[:n_train]
    val_dates   = processed_dates[n_train:n_train + n_val]
    test_dates  = processed_dates[n_train + n_val:]

    X_train = pd.concat(all_X_list[:n_train], ignore_index=True)
    y1_train = pd.concat(all_y1_list[:n_train], ignore_index=True)
    y2_train = pd.concat(all_y2_list[:n_train], ignore_index=True)
    yflux_train = pd.concat(all_yflux_list[:n_train], ignore_index=True)
    yclass_train = pd.concat(all_yclass_list[:n_train], ignore_index=True)

    X_val = pd.concat(all_X_list[n_train:n_train + n_val], ignore_index=True)
    y1_val = pd.concat(all_y1_list[n_train:n_train + n_val], ignore_index=True)
    y2_val = pd.concat(all_y2_list[n_train:n_train + n_val], ignore_index=True)
    yflux_val = pd.concat(all_yflux_list[n_train:n_train + n_val], ignore_index=True)
    yclass_val = pd.concat(all_yclass_list[n_train:n_train + n_val], ignore_index=True)

    X_test = pd.concat(all_X_list[n_train + n_val:], ignore_index=True)
    y1_test = pd.concat(all_y1_list[n_train + n_val:], ignore_index=True)
    y2_test = pd.concat(all_y2_list[n_train + n_val:], ignore_index=True)
    yflux_test = pd.concat(all_yflux_list[n_train + n_val:], ignore_index=True)
    yclass_test = pd.concat(all_yclass_list[n_train + n_val:], ignore_index=True)

    print(f"\n=== CHRONOLOGICAL SPLIT ===")
    print(f"Train Days ({len(train_dates)}): {train_dates}")
    print(f"Validation Days ({len(val_dates)}):  {val_dates}")
    print(f"Test Days ({len(test_dates)}):       {test_dates}")
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    print(f"\n=== TRAINING DATASET SUMMARY ===")
    print(f"Train Time-Series Samples: {len(X_train):,}")
    print(f"Total Detected Flare Events (all days): {len(all_events_summary)}")
    print(f"Train 1-Hour Ahead Positives: {np.sum(y1_train):,} ({np.mean(y1_train)*100:.2f}%)")
    print(f"Train 2-Hour Ahead Positives: {np.sum(y2_train):,} ({np.mean(y2_train)*100:.2f}%)")

    # Train AI model on TRAINING DATA ONLY
    ai = SolarFlareAI()
    ai.fit(X_train, y1_train, y2_train, yflux_train, yclass_train)

    # Evaluate on held-out VALIDATION set
    print("\n--- Validation Set ---")
    ai.evaluate(X_val, y1_val, y2_val, yflux_val, yclass_val, label="VALIDATION")

    # Evaluate on held-out TEST set
    print("\n--- Test Set ---")
    test_metrics_1h, test_metrics_2h, test_metrics_class, test_event_1h, test_event_2h = ai.evaluate(X_test, y1_test, y2_test, yflux_test, yclass_test, label="TEST")

    # Save Model
    model_path = os.path.join(output_dir, 'solar_flare_ai_model.pkl')
    ai.save(model_path)

    # Save Catalog & Metrics
    summary = {
        'total_days': len(processed_dates),
        'discovered_days': len(sorted_dates),
        'processed_days': len(processed_dates),
        'skipped_days': len(sorted_dates) - len(processed_dates),
        'split': {
            'train_days': train_dates,
            'val_days': val_dates,
            'test_days': test_dates,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
        },
        'total_flares_detected': len(all_events_summary),
        'flare_events_catalog': all_events_summary,
        'metrics': ai.metrics,
        'test_metrics_1h': test_metrics_1h,
        'test_metrics_2h': test_metrics_2h,
        'test_metrics_class': test_metrics_class,
        'test_event_metrics_1h': test_event_1h,
        'test_event_metrics_2h': test_event_2h,
        'features': list(X_train.columns),
        'causal_pipeline': True,
        'leakage_free': True,
    }

    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary catalog saved to {summary_path}")
    print("AI Model Training Pipeline Completed Successfully (LEAKAGE-FREE)!")
    return ai, summary

if __name__ == '__main__':
    run_training_pipeline()
