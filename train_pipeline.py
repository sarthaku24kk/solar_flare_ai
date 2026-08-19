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
    
    print(f"=== SOLAR FLARE AI TRAINING PIPELINE ===")
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
    print(f"Discovered {len(sorted_dates)} observation days for training.\n")
    
    all_X_list = []
    all_y1_list = []
    all_y2_list = []
    all_yflux_list = []
    all_events_summary = []
    
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
            
            # Physics features & Event detection
            physics_df = compute_physics_features(synced_df, dt_sec=10)
            events = detect_flare_events(physics_df, min_prominence=8.0, dt_sec=10)
            
            # Feature extraction for AI training
            X, y1, y2, yflux, yclass = extract_features_and_targets(physics_df, dt_sec=10)
            
            all_X_list.append(X)
            all_y1_list.append(y1)
            all_y2_list.append(y2)
            all_yflux_list.append(yflux)
            
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
            print(f" Done ({elapsed:.1f}s) | Points: {len(synced_df)} | Flares Detected: {len(events)}")
            
        except Exception as e:
            print(f" -> Error: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"\nCompleted Data Ingestion in {time.time() - total_start:.1f}s.")
    
    if not all_X_list:
        raise RuntimeError("No training data could be processed.")
        
    # Combine all days into one dataset
    X_full = pd.concat(all_X_list, ignore_index=True)
    y1_full = pd.concat(all_y1_list, ignore_index=True)
    y2_full = pd.concat(all_y2_list, ignore_index=True)
    yflux_full = pd.concat(all_yflux_list, ignore_index=True)
    
    print(f"\n=== TRAINING DATASET SUMMARY ===")
    print(f"Total Time-Series Samples: {len(X_full):,}")
    print(f"Total Detected Flare Events: {len(all_events_summary)}")
    print(f"1-Hour Ahead Positive Targets: {np.sum(y1_full):,} ({np.mean(y1_full)*100:.2f}%)")
    print(f"2-Hour Ahead Positive Targets: {np.sum(y2_full):,} ({np.mean(y2_full)*100:.2f}%)")
    
    # Train AI model
    ai = SolarFlareAI()
    ai.fit(X_full, y1_full, y2_full, yflux_full)
    
    # Save Model
    model_path = os.path.join(output_dir, 'solar_flare_ai_model.pkl')
    ai.save(model_path)
    
    # Save Catalog & Metrics
    summary = {
        'total_days': len(sorted_dates),
        'total_samples': len(X_full),
        'total_flares_detected': len(all_events_summary),
        'flare_events_catalog': all_events_summary,
        'metrics': ai.metrics,
        'features': list(X_full.columns)
    }
    
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSummary catalog saved to {summary_path}")
    print("AI Model Training Pipeline Completed Successfully!")
    return ai, summary

if __name__ == '__main__':
    run_training_pipeline()
