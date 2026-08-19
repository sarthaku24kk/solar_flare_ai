import zipfile
import io
import os
import gzip
import numpy as np
import pandas as pd
from astropy.io import fits

def clean_fits_array(arr, dtype='<f8'):
    """Safely converts big-endian FITS array into native float64."""
    return np.array(arr, dtype=dtype)

def load_solexs_from_zip(zip_source):
    """
    Parses a SoLEXS 24-hour daily zip file (either filepath or file-like / bytes buffer).
    Extracts the primary lightcurve (SDD2/SDD1) into a pandas DataFrame.
    """
    with zipfile.ZipFile(zip_source, 'r') as z:
        lc_files = [n for n in z.namelist() if n.endswith(('.lc.gz', '.lc')) and not n.startswith('__MACOSX')]
        if not lc_files:
            raise ValueError("No lightcurve (.lc/.lc.gz) file found in SoLEXS zip archive.")
        
        # Prefer SDD2 detector if present, otherwise first available
        target = [f for f in lc_files if 'SDD2' in f]
        target_file = target[0] if target else lc_files[0]
        
        raw = z.read(target_file)
        if target_file.endswith('.gz'):
            raw = gzip.decompress(raw)
        
        with fits.open(io.BytesIO(raw)) as hdul:
            rate_hdu = hdul['RATE'] if 'RATE' in hdul else hdul[1]
            rate_data = rate_hdu.data
            time_arr = clean_fits_array(rate_data['TIME'])
            counts_arr = clean_fits_array(rate_data['COUNTS'])
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(time_arr, unit='s', utc=True),
                'solexs_counts': counts_arr
            })
            
            # Drop NaN or negative values (telemetry loss / bad packets)
            df = df[~np.isnan(df['solexs_counts']) & (df['solexs_counts'] >= 0)]
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            return df

def load_hel1os_from_zip(zip_source):
    """
    Parses a HEL1OS 12-hour (or 24-hour) zip file.
    Extracts multi-band CZT and CDTE lightcurves.
    """
    with zipfile.ZipFile(zip_source, 'r') as z:
        lc_files = [n for n in z.namelist() if 'lightcurve' in n and n.endswith('.fits') and not n.startswith('__MACOSX')]
        if not lc_files:
            # Check for any .fits with rate data
            lc_files = [n for n in z.namelist() if n.endswith('.fits') and ('czt' in n or 'cdte' in n) and not n.startswith('__MACOSX')]
        
        if not lc_files:
            raise ValueError("No lightcurve FITS files found in HEL1OS zip archive.")
        
        records = {}
        for lcf in lc_files:
            try:
                with fits.open(io.BytesIO(z.read(lcf))) as hdul:
                    for hdu in hdul:
                        if hasattr(hdu, 'columns') and hdu.columns and 'ISOT' in hdu.columns.names and 'CTR' in hdu.columns.names:
                            hname = hdu.name.strip()
                            isot_arr = [str(s).strip() for s in hdu.data['ISOT']]
                            ctr_arr = clean_fits_array(hdu.data['CTR'])
                            
                            sub_df = pd.DataFrame({
                                'timestamp': pd.to_datetime(isot_arr, utc=True),
                                hname: ctr_arr
                            })
                            sub_df = sub_df.drop_duplicates('timestamp').set_index('timestamp')
                            records[hname] = sub_df
            except Exception:
                continue
        
        if not records:
            raise ValueError("Could not extract valid count-rate HDUs from HEL1OS archive.")
        
        # Combine on timestamp index
        combined = pd.concat(records.values(), axis=1, sort=False).sort_index()
        
        # Identify major energy bands
        czt_total_cols = [c for c in combined.columns if 'CZT' in c and ('18.00KEV_TO_160.00KEV' in c or '10.00KEV_TO_150.00KEV' in c)]
        if not czt_total_cols:
            czt_total_cols = [c for c in combined.columns if 'CZT' in c]
        
        combined['hel1os_czt_total'] = combined[czt_total_cols].sum(axis=1) if czt_total_cols else 0.0
        
        # Specific sub-bands if present
        for band_name, col_key in [
            ('hel1os_10_20', '10.00KEV_TO_20.00KEV'),
            ('hel1os_20_40', '20.00KEV_TO_40.00KEV'),
            ('hel1os_40_60', '40.00KEV_TO_60.00KEV'),
            ('hel1os_60_80', '60.00KEV_TO_80.00KEV'),
            ('hel1os_80_150', '80.00KEV_TO_150.00KEV'),
        ]:
            bcols = [c for c in combined.columns if col_key in c]
            if bcols:
                combined[band_name] = combined[bcols].sum(axis=1)
            else:
                combined[band_name] = 0.0
        
        return combined.reset_index()

def merge_and_synchronize(solexs_df, hel1os_df_list, resample_freq='10s'):
    """
    Merges 1 SoLEXS 24h DataFrame with 1 or 2 HEL1OS DataFrames,
    and resamples onto a unified continuous UTC time-grid.
    """
    # 1. Merge HEL1OS parts
    if isinstance(hel1os_df_list, pd.DataFrame):
        hel1os_df_list = [hel1os_df_list]
    
    if len(hel1os_df_list) == 1:
        hls_all = hel1os_df_list[0]
    else:
        hls_all = pd.concat(hel1os_df_list, ignore_index=True)
    
    hls_all = hls_all.sort_values('timestamp').drop_duplicates('timestamp')
    
    # 2. Resample SoLEXS
    slx_resampled = (
        solexs_df.set_index('timestamp')
        .resample(resample_freq)
        .mean()
        .interpolate(method='linear', limit=12) # fill up to 2 mins gap
    )
    
    # 3. Resample HEL1OS
    hls_resampled = (
        hls_all.set_index('timestamp')
        .resample(resample_freq)
        .mean()
        .interpolate(method='linear', limit=12)
    )
    
    # 4. Align both on shared time index
    merged = pd.merge(
        slx_resampled,
        hls_resampled,
        left_index=True,
        right_index=True,
        how='outer'
    ).sort_index()
    
    # Fill remaining NaNs with forward/backward fill or baseline 0
    merged['solexs_counts'] = merged['solexs_counts'].ffill().bfill().fillna(0.0)
    merged['hel1os_czt_total'] = merged['hel1os_czt_total'].ffill().bfill().fillna(0.0)
    
    for b in ['hel1os_10_20', 'hel1os_20_40', 'hel1os_40_60', 'hel1os_60_80', 'hel1os_80_150']:
        if b in merged.columns:
            merged[b] = merged[b].ffill().bfill().fillna(0.0)
        else:
            merged[b] = 0.0
            
    return merged.reset_index()
