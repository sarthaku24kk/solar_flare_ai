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

def validate_fits_header(hdul, required_hdus=None, required_cols=None):
    """Validate FITS file structure and required columns."""
    if required_hdus:
        for hdu_name in required_hdus:
            if hdu_name not in hdul:
                raise ValueError(f"Required HDU '{hdu_name}' not found in FITS file. Available: {list(hdul.keys())}")
    if required_cols:
        for hdu_name, cols in required_cols.items():
            if hdu_name in hdul:
                hdu_cols = hdul[hdu_name].columns.names if hasattr(hdul[hdu_name], 'columns') else []
                for col in cols:
                    if col not in hdu_cols:
                        raise ValueError(f"Required column '{col}' not found in HDU '{hdu_name}'. Available: {hdu_cols}")
    return True

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
            # Validate FITS structure
            validate_fits_header(hdul, required_hdus=['RATE'], required_cols={'RATE': ['TIME', 'COUNTS']})
            
            rate_hdu = hdul['RATE']
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
                            # Validate this HDU has required columns
                            validate_fits_header(hdul, required_hdus=[hdu.name], required_cols={hdu.name: ['ISOT', 'CTR']})
                            
                            hname = hdu.name.strip()
                            isot_arr = [str(s).strip() for s in hdu.data['ISOT']]
                            ctr_arr = clean_fits_array(hdu.data['CTR'])
                            
                            sub_df = pd.DataFrame({
                                'timestamp': pd.to_datetime(isot_arr, utc=True),
                                hname: ctr_arr
                            })
                            sub_df = sub_df.drop_duplicates('timestamp').set_index('timestamp')
                            records[hname] = sub_df
            except Exception as e:
                print(f"Warning: Failed to parse {lcf}: {e}")
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
    Adds gap indicator features so the model knows when data was imputed.
    """
    # 1. Merge HEL1OS parts
    if isinstance(hel1os_df_list, pd.DataFrame):
        hel1os_df_list = [hel1os_df_list]
    
    if len(hel1os_df_list) == 1:
        hls_all = hel1os_df_list[0]
    else:
        hls_all = pd.concat(hel1os_df_list, ignore_index=True)
    
    hls_all = hls_all.sort_values('timestamp').drop_duplicates('timestamp')
    
    # 2. Resample SoLEXS - causal forward-fill with gap tracking
    slx_resampled = solexs_df.set_index('timestamp').resample(resample_freq).mean()
    slx_gaps = slx_resampled['solexs_counts'].isna().astype(int)  # 1 = gap
    slx_resampled = slx_resampled.ffill(limit=12).fillna(0.0)  # short gaps: ffill; long gaps: 0
    
    # 3. Resample HEL1OS - causal forward-fill with gap tracking
    hls_resampled = hls_all.set_index('timestamp').resample(resample_freq).mean()
    hls_gaps = hls_resampled['hel1os_czt_total'].isna().astype(int)  # 1 = gap
    hls_resampled = hls_resampled.ffill(limit=12).fillna(0.0)
    
    # 4. Align both on shared time index
    merged = pd.merge(
        slx_resampled,
        hls_resampled,
        left_index=True,
        right_index=True,
        how='outer'
    ).sort_index()
    
    # Gap indicators (1 = data was imputed/zero-filled, 0 = real measurement)
    merged['solexs_gap'] = slx_gaps.reindex(merged.index, fill_value=1).astype(int)
    merged['hel1os_gap'] = hls_gaps.reindex(merged.index, fill_value=1).astype(int)
    
    # Fill remaining NaNs with CAUSAL forward-fill only (no bfill = no future data)
    merged['solexs_counts'] = merged['solexs_counts'].ffill().fillna(0.0)
    merged['hel1os_czt_total'] = merged['hel1os_czt_total'].ffill().fillna(0.0)
    
    for b in ['hel1os_10_20', 'hel1os_20_40', 'hel1os_40_60', 'hel1os_60_80', 'hel1os_80_150']:
        if b in merged.columns:
            merged[b] = merged[b].ffill().fillna(0.0)
        else:
            merged[b] = 0.0
            
    return merged.reset_index()
