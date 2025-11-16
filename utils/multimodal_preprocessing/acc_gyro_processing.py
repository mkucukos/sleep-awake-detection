import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from utils.spectral_features import calculate_spectral_features
from utils.statistical_features import compute_rms, compute_rms_auc

# === Helper ===
def _resample_binary_mean(df_group, window_duration, col_name):
    """Resample a binary event column to per-window mean."""
    s = (
        df_group.set_index('datetime')[col_name]
        .dropna()
        .resample(f'{window_duration}S')
        .mean()
    )
    out = s.to_frame(name=f"{col_name.replace(' ', '_')}_mean")
    out['datetime'] = out.index
    return out


# ===========================
# Accelerometer & Gyroscope
# ===========================
def process_acc_gyro(df_acc_gyro, window_duration, fs):
    """
    Compute statistical and spectral features for accelerometer and gyroscope data.
    Resamples 'state', 'Obstructive Apnea', and 'Central Apnea' into windowed means.
    """
    results = []
    for subject_id, group in df_acc_gyro.groupby('subject_id'):
        print(f"Processing subject_id: {subject_id} in Accelerometer/Gyroscope Dataset...")
        group = group.copy()
        group['datetime'] = pd.to_datetime(group['datetime'])

        subject_results = {}
        skip_cols = {'datetime', 'state', 'sleep_state', 'subject_id', 'Obstructive Apnea', 'Central Apnea'}

        for feature in group.columns:
            if feature in skip_cols:
                continue

            non_nan_group = group.set_index('datetime')[feature].dropna()

            # Statistical features
            resampled = non_nan_group.resample(f'{window_duration}S').agg(
                ['mean', 'median', 'std', 'var', 'min', 'max']
            )
            resampled['range'] = resampled['max'] - resampled['min']

            # Shape features
            resampled['skew'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: skew(x.dropna()) if len(x) > 1 else np.nan
            )
            resampled['kurtosis'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: kurtosis(x.dropna()) if len(x) > 1 else np.nan
            )

            # Spectral features
            resampled['dominant_freq'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: calculate_spectral_features(x, fs)[0] if len(x.dropna()) > 1 else np.nan
            )
            resampled['spectral_entropy'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: calculate_spectral_features(x, fs)[1] if len(x.dropna()) > 1 else np.nan
            )

            # RMS features
            resampled['rms_max'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: compute_rms(x.dropna()) if len(x) > 0 else np.nan
            )
            resampled['rms_auc'] = non_nan_group.resample(f'{window_duration}S').apply(
                lambda x: compute_rms_auc(x.dropna(), fs) if len(x) > 0 else np.nan
            )

            resampled['datetime'] = resampled.index
            resampled.columns = [f"{feature}_{stat}" for stat in resampled.columns]
            subject_results[feature] = resampled

        # Binary event means
        for col in ['state', 'Obstructive Apnea', 'Central Apnea']:
            if col in group.columns:
                subject_results[col.replace(' ', '_')] = _resample_binary_mean(group, window_duration, col)

        subject_df = pd.concat(subject_results.values(), axis=1)
        subject_df['subject_id'] = subject_id
        results.append(subject_df.reset_index(drop=True))

    return pd.concat(results).reset_index(drop=True)
