import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from utils.led_feature_extraction import process_led_features

def process_led(df_led, window_duration, original_fs, target_fs, lowcut, highcut, order):
    """Process LED dataset to compute windowed HR, HRV, and spectral features."""
    results = []
    for subject_id, group in df_led.groupby('subject_id'):
        print(f"Processing subject_id: {subject_id} in LED Dataset...")
        group = group.copy()
        group['datetime'] = pd.to_datetime(group['datetime'])
        group.set_index('datetime', inplace=True)

        resampled = group['ledGreen'].resample(f'{window_duration}S')
        windows = [window.dropna().values for _, window in resampled]
        datetime_range = resampled.size().index

        if len(windows) == 0:
            continue

        ideal_window_size = original_fs * window_duration
        min_window_size = int(ideal_window_size / 1.5)

        state_resampled = group['state'].resample(f'{window_duration}S').mean() if 'state' in group.columns else None
        oa_resampled = group['Obstructive Apnea'].resample(f'{window_duration}S').mean() if 'Obstructive Apnea' in group.columns else None
        ca_resampled = group['Central Apnea'].resample(f'{window_duration}S').mean() if 'Central Apnea' in group.columns else None

        feature_list = []
        for i, window in enumerate(windows):
            if len(window) < min_window_size:
                continue

            stats = {
                'ledGreen_skew': skew(window) if len(window) > 1 else np.nan,
                'ledGreen_kurtosis': kurtosis(window) if len(window) > 1 else np.nan,
            }

            try:
                advanced_features_df = process_led_features(
                    signal=window,
                    original_fs=original_fs,
                    target_fs=target_fs,
                    lowcut=lowcut,
                    highcut=highcut,
                    order=order,
                    window_duration=window_duration
                )
                if not advanced_features_df.empty:
                    stats.update(advanced_features_df.iloc[0].to_dict())
            except Exception as e:
                print(f"Advanced feature calculation failed for window {i}: {e}")
                continue

            stats['datetime'] = datetime_range[i]
            stats['subject_id'] = subject_id

            if state_resampled is not None and i < len(state_resampled):
                stats['state_mean'] = state_resampled.iloc[i]
            if oa_resampled is not None and i < len(oa_resampled):
                stats['Obstructive_Apnea_mean'] = oa_resampled.iloc[i]
            if ca_resampled is not None and i < len(ca_resampled):
                stats['Central_Apnea_mean'] = ca_resampled.iloc[i]

            feature_list.append(stats)

        results.append(pd.DataFrame(feature_list))

    return pd.concat(results, ignore_index=True)

