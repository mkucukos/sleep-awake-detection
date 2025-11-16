import pandas as pd
import numpy as np
from .acc_gyro_processing import _resample_binary_mean

def process_temp(df_temp, window_duration):
    """Process Temperature dataset with statistical metrics and event means."""
    results = []
    for subject_id, group in df_temp.groupby('subject_id'):
        print(f"Processing subject_id: {subject_id} in Temperature Dataset...")
        group = group.copy()
        group['datetime'] = pd.to_datetime(group['datetime'])

        subject_results = {}
        skip_cols = {'datetime', 'state', 'sleep_state', 'subject_id', 'Obstructive Apnea', 'Central Apnea'}

        for feature in group.columns:
            if feature in skip_cols:
                continue

            non_nan_group = group.set_index('datetime')[feature].dropna()
            resampled = non_nan_group.resample(f'{window_duration}S').agg(['mean', 'median', 'std', 'var', 'min', 'max'])
            resampled['range'] = resampled['max'] - resampled['min']
            resampled.columns = [f"{feature}_{stat}" for stat in resampled.columns]
            resampled['datetime'] = resampled.index
            subject_results[feature] = resampled

        for col in ['state', 'Obstructive Apnea', 'Central Apnea']:
            if col in group.columns:
                subject_results[col.replace(' ', '_')] = _resample_binary_mean(group, window_duration, col)

        subject_df = pd.concat(subject_results.values(), axis=1)
        subject_df['subject_id'] = subject_id
        results.append(subject_df.reset_index(drop=True))

    return pd.concat(results).reset_index(drop=True)
