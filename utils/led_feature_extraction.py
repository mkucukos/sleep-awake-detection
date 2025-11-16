import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.interpolate import interp1d
from .filters import butter_bandpass_filter

# Process LED-related features
def process_led_features(signal, original_fs, target_fs, lowcut, highcut, order, window_duration):
    """
    Process LED-related features from a given signal, including HRV-related metrics using nk.hrv.
    """
    if len(signal) < (original_fs * window_duration) / 1.5:  # Ensure signal is large enough for meaningful processing
        print("Signal too short for processing.")
        return pd.DataFrame()

    try:
        # Upsample using linear interpolation
        time_original = np.arange(len(signal)) / original_fs
        num_samples = int(len(signal) * (target_fs / original_fs))
        time_upsampled = np.linspace(0, time_original[-1], num=num_samples)
        interpolator = interp1d(time_original, signal, kind="linear")
        signal_upsampled = interpolator(time_upsampled)

        num_samples_per_window = int(target_fs * window_duration)
        windows = [
            signal_upsampled[i:i + num_samples_per_window]
            for i in range(0, len(signal_upsampled), num_samples_per_window)
        ]

        led_metrics_list = []

        for window in windows:
            # Bandpass filter the window
            filtered_signal = butter_bandpass_filter(window, lowcut, highcut, target_fs, order)

            # Clean PPG signal and detect peaks
            ppg_cleaned = nk.ppg_clean(filtered_signal, sampling_rate=target_fs)
            peaks_info = nk.ppg_findpeaks(ppg_cleaned, sampling_rate=target_fs)
            filtered_peaks = peaks_info["PPG_Peaks"]

            # Check the number of peaks
            num_peaks = len(filtered_peaks)
            if num_peaks < 15:  # Minimum number of peaks for robust HRV calculation
                print(f"Skipping HRV calculation due to insufficient peaks: {num_peaks}")
                continue

            # Calculate heart rate
            heart_rate = nk.ppg_rate(peaks_info, sampling_rate=target_fs, desired_length=len(ppg_cleaned))
            min_hr = np.min(heart_rate) if len(heart_rate) > 0 else np.nan
            max_hr = np.max(heart_rate) if len(heart_rate) > 0 else np.nan
            mean_hr = np.mean(heart_rate) if len(heart_rate) > 0 else np.nan

            # Use nk.hrv to calculate HRV metrics
            try:
                hrv_indices = nk.hrv(peaks=filtered_peaks, sampling_rate=target_fs, show=False)
                hrv_metrics = hrv_indices.iloc[0].to_dict()  # Convert the HRV metrics to a dictionary
            except Exception as e:
                print(f"HRV calculation error: {e}")
                hrv_metrics = {}

            # Add heart rate and HRV stats to the metrics list
            metrics = {
                "Min_HR": min_hr,
                "Max_HR": max_hr,
                "Mean_HR": mean_hr,
                **hrv_metrics,  # Add HRV metrics dynamically
            }
            led_metrics_list.append(metrics)

        if not led_metrics_list:
            print("No valid windows for feature calculation. Returning empty DataFrame.")
            return pd.DataFrame()

        return pd.DataFrame(led_metrics_list)

    except Exception as e:
        print(f"Error in process_led_features: {e}")
        return pd.DataFrame()