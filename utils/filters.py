# Butterworth bandpass filter
import numpy as np
from scipy.signal import butter, filtfilt
import pandas as pd

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# Define the high-pass filter
def high_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def filter_days_with_less_than_10_rows(df):
    """
    Filters the given DataFrame to remove rows from days that have fewer than 10 rows.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing at least a 'unixTimes' column.

    Returns:
    --------
    pandas.DataFrame
        A filtered DataFrame with rows from days that have at least 10 rows.
    """
    if 'unixTimes' not in df.columns:
        raise ValueError("The DataFrame does not have a 'unixTimes' column.")
    
    # Convert 'unixTimes' to datetime
    df['time'] = pd.to_datetime(df['unixTimes'], unit='ms', errors='coerce')
    
    # Extract the date from the 'time' column
    df['date'] = df['time'].dt.date
    
    # Group by date and count the number of rows for each day
    day_counts = df.groupby('date').size().reset_index(name='row_count')
    
    # Identify the days with at least 10 rows
    valid_days = day_counts[day_counts['row_count'] >= 10]['date']
    
    # Filter the DataFrame to include only rows from valid days
    filtered_df = df[df['date'].isin(valid_days)].drop(columns=['date', 'time'])
    
    return filtered_df