import numpy as np
from scipy.signal import welch

def calculate_spectral_features(signal, fs):
    """Return dominant frequency and spectral entropy."""
    clean_signal = signal.dropna()
    if len(clean_signal) < 2:
        return np.nan, np.nan

    nperseg = min(125, len(clean_signal))
    freqs, psd = welch(clean_signal, fs=fs, nperseg=nperseg)
    psd /= np.sum(psd)
    dominant_freq = freqs[np.argmax(psd)]
    spectral_entropy = -np.sum(psd * np.log(psd + 1e-10))
    return dominant_freq, spectral_entropy