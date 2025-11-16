import numpy as np

def compute_rms(signal):
    """Compute Root Mean Square (RMS) of a signal."""
    return np.sqrt(np.mean(np.square(signal)))

def compute_rms_auc(signal, sampling_rate):
    """Compute RMS Area Under Curve (AUC)."""
    time = np.linspace(0, len(signal) / sampling_rate, len(signal))
    return np.trapz([compute_rms(signal)] * len(signal), time)
