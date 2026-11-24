"""R–R features from the original comparator; extracted unchanged."""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, sosfiltfilt
FS = 300
SOS = butter(4, [5, 25], btype="bandpass", fs=FS, output="sos")
FEATURES = ("rate_per_min", "mean_rr", "sd_rr", "rmssd", "pnn50", "cv_rr", "median_rr", "rr_p90_minus_p10")

def extract(path: Path) -> tuple[list[float], int]:
    raw = np.asarray(loadmat(path)["val"], dtype=np.float64).reshape(-1)
    if not np.isfinite(raw).all() or len(raw) < 9 * FS:
        return [float("nan")] * len(FEATURES), 0
    filtered = sosfiltfilt(SOS, raw)
    derivative = np.abs(np.diff(filtered, prepend=filtered[0]))
    envelope = np.convolve(derivative, np.ones(45) / 45, mode="same")
    prominence = max(1e-12, 0.25 * float(np.quantile(envelope, 0.95) - np.quantile(envelope, 0.50)))
    peaks, _ = find_peaks(envelope, distance=75, prominence=prominence)
    rr = np.diff(peaks).astype(np.float64) / FS
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    if len(rr) < 3:
        return [float("nan")] * len(FEATURES), len(peaks)
    mean = float(np.mean(rr))
    diffs = np.diff(rr)
    return [
        float(len(peaks) / (len(raw) / FS / 60)), mean, float(np.std(rr, ddof=1)),
        float(np.sqrt(np.mean(diffs ** 2))), float(np.mean(np.abs(diffs) > 0.05)),
        float(np.std(rr, ddof=1) / mean), float(np.median(rr)),
        float(np.quantile(rr, 0.90) - np.quantile(rr, 0.10)),
    ], len(peaks)
