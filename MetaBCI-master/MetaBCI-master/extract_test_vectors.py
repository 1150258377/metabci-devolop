#!/usr/bin/env python3
"""
Extract two example 3-dim EEG feature vectors from DREAMER dataset that
current trained model classifies as 0 and 1 respectively.  The features
must match those used in the Streamlit demo:
    feat0 = AF3 band-power (8-13 Hz)
    feat1 = AF4 band-power (8-13 Hz)
    feat2 = |feat0 − feat1|  (alpha diff abs)
The script prints the vectors in µV so they can be pasted directly into
simulate_circuit_cli.py for hardware current calculation.

Run inside project root:
    python MetaBCI-master/extract_test_vectors.py
"""
import argparse
import numpy as np
import torch
from scipy import signal

try:
    from metabci.brainda.datasets.dreamer import DREAMER
except ImportError:
    raise SystemExit("MetaBCI not installed – activate venv or pip install -e .")

# ------------------------------------------------------------------
# CLI arguments
# ------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Extract EEG feature vectors that the trained/placeholder model classifies as each class.")
parser.add_argument("--subject", type=int, default=23, help="DREAMER subject ID (1-23)")
parser.add_argument("--n_per_class", "-n", type=int, default=5, help="Number of vectors to collect for each class")
parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save CSV of vectors (class,col0,col1,col2)")
args = parser.parse_args()

# ------------------------------------------------------------------
# Load SAME model object as Streamlit demo
# ------------------------------------------------------------------
try:
    from MetaBCI_master.demos.eeg_platform_streamlit import model, fs as model_fs, bandpass_filter
    streamlit_available = True
except Exception:
    streamlit_available = False

# ------------------------------------------------------------------
# Fallback: create a dummy logistic model when Streamlit demo weights
# are not available.  The logit is defined as (AF3_power − AF4_power)
# so that a positive difference maps to class-1 probability>0.5.
# This lets the script remain usable in standalone CLI mode.
# ------------------------------------------------------------------

if not streamlit_available:
    import types
    import torch

    def _dummy_model(x: torch.Tensor):
        """Return logit = feat0 − feat1 (shape: [N,1])."""
        if x.ndim == 1:
            x = x.unsqueeze(0)
        diff = x[:, 0] - x[:, 1]
        return diff.unsqueeze(1)

    model = _dummy_model  # type: ignore
    model_fs = 128  # default DREAMER EEG sampling rate

    # Re-export bandpass_filter compatible with demo signature
    def bandpass_filter(data: np.ndarray, fs: int, l_freq: float, h_freq: float):
        b, a = signal.butter(4, [l_freq, h_freq], fs=fs, btype="band")
        return signal.lfilter(b, a, data, axis=-1)

    print("[INFO] Streamlit model not found – using dummy logistic model (logit = AF3−AF4).")

# ------------------------------------------------------------------
# Helper: alpha band-power feature
# ------------------------------------------------------------------
alpha_low, alpha_high = 8, 13
channel_names = ['AF3','F7','F3','FC5','T7','P7','O1','O2','P8','T8','FC6','F4','F8','AF4']
idx_af3 = channel_names.index('AF3')
idx_af4 = channel_names.index('AF4')


def alpha_power(x: np.ndarray, sfreq: int):
    b, a = signal.butter(4, [alpha_low, alpha_high], fs=sfreq, btype="band")
    xf = signal.lfilter(b, a, x)
    return np.mean(xf ** 2)

# ------------------------------------------------------------------
# Load DREAMER subject 23 first session
# ------------------------------------------------------------------
print(f"Loading DREAMER subject {args.subject} …")
dream = DREAMER(subjects=[args.subject])
subj = dream._get_single_subject_data(args.subject)
raw0 = subj['session_1']['run_1']
raw0.load_data()
fs = int(raw0.info['sfreq'])
if fs != model_fs:
    print(f"[WARN] Dataset fs={fs} Hz ≠ model fs={model_fs} Hz; continue anyway.")

data = raw0.get_data()  # (14, n_times)
win_s = 1.0
win_len = int(fs * win_s)
num_win = data.shape[1] // win_len

# collect lists
vecs0, vecs1 = [], []

for w in range(num_win):
    start = w * win_len
    seg3 = data[idx_af3, start:start + win_len]
    seg4 = data[idx_af4, start:start + win_len]
    f0 = alpha_power(seg3, fs)
    f1 = alpha_power(seg4, fs)
    f2 = abs(f0 - f1)
    vec = np.array([f0, f1, f2], dtype=np.float32)

    with torch.no_grad():
        prob = torch.sigmoid(model(torch.tensor(vec).unsqueeze(0))).item()

    if prob <= 0.5 and len(vecs0) < args.n_per_class:
        vecs0.append(vec)
    elif prob > 0.5 and len(vecs1) < args.n_per_class:
        vecs1.append(vec)

    if len(vecs0) >= args.n_per_class and len(vecs1) >= args.n_per_class:
        break

print("\n=== Collected vectors (µV) ===")
for i, v in enumerate(vecs0):
    print(f"Class 0 #{i+1} →", v * 1e6)
for i, v in enumerate(vecs1):
    print(f"Class 1 #{i+1} →", v * 1e6)

if args.save_csv:
    import csv, os
    os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
    with open(args.save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "feat0", "feat1", "feat2"])
        for v in vecs0:
            writer.writerow([0, *(v * 1e6)])
        for v in vecs1:
            writer.writerow([1, *(v * 1e6)])
    print(f"Saved CSV to {args.save_csv}") 