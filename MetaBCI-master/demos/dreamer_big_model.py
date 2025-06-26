# -*- coding: utf-8 -*-
"""Simple demo showcasing how to replace the default SVM with a
PyTorch neural network or any custom model.

This script loads the DREAMER dataset via MetaBCI, extracts basic
bandpower features and then trains or loads a model for emotion
classification.  You can pass in a path to a saved ``.pt`` file to
load your own fine-tuned model.  Alternatively, features can be sent
to a remote API endpoint for inference.
"""

import argparse
import os
from typing import Optional, Sequence

import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import requests

from metabci.brainda.datasets.dreamer import DREAMER
from metabci.brainda.algorithms.feature_analysis.freq_analysis import bandpower


def extract_bandpower_features(raw) -> np.ndarray:
    """Extract theta/alpha/beta/gamma band power for one Raw object."""
    raw.filter(l_freq=1, h_freq=45, verbose=False)
    data = raw.get_data()
    bands = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
    }
    feats = []
    for (start, dur, desc) in zip(raw.annotations.onset,
                                  raw.annotations.duration,
                                  raw.annotations.description):
        start_idx = int(start * raw.info["sfreq"])
        end_idx = start_idx + int(dur * raw.info["sfreq"])
        seg = data[:, start_idx:end_idx]
        vec = []
        for fmin, fmax in bands.values():
            vec.extend(bandpower(seg, raw.info["sfreq"], fmin, fmax))
        feats.append(vec)
    return np.array(feats, dtype=np.float32)


class SimpleNet(nn.Module):
    """Two-layer MLP used when no external model is given."""

    def __init__(self, in_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def remote_predict(url: str, api_key: Optional[str], features: Sequence[Sequence[float]]) -> np.ndarray:
    """Send features to a remote API and return predictions."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"features": features}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    preds = data.get("predictions")
    if preds is None:
        raise ValueError("Response JSON must contain 'predictions' field")
    return np.array(preds, dtype=float)


def load_model(in_dim: int, ckpt: Optional[str]) -> nn.Module:
    """Load a custom model if ``ckpt`` provided, else create SimpleNet."""
    if ckpt and os.path.isfile(ckpt):
        model = torch.load(ckpt, map_location="cpu")
        return model
    return SimpleNet(in_dim)


def main(args: argparse.Namespace) -> None:
    ds = DREAMER()
    subj_id = args.subject
    data = ds._get_single_subject_data(subj_id)

    X_list, y_list = [], []
    for sess in data.values():
        for raw in sess.values():
            feats = extract_bandpower_features(raw)
            labels = [1 if float(d) > 3 else 0 for d in raw.annotations.description]
            X_list.append(feats)
            y_list.extend(labels)
    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    in_dim = X_train.shape[1]
    if args.remote_url:
        preds = remote_predict(args.remote_url, args.api_key, X_test.tolist())
        acc = (preds == y_test).mean()
        print(f"Remote model accuracy: {acc:.3f}")
    else:
        model = load_model(in_dim, args.model)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        if not args.model:
            model.train()
            for _ in range(args.epochs):
                optimizer.zero_grad()
                out = model(X_train_t)
                loss = criterion(out, y_train_t)
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            model.eval()
            X_test_t = torch.tensor(X_test, dtype=torch.float32)
            logits = model(X_test_t).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).float().numpy()
            acc = (preds == y_test).mean()
            print(f"Accuracy: {acc:.3f}")

        if args.save_path and not args.model:
            torch.save(model, args.save_path)
            print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DREAMER big model demo")
    parser.add_argument("--subject", type=int, default=1, help="subject id")
    parser.add_argument(
        "--model", type=str, default=None, help="load external .pt model"
    )
    parser.add_argument(
        "--save-path", type=str, default=None, help="path to save trained model"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--remote-url", type=str, default=None, help="URL of remote model API"
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="API key for remote model"
    )
    main(parser.parse_args())

