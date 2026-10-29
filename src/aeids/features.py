"""Feature scaling fitted on benign training traffic only.

CIC-IDS2017 features are extremely heavy-tailed (byte counts, durations in
microseconds), and a few can be -1. A signed log, sign(x)*log1p(|x|), tames the
tails; then each column is standardised with the benign-train mean and std.
Test values are clipped to +/- `clip` standard deviations so one absurd flow
cannot dominate an autoencoder's reconstruction error.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class Preprocessor:
    def __init__(self, clip: float = 10.0, min_std: float = 1e-6):
        self.clip = clip
        self.min_std = min_std

    def fit(self, X: np.ndarray, columns: list[str]) -> "Preprocessor":
        Z = signed_log(X.astype(np.float64))
        std = Z.std(axis=0)
        self.keep_ = std > self.min_std  # columns constant in benign train carry no signal for the AE
        self.columns_ = [c for c, k in zip(columns, self.keep_) if k]
        self.dropped_ = [c for c, k in zip(columns, self.keep_) if not k]
        self.mean_ = Z.mean(axis=0)[self.keep_]
        self.std_ = std[self.keep_]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = signed_log(X[:, self.keep_].astype(np.float64))
        Z = (Z - self.mean_) / self.std_
        return np.clip(Z, -self.clip, self.clip).astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "Preprocessor":
        return joblib.load(path)
