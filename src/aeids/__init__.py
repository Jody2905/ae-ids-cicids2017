"""Autoencoder intrusion detection on CIC-IDS2017."""

META_COLS = ["label", "family", "day", "split", "is_attack"]

DEFAULT_PATHS = {
    "raw": "data/raw",
    "processed": "data/processed/flows.parquet",
    "models": "models",
    "results": "results",
}


def feature_columns(df):
    """Every column that is not metadata is a model feature."""
    return [c for c in df.columns if c not in META_COLS]


def set_threads(n: int) -> None:
    """Limit CPU threads so training can share the laptop with other jobs."""
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(n))
    try:
        import torch

        torch.set_num_threads(n)
    except ImportError:
        pass
