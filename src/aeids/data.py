"""ids-prepare: load, clean and split the CIC-IDS2017 CSV files into one parquet file.

Split design (no attack data ever reaches the unsupervised models):
  train_benign  benign flows from the training day (Monday), 80%  -> fits AE / Isolation Forest
  val_benign    benign flows from the training day, 20%           -> early stopping + thresholds
  sup_train     50% of Tuesday-Friday, stratified by label        -> Random Forest only
  test          the other 50% of Tuesday-Friday                   -> every model is scored here
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import DEFAULT_PATHS

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]

# Ordered rules: first match wins. Applied to the cleaned, lower-cased label.
FAMILY_RULES = [
    (r"^benign$", "Benign"),
    (r"^ddos", "DDoS"),
    (r"^dos", "DoS"),
    (r"port\s?scan", "PortScan"),
    (r"patator", "BruteForce"),
    (r"^web attack", "WebAttack"),
    (r"infiltration", "Infiltration"),
    (r"^bot", "Bot"),
    (r"heartbleed", "Heartbleed"),
]

# Identifier-like columns: they describe *which* hosts talked, not *how*,
# and let a model memorise the testbed instead of learning traffic behaviour.
ID_COLS = {
    "flow_id", "source_ip", "src_ip", "destination_ip", "dst_ip",
    "timestamp", "source_port", "src_port", "simillarhttp", "attempted_category",
}
PORT_COLS = {"destination_port", "dst_port"}


def norm_col(name: str) -> str:
    name = name.strip().lower()
    return re.sub(r"[^0-9a-z]+", "_", name).strip("_")


def clean_label(raw: object) -> str:
    s = str(raw).strip()
    # The original CSVs store "Web Attack – Brute Force" with a broken dash.
    # Read as latin-1, the UTF-8 replacement character arrives as "\u00ef\u00bf\u00bd".
    for bad in ("\u00ef\u00bf\u00bd", "\ufffd", "\x96", "\u2013", "\u2014"):
        s = s.replace(bad, "-")
    return re.sub(r"\s+", " ", s)


def family_of(label: str) -> str:
    low = label.lower()
    if "attempted" in low:
        return "Attempted"
    for pattern, fam in FAMILY_RULES:
        if re.search(pattern, low):
            return fam
    return "Other"


def day_of(path: Path) -> str:
    low = path.name.lower()
    for d in DAYS:
        if d in low:
            return d
    return "unknown"


def load_file(path: Path, keep_port: bool, log: dict) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df.columns = [norm_col(c) for c in df.columns]
    if "label" not in df.columns:
        raise ValueError(f"{path.name}: no 'Label' column found")

    # pandas renames a repeated header to "<name>.1" -> "<name>_1"; CIC-IDS2017
    # repeats "Fwd Header Length", so drop the copy.
    dupes = [c for c in df.columns if c.endswith("_1") and c[:-2] in df.columns]
    drop = set(dupes) | (ID_COLS & set(df.columns))
    if not keep_port:
        drop |= PORT_COLS & set(df.columns)
    df = df.drop(columns=sorted(drop))

    df["label"] = df["label"].map(clean_label)
    df = df[df["label"].str.lower() != "nan"]  # blank separator rows in some files

    feats = [c for c in df.columns if c != "label"]
    for c in feats:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    arr = df[feats].to_numpy(dtype=np.float64)
    finite = np.isfinite(arr).all(axis=1)
    too_big = (np.abs(np.where(np.isfinite(arr), arr, 0)) > np.finfo(np.float32).max).any(axis=1)
    keep = finite & ~too_big

    out = pd.DataFrame(arr[keep].astype(np.float32), columns=feats)
    out["label"] = df["label"].to_numpy()[keep]
    out["day"] = day_of(path)
    log["files"].append({
        "file": path.name, "day": day_of(path), "rows_read": int(len(df)),
        "rows_dropped_nan_inf": int((~keep).sum()), "columns_dropped": sorted(drop),
    })
    return out


def split(df: pd.DataFrame, train_day: str, benign_frac: float, train_max: int,
          seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    benign = df["family"] == "Benign"
    on_train_day = df["day"] == train_day
    if not on_train_day.any():
        raise SystemExit(f"No rows from training day '{train_day}'. Check the file names or pass --train-day.")

    pool = df[on_train_day & benign]
    if train_max and len(pool) > train_max:
        pool = pool.sample(n=train_max, random_state=seed)
    tr_idx, va_idx = train_test_split(pool.index, test_size=0.2, random_state=seed)

    rest = df[~(on_train_day & benign)]
    if benign_frac < 1.0:
        rb = rest.index[rest["family"] == "Benign"]
        keep_b = rng.choice(rb, size=int(len(rb) * benign_frac), replace=False)
        rest = rest.loc[np.concatenate([rest.index[rest["family"] != "Benign"], keep_b])]

    counts = rest["label"].value_counts()
    rare = rest["label"].isin(counts[counts < 2].index)
    sup_idx, test_idx = train_test_split(
        rest.index[~rare], test_size=0.5, random_state=seed, stratify=rest.loc[~rare, "label"]
    )
    test_idx = np.concatenate([test_idx, rest.index[rare]])  # singletons go to test

    df = df.copy()
    df["split"] = ""
    df.loc[tr_idx, "split"] = "train_benign"
    df.loc[va_idx, "split"] = "val_benign"
    df.loc[sup_idx, "split"] = "sup_train"
    df.loc[test_idx, "split"] = "test"
    df = df[df["split"] != ""]
    df["split"] = df["split"].astype("category")
    return df


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=DEFAULT_PATHS["raw"], help="folder with the CIC-IDS2017 CSV files")
    ap.add_argument("--out", default=DEFAULT_PATHS["processed"])
    ap.add_argument("--train-day", default="monday", choices=DAYS)
    ap.add_argument("--benign-frac", type=float, default=1.0,
                    help="keep this fraction of Tue-Fri benign flows (e.g. 0.3 on an 8 GB laptop)")
    ap.add_argument("--train-max", type=int, default=0, help="cap on training-day benign rows (0 = no cap)")
    ap.add_argument("--attempted", choices=["drop", "benign", "attack"], default="drop",
                    help="corrected-dataset '- Attempted' flows: drop them, call them benign, or keep as attacks")
    ap.add_argument("--keep-port", action="store_true", help="keep Destination Port as a feature")
    ap.add_argument("--no-dedupe", action="store_true", help="keep exact duplicate flows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    files = sorted(Path(args.raw).glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {args.raw}. See README: download MachineLearningCSV.zip and unzip it there.")

    log: dict = {"files": []}
    frames = []
    for f in files:
        print(f"reading {f.name} ...", flush=True)
        frames.append(load_file(f, args.keep_port, log))
    df = pd.concat(frames, join="inner", ignore_index=True)
    del frames
    log["rows_after_cleaning"] = int(len(df))

    df["family"] = df["label"].map(family_of)
    if (df["family"] == "Attempted").any():
        if args.attempted == "drop":
            df = df[df["family"] != "Attempted"]
        elif args.attempted == "benign":
            df.loc[df["family"] == "Attempted", "family"] = "Benign"
    log["unmapped_labels"] = sorted(df.loc[df["family"] == "Other", "label"].unique().tolist())

    feats = [c for c in df.columns if c not in ("label", "family", "day")]
    if not args.no_dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=feats + ["label"]).reset_index(drop=True)
        log["duplicates_removed"] = int(before - len(df))

    const = [c for c in feats if df[c].min() == df[c].max()]
    df = df.drop(columns=const)
    log["constant_columns_dropped"] = const

    df["is_attack"] = (df["family"] != "Benign").astype(np.int8)
    df = split(df, args.train_day, args.benign_frac, args.train_max, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for c in ("label", "family", "day"):
        df[c] = df[c].astype("category")
    df.to_parquet(out, index=False)

    log["n_features"] = len([c for c in df.columns if c not in ("label", "family", "day", "split", "is_attack")])
    log["rows_per_split"] = {str(k): int(v) for k, v in df["split"].value_counts().items()}
    log["labels_per_split"] = {
        str(s): {str(k): int(v) for k, v in g["label"].value_counts().items() if v}
        for s, g in df.groupby("split", observed=True)
    }
    log["args"] = vars(args)
    (out.parent / "prep_report.json").write_text(json.dumps(log, indent=2))

    print(f"\nwrote {out}: {len(df):,} flows, {log['n_features']} features")
    for s, n in log["rows_per_split"].items():
        print(f"  {s:13s} {n:>10,}")
    if log["unmapped_labels"]:
        print(f"  WARNING: labels with no family rule -> 'Other': {log['unmapped_labels']}")


if __name__ == "__main__":
    main()
