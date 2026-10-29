"""Write small synthetic CSVs shaped like CIC-IDS2017 so the pipeline can be
tested in seconds before downloading the real dataset.

    python tools/make_fake_data.py --out data/fake_raw
    ids-prepare --raw data/fake_raw --out data/fake/flows.parquet

The numbers are random; results on this data mean nothing. It reproduces the
real files' quirks: leading spaces in headers, a repeated column, a constant
column, inf/NaN values, the broken dash in web-attack labels, and day names
in the file names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLS = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
        "Total Length of Fwd Packets", " Total Length of Bwd Packets", " Fwd Packet Length Mean",
        " Bwd Packet Length Mean", "Flow Bytes/s", " Flow Packets/s", " Flow IAT Mean", " Flow IAT Std",
        " Fwd IAT Mean", " Bwd IAT Mean", " SYN Flag Count", " RST Flag Count", " PSH Flag Count",
        " ACK Flag Count", " Down/Up Ratio", " Average Packet Size", "Init_Win_bytes_forward",
        " Init_Win_bytes_backward", " Active Mean", " Idle Mean", " Fwd Header Length",
        " Bwd PSH Flags", " Fwd Header Length"]  # repeated on purpose, like the real files

# label -> (feature indices to shift, log-scale shift)
ATTACKS = {
    "DoS Hulk": ([1, 8, 9, 19], 2.5),
    "DoS slowloris": ([1, 10, 22], 1.8),
    "DDoS": ([2, 4, 9], 2.2),
    "PortScan": ([2, 3, 14, 15], -2.0),
    "FTP-Patator": ([0, 6, 20], 1.5),
    "SSH-Patator": ([0, 6, 21], 1.2),
    "Web Attack \x96 Brute Force": ([7, 19], 0.9),
    "Web Attack \x96 XSS": ([7, 18], 0.8),
    "Bot": ([10, 11, 23], 0.6),
    "Infiltration": ([5], 0.3),  # deliberately subtle
}
FILES = {
    "Monday-WorkingHours.pcap_ISCX.csv": [],
    "Tuesday-WorkingHours.pcap_ISCX.csv": ["FTP-Patator", "SSH-Patator"],
    "Wednesday-workingHours.pcap_ISCX.csv": ["DoS Hulk", "DoS slowloris"],
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": ["Web Attack \x96 Brute Force", "Web Attack \x96 XSS", "Infiltration"],
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": ["DDoS", "PortScan", "Bot"],
}


def flows(rng, n, shift_idx=(), shift=0.0):
    d = len(COLS) - 1
    base = rng.normal(loc=np.linspace(1, 9, d), scale=1.0, size=(n, d))
    base[:, 14:18] = rng.integers(0, 2, size=(n, 4))  # flag counts
    for j in shift_idx:
        base[:, j] += shift
    X = np.expm1(np.abs(base)) * np.sign(base)
    X[:, 25] = 0  # Bwd PSH Flags: constant, as in the real data
    return np.column_stack([X, X[:, 24]])  # repeated Fwd Header Length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/fake_raw")
    ap.add_argument("--benign", type=int, default=6000, help="benign flows per file")
    ap.add_argument("--attack", type=int, default=300, help="flows per attack label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strength", type=float, default=2.0, help="multiplier on every attack's shift")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, attacks in FILES.items():
        parts, labels = [flows(rng, args.benign)], ["BENIGN"] * args.benign
        for a in attacks:
            idx, s = ATTACKS[a]
            parts.append(flows(rng, args.attack, idx, s * args.strength))
            labels += [a] * args.attack
        X = np.vstack(parts)
        df = pd.DataFrame(X, columns=COLS)
        df[" Label"] = labels
        bad = rng.choice(len(df), 20, replace=False)
        df.iloc[bad[:10], 8] = np.inf
        df.iloc[bad[10:], 9] = np.nan
        df = pd.concat([df, df.iloc[:25]])  # exact duplicates
        df.sample(frac=1, random_state=args.seed).to_csv(out / name, index=False, encoding="latin-1")
        print(f"wrote {out / name} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
