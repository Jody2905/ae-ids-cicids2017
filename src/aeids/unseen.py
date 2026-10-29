"""ids-unseen: how does a supervised model do on an attack family it never saw?

For each attack family F, retrain the Random Forest with every F flow removed
from its training data, then measure how many test flows of F it still flags.
The autoencoder and Isolation Forest never see attacks at all, so their F
detection rates (from ids-train) are directly comparable. This is the core
argument for anomaly detection: catching attacks nobody has labelled yet.

Writes results/unseen.csv, results/unseen.md and results/figures/unseen_families.png.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import DEFAULT_PATHS, feature_columns, set_threads
from .evaluate import md_table
from .features import Preprocessor
from .plots import MODEL_COLORS, MODEL_NAMES, style


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_PATHS["processed"])
    ap.add_argument("--model-dir", default=DEFAULT_PATHS["models"])
    ap.add_argument("--results", default=DEFAULT_PATHS["results"])
    ap.add_argument("--families", default="", help="comma list; default = every attack family with >= --min-test flows")
    ap.add_argument("--min-test", type=int, default=20)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-max-train", type=int, default=300_000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    from sklearn.ensemble import RandomForestClassifier

    set_threads(args.threads)
    res = Path(args.results)
    df = pd.read_parquet(args.data)
    feats = feature_columns(df)
    pre = Preprocessor.load(Path(args.model_dir) / "preprocessor.joblib")
    scores = pd.read_parquet(res / "scores_test.parquet")
    thr = json.loads((res / "thresholds.json").read_text())["thresholds"]

    train = df[df["split"].isin(["train_benign", "sup_train"])]
    test = df[df["split"] == "test"]
    Xtrain = pre.transform(train[feats].to_numpy())
    ytrain = train["is_attack"].to_numpy()
    fam_train = train["family"].astype(str).to_numpy()
    Xtest = pre.transform(test[feats].to_numpy())
    fam_test = test["family"].astype(str).to_numpy()
    benign_test = fam_test == "Benign"

    counts = pd.Series(fam_test[~benign_test]).value_counts()
    fams = [f.strip() for f in args.families.split(",") if f.strip()] or \
        counts[counts >= args.min_test].index.tolist()

    rng = np.random.default_rng(args.seed)
    rows = []
    for fam in fams:
        keep = np.flatnonzero(fam_train != fam)
        if args.rf_max_train and len(keep) > args.rf_max_train:
            keep = rng.choice(keep, args.rf_max_train, replace=False)
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=args.rf_trees, n_jobs=args.threads, random_state=args.seed,
                                    class_weight="balanced_subsample", min_samples_leaf=2)
        rf.fit(Xtrain[keep], ytrain[keep])
        is_f = fam_test == fam
        unseen_rate = float((rf.predict_proba(Xtest[is_f])[:, 1] > 0.5).mean())
        fpr = float((rf.predict_proba(Xtest[benign_test])[:, 1] > 0.5).mean())
        row = {"family": fam, "n": int(is_f.sum()), "rf_unseen": unseen_rate, "rf_unseen_fpr": fpr}
        g = scores[scores["family"].astype(str) == fam]
        for m in ("rf", "ae", "if"):
            if f"score_{m}" in g.columns:
                row[m] = float((g[f"score_{m}"] > thr[m]).mean())
        rows.append(row)
        print(f"{fam:14s} n={row['n']:>7,}  RF seen {row.get('rf', float('nan')):.1%}  "
              f"RF unseen {unseen_rate:.1%}  AE {row.get('ae', float('nan')):.1%}  "
              f"IF {row.get('if', float('nan')):.1%}  ({time.time() - t0:.0f}s)", flush=True)

    out = pd.DataFrame(rows)
    cols = ["family", "n"] + [c for c in ("rf", "rf_unseen", "ae", "if") if c in out.columns] + ["rf_unseen_fpr"]
    out = out[cols]
    out.to_csv(res / "unseen.csv", index=False)
    pct = [c for c in cols if c not in ("family", "n")]
    (res / "unseen.md").write_text(
        "# Detection of attack families the Random Forest never saw\n\n"
        "rf = trained with the family; rf_unseen = retrained without it; "
        "rf_unseen_fpr = that retrained model's false-positive rate on benign test flows. "
        "The autoencoder and Isolation Forest never see any attacks.\n\n"
        + md_table(out.rename(columns={"rf_unseen_fpr": "RF unseen FPR"}), pct + ["RF unseen FPR"]) + "\n")

    plt = style()
    series = [c for c in ("ae", "if", "rf", "rf_unseen") if c in out.columns]
    o = out.sort_values("ae" if "ae" in out.columns else "rf").reset_index(drop=True)
    y = np.arange(len(o))
    h = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(o) * len(series) + 1.6))
    for i, m in enumerate(series):
        pos = y + ((len(series) - 1) / 2 - i) * h
        ax.barh(pos, 100 * o[m], height=h * 0.85, color=MODEL_COLORS[m], label=MODEL_NAMES[m], edgecolor="none")
        if m == "rf_unseen":  # the headline number; label it so a 0% bar is still visible
            for yy, v in zip(pos, o[m]):
                ax.text(100 * v + 1, yy, f"{100 * v:.0f}%", va="center", fontsize=8, color="#52514e")
    ax.set_yticks(y, o["family"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Flows detected (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Detecting attack families the supervised model never saw")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    (res / "figures").mkdir(exist_ok=True)
    fig.savefig(res / "figures" / "unseen_families.png")
    plt.close(fig)
    print(f"\nwrote {res}/unseen.csv, unseen.md and figures/unseen_families.png")


if __name__ == "__main__":
    main()
