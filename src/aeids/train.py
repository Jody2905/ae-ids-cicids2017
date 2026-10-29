"""ids-train: fit the preprocessor and the three detectors, then score the test set.

  ae   autoencoder, unsupervised: trained on benign flows only, score = reconstruction error
  if   Isolation Forest, unsupervised: trained on benign flows only
  rf   Random Forest, supervised: trained on labelled benign + attack flows (upper-bound baseline)

Unsupervised thresholds come from the benign validation scores (--percentile),
so the expected false-positive rate is set before any attack is seen.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import DEFAULT_PATHS, feature_columns, set_threads
from .features import Preprocessor


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_PATHS["processed"])
    ap.add_argument("--models", default="ae,if,rf", help="comma list from: ae, if, rf")
    ap.add_argument("--model-dir", default=DEFAULT_PATHS["models"])
    ap.add_argument("--out", default=DEFAULT_PATHS["results"])
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads (keep low while Ollama or another experiment is running)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="threshold = this percentile of benign validation scores (99 -> ~1%% expected FPR)")
    # autoencoder
    ap.add_argument("--hidden", default="64,32,8", help="encoder layer sizes; the last one is the bottleneck")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    # forests
    ap.add_argument("--if-trees", type=int, default=200)
    ap.add_argument("--rf-trees", type=int, default=100)
    ap.add_argument("--rf-max-train", type=int, default=500_000, help="subsample RF training rows (0 = all)")
    args = ap.parse_args(argv)

    set_threads(args.threads)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    out, mdir = Path(args.out), Path(args.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.data)
    feats = feature_columns(df)
    parts = {s: df[df["split"] == s] for s in ("train_benign", "val_benign", "sup_train", "test")}
    for s, p in parts.items():
        print(f"{s:13s} {len(p):>10,} rows")

    pre = Preprocessor().fit(parts["train_benign"][feats].to_numpy(), feats)
    pre.save(mdir / "preprocessor.joblib")
    if pre.dropped_:
        print(f"features constant in benign train (ignored): {pre.dropped_}")
    X = {s: pre.transform(p[feats].to_numpy()) for s, p in parts.items()}

    meta = ["label", "family", "day", "is_attack"]
    scores_test = parts["test"][meta].reset_index(drop=True).copy()
    scores_val = pd.DataFrame(index=range(len(parts["val_benign"])))
    thresholds: dict[str, float] = {}
    info: dict = {"args": vars(args), "n_features": len(pre.columns_), "models": {}}

    if "ae" in wanted:
        from .autoencoder import feature_errors, reconstruction_error, train_autoencoder
        import torch

        hidden = [int(h) for h in args.hidden.split(",")]
        print(f"\n[ae] training autoencoder {len(pre.columns_)} -> {hidden}")
        t0 = time.time()
        model, history = train_autoencoder(X["train_benign"], X["val_benign"], hidden, args.epochs,
                                           args.batch_size, args.lr, args.patience, args.seed)
        fit_s = time.time() - t0
        t0 = time.time()
        scores_test["score_ae"] = reconstruction_error(model, X["test"])
        score_s = time.time() - t0
        scores_val["score_ae"] = reconstruction_error(model, X["val_benign"])
        thresholds["ae"] = float(np.percentile(scores_val["score_ae"], args.percentile))
        torch.save({"state_dict": model.state_dict(), "n_features": len(pre.columns_), "hidden": hidden},
                   mdir / "autoencoder.pt")
        (out / "ae_history.json").write_text(json.dumps(history, indent=2))
        info["models"]["ae"] = {"fit_seconds": round(fit_s, 1), "score_seconds": round(score_s, 2),
                                "epochs_run": len(history),
                                "params": sum(p.numel() for p in model.parameters())}

        # Which features does the AE fail to reconstruct for each attack family?
        # Errors are relative to the same feature's average error on benign validation flows.
        fe_val = feature_errors(model, X["val_benign"]).mean(axis=0) + 1e-9
        fe_test = feature_errors(model, X["test"])
        rows = []
        for fam, idx in scores_test.groupby("family", observed=True).groups.items():
            ratio = fe_test[np.asarray(idx)].mean(axis=0) / fe_val
            for rank, j in enumerate(np.argsort(ratio)[::-1][:5], start=1):
                rows.append({"family": fam, "rank": rank, "feature": pre.columns_[j],
                             "error_vs_benign": round(float(ratio[j]), 2)})
        pd.DataFrame(rows).to_csv(out / "ae_top_features.csv", index=False)
        del fe_test

    if "if" in wanted:
        from sklearn.ensemble import IsolationForest

        print("\n[if] training Isolation Forest on benign flows")
        t0 = time.time()
        iso = IsolationForest(n_estimators=args.if_trees, random_state=args.seed, n_jobs=args.threads)
        iso.fit(X["train_benign"])
        fit_s = time.time() - t0
        t0 = time.time()
        scores_test["score_if"] = -iso.score_samples(X["test"])  # higher = more anomalous
        score_s = time.time() - t0
        scores_val["score_if"] = -iso.score_samples(X["val_benign"])
        thresholds["if"] = float(np.percentile(scores_val["score_if"], args.percentile))
        joblib.dump(iso, mdir / "isolation_forest.joblib")
        info["models"]["if"] = {"fit_seconds": round(fit_s, 1), "score_seconds": round(score_s, 2)}

    if "rf" in wanted:
        from sklearn.ensemble import RandomForestClassifier

        Xs = np.vstack([X["train_benign"], X["sup_train"]])
        ys = np.concatenate([np.zeros(len(X["train_benign"]), np.int8), parts["sup_train"]["is_attack"].to_numpy()])
        if args.rf_max_train and len(Xs) > args.rf_max_train:
            pick = np.random.default_rng(args.seed).choice(len(Xs), args.rf_max_train, replace=False)
            Xs, ys = Xs[pick], ys[pick]
        print(f"\n[rf] training Random Forest on {len(Xs):,} labelled flows ({ys.mean():.1%} attacks)")
        t0 = time.time()
        rf = RandomForestClassifier(n_estimators=args.rf_trees, n_jobs=args.threads, random_state=args.seed,
                                    class_weight="balanced_subsample", min_samples_leaf=2)
        rf.fit(Xs, ys)
        fit_s = time.time() - t0
        t0 = time.time()
        scores_test["score_rf"] = rf.predict_proba(X["test"])[:, 1]
        score_s = time.time() - t0
        scores_val["score_rf"] = rf.predict_proba(X["val_benign"])[:, 1]
        thresholds["rf"] = 0.5
        joblib.dump(rf, mdir / "random_forest.joblib")
        importances = sorted(zip(pre.columns_, rf.feature_importances_), key=lambda t: -t[1])[:15]
        info["models"]["rf"] = {"fit_seconds": round(fit_s, 1), "score_seconds": round(score_s, 2),
                                "train_rows": int(len(Xs)),
                                "top_features": [[f, round(float(v), 4)] for f, v in importances]}
        del Xs

    scores_test.to_parquet(out / "scores_test.parquet", index=False)
    scores_val.to_parquet(out / "scores_val.parquet", index=False)
    (out / "thresholds.json").write_text(json.dumps(
        {"percentile": args.percentile, "thresholds": thresholds}, indent=2))
    (out / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"\nscores and thresholds written to {out}/  ->  next: ids-evaluate")


if __name__ == "__main__":
    main()
