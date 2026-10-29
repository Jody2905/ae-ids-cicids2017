"""ids-evaluate: metrics, per-attack detection rates, threshold sweep and figures.

Reads results/scores_test.parquet, scores_val.parquet and thresholds.json
written by ids-train, and writes:
  metrics.json, summary.md            overall precision / recall / F1 / FPR / ROC-AUC / PR-AUC
  per_family.csv, per_label.csv       detection rate per attack family and per original label
  threshold_sweep.csv                 unsupervised models at several benign-validation percentiles
  figures/*.png                       per-family detection, ROC (log FPR), AE error by family, AE training
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from . import DEFAULT_PATHS
from .plots import MODEL_COLORS, MODEL_NAMES, style

SWEEP = [90, 95, 97.5, 99, 99.5, 99.9]


def binary_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score > thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum()); tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": thr, "precision": precision, "recall": recall, "f1": f1,
            "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "roc_auc": float(roc_auc_score(y, score)), "pr_auc": float(average_precision_score(y, score)),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def detection_table(scores: pd.DataFrame, models: list[str], thr: dict, by: str) -> pd.DataFrame:
    rows = []
    for key, g in scores.groupby(by, observed=True):
        row = {by: key, "n": len(g)}
        for m in models:
            row[m] = float((g[f"score_{m}"] > thr[m]).mean())
        rows.append(row)
    t = pd.DataFrame(rows)
    # Benign rows report the false-positive rate; attack rows the detection rate (recall).
    order = "ae" if "ae" in models else models[0]
    return t.sort_values(by=[order], ascending=False).reset_index(drop=True)


def md_table(df: pd.DataFrame, pct_cols: list[str]) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(MODEL_NAMES.get(c, c) for c in cols) + " |",
             "|" + "|".join("---:" if c in pct_cols or c == "n" else "---" for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c in pct_cols:
                cells.append(f"{100 * v:.1f}%")
            elif c == "n":
                cells.append(f"{int(v):,}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_family(fam: pd.DataFrame, models: list[str], path: Path) -> None:
    plt = style()
    attacks = fam[fam["family"] != "Benign"].iloc[::-1]
    n = len(attacks)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n * len(models) / 2 + 1.6))
    h = 0.8 / len(models)
    y = np.arange(n)
    for i, m in enumerate(models):
        ax.barh(y + ((len(models) - 1) / 2 - i) * h, 100 * attacks[m], height=h * 0.85,
                color=MODEL_COLORS[m], label=MODEL_NAMES[m], edgecolor="none")
    ax.set_yticks(y, [f"{f}  (n={k:,})" for f, k in zip(attacks["family"], attacks["n"])])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Flows detected (%)")
    ax.grid(axis="y", visible=False)
    benign = fam[fam["family"] == "Benign"]
    fpr = ", ".join(f"{MODEL_NAMES[m]} {100 * benign[m].iloc[0]:.2f}%" for m in models) if len(benign) else ""
    ax.set_title("Detection rate by attack family", pad=26 if fpr else 6)
    if fpr:
        ax.text(0, 1.015, f"False-positive rate on benign test flows: {fpr}", transform=ax.transAxes,
                fontsize=8.5, color="#52514e", va="bottom")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(models))
    fig.savefig(path)
    plt.close(fig)


def plot_roc(scores: pd.DataFrame, models: list[str], metrics: dict, path: Path) -> None:
    plt = style()
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    y = scores["is_attack"].to_numpy()
    for m in models:
        fpr, tpr, _ = roc_curve(y, scores[f"score_{m}"])
        ax.plot(np.clip(fpr, 1e-5, 1), tpr, color=MODEL_COLORS[m], lw=2,
                label=f"{MODEL_NAMES[m]} (AUC {metrics[m]['roc_auc']:.3f})")
        op = metrics[m]
        ax.plot(max(op["fpr"], 1e-5), op["recall"], "o", ms=8, color=MODEL_COLORS[m],
                markeredgecolor="#fcfcfb", markeredgewidth=2)
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False-positive rate (log scale)")
    ax.set_ylabel("Detection rate")
    ax.set_title("ROC curves (dots = chosen operating threshold)")
    ax.legend(loc="lower right")
    fig.savefig(path)
    plt.close(fig)


def plot_ae_errors(scores: pd.DataFrame, thr: float, path: Path) -> None:
    plt = style()
    fams = scores.groupby("family", observed=True)["score_ae"].median().sort_values().index.tolist()
    data = [np.log10(scores.loc[scores["family"] == f, "score_ae"].to_numpy() + 1e-9) for f in fams]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(fams) + 1.4))
    ax.boxplot(data, vert=False, widths=0.55, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#cfe0f6", edgecolor=MODEL_COLORS["ae"], linewidth=1.2),
               medianprops=dict(color=MODEL_COLORS["ae"], linewidth=2),
               whiskerprops=dict(color=MODEL_COLORS["ae"]), capprops=dict(color=MODEL_COLORS["ae"]))
    ax.set_yticks(range(1, len(fams) + 1), fams)
    ax.axvline(np.log10(thr), color="#0b0b0b", lw=1.2, ls="--")
    ax.text(np.log10(thr), len(fams) + 0.55, " threshold", fontsize=8.5, color="#0b0b0b", va="bottom")
    ax.set_xlabel("log10 reconstruction error")
    ax.set_title("Autoencoder reconstruction error by family", pad=18)
    ax.grid(axis="y", visible=False)
    fig.savefig(path)
    plt.close(fig)


def plot_history(history: list[dict], path: Path) -> None:
    plt = style()
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h["epoch"], h["train_loss"], color="#2a78d6", lw=2, label="Train (benign)")
    ax.plot(h["epoch"], h["val_loss"], color="#eb6834", lw=2, label="Validation (benign)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error")
    ax.set_title("Autoencoder training")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=DEFAULT_PATHS["results"])
    args = ap.parse_args(argv)
    res = Path(args.results)
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(res / "scores_test.parquet")
    val = pd.read_parquet(res / "scores_val.parquet")
    thr_info = json.loads((res / "thresholds.json").read_text())
    thr = thr_info["thresholds"]
    models = [m for m in ("ae", "if", "rf") if f"score_{m}" in scores.columns]
    y = scores["is_attack"].to_numpy()

    metrics = {m: binary_metrics(y, scores[f"score_{m}"].to_numpy(), thr[m]) for m in models}
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fam = detection_table(scores, models, thr, "family")
    lab = detection_table(scores, models, thr, "label")
    fam.to_csv(res / "per_family.csv", index=False)
    # Thresholds are set on Monday's benign traffic; does the false-alarm rate hold on later days?
    ben = scores[scores["is_attack"] == 0]
    drift = pd.DataFrame([{"day": d, "n": len(g), **{m: float((g[f"score_{m}"] > thr[m]).mean()) for m in models}}
                          for d, g in ben.groupby("day", observed=True)])
    order = {d: i for i, d in enumerate(["monday", "tuesday", "wednesday", "thursday", "friday"])}
    drift = drift.sort_values("day", key=lambda col: col.map(order)).reset_index(drop=True)
    drift.to_csv(res / "benign_fpr_by_day.csv", index=False)
    lab.to_csv(res / "per_label.csv", index=False)

    sweep = []
    for m in [m for m in models if m != "rf"]:
        for p in SWEEP:
            t = float(np.percentile(val[f"score_{m}"], p))
            r = binary_metrics(y, scores[f"score_{m}"].to_numpy(), t)
            sweep.append({"model": m, "val_percentile": p, **{k: r[k] for k in
                          ("threshold", "precision", "recall", "f1", "fpr")}})
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(res / "threshold_sweep.csv", index=False)

    plot_family(fam, models, figs / "per_family_detection.png")
    plot_roc(scores, models, metrics, figs / "roc.png")
    if "ae" in models:
        plot_ae_errors(scores, thr["ae"], figs / "ae_error_by_family.png")
        hist = res / "ae_history.json"
        if hist.exists():
            plot_history(json.loads(hist.read_text()), figs / "ae_training.png")

    overall = pd.DataFrame([{"model": MODEL_NAMES[m], **{k: metrics[m][k] for k in
                             ("precision", "recall", "f1", "fpr")},
                             "roc_auc": f"{metrics[m]['roc_auc']:.4f}", "pr_auc": f"{metrics[m]['pr_auc']:.4f}"}
                            for m in models])
    info_path = res / "train_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"models": {}}
    timing = pd.DataFrame([{"model": MODEL_NAMES[m], "fit_seconds": info["models"].get(m, {}).get("fit_seconds", ""),
                            "score_seconds": info["models"].get(m, {}).get("score_seconds", "")} for m in models])

    md = [
        "# Results summary",
        "",
        f"Test set: {len(scores):,} flows, {int(y.sum()):,} attacks ({y.mean():.1%}). "
        f"Unsupervised thresholds = {thr_info['percentile']}th percentile of benign validation scores; "
        "Random Forest threshold = 0.5.",
        "",
        "## Overall",
        "",
        md_table(overall, ["precision", "recall", "f1", "fpr"]),
        "",
        "## Detection rate by attack family",
        "",
        "The Benign row is the false-positive rate.",
        "",
        md_table(fam, models),
        "",
        "## Detection rate by original label",
        "",
        md_table(lab, models),
        "",
        "## False-positive rate on benign test flows, by day",
        "",
        f"Unsupervised thresholds were set so about {100 - thr_info['percentile']:g}% of Monday's benign "
        "validation flows are flagged. Higher rates on later days indicate drift in normal traffic.",
        "",
        md_table(drift, models),
        "",
        "## Threshold sweep (unsupervised models)",
        "",
        md_table(sweep.assign(model=sweep["model"].map(MODEL_NAMES), val_percentile=sweep["val_percentile"].astype(str),
                              threshold=sweep["threshold"].map(lambda v: f"{v:.4g}")),
                 ["precision", "recall", "f1", "fpr"]),
        "",
        "## Training and scoring time",
        "",
        md_table(timing, []),
        "",
    ]
    top = res / "ae_top_features.csv"
    if top.exists():
        t = pd.read_csv(top)
        t = t[t["rank"] <= 3].groupby("family")["feature"].apply(", ".join).reset_index()
        md += ["## Features the autoencoder reconstructs worst, per family",
               "", "(relative to its error on benign validation flows)", "",
               md_table(t.rename(columns={"feature": "top 3 features"}), []), ""]
    (res / "summary.md").write_text("\n".join(md))

    print(md_table(overall, ["precision", "recall", "f1", "fpr"]))
    print()
    print(md_table(fam, models))
    print()
    print(md_table(drift, models))
    print(f"\nwrote {res}/summary.md and {figs}/")


if __name__ == "__main__":
    main()
