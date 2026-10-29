# Autoencoder Intrusion Detection on CIC-IDS2017

Can a model that has **only ever seen normal traffic** catch network attacks, and
how does it compare with a supervised model that was trained on labelled attacks?

This repo trains a dense autoencoder on benign network flows from CIC-IDS2017,
flags flows it cannot reconstruct well, and compares it with:

| Model | Type | Sees attacks during training? |
|---|---|---|
| Autoencoder (PyTorch) | unsupervised anomaly detection | no |
| Isolation Forest | unsupervised anomaly detection | no |
| Random Forest | supervised classifier (upper-bound baseline) | yes |

## Research questions

1. **Overall:** at a fixed false-positive budget (about 1% of benign flows), how many attacks does each model catch?
2. **Per attack family:** which attacks are easy for an anomaly detector (DoS, PortScan) and which look like normal traffic (Bot, Infiltration, web attacks)?
3. **Unseen attacks:** if the Random Forest never saw a family during training, does it still catch it? The autoencoder never sees *any* attacks, so this is a fair test of the main argument for anomaly detection.
4. **Why it fails:** which features the autoencoder reconstructs worst for each family, as a starting point for failure analysis.

## Setup (Windows PowerShell)

```powershell
cd C:\Projects
# unzip ae-ids-cicids2017.zip here, then:
cd ae-ids-cicids2017
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

`pip install -e .` installs a CPU build of PyTorch on Windows, which is all this project needs.

## Get the data

1. Go to the [CIC-IDS2017 page](https://www.unb.ca/cic/datasets/ids-2017.html) and download `MachineLearningCSV.zip` (CIC asks for a short registration form first).
2. Unzip it and put the eight `*.pcap_ISCX.csv` files directly in `data\raw\`.

The day of the week is read from each file name; Monday (all benign) is used for training.

## Run it

**1. Smoke test with fake data (about 1 minute).** This checks your setup before you touch the real dataset:

```powershell
python tools\make_fake_data.py --out data\fake_raw
ids-prepare --raw data\fake_raw --out data\fake\flows.parquet
ids-train --data data\fake\flows.parquet --model-dir data\fake\models --out data\fake\results
ids-evaluate --results data\fake\results
```

**2. The real experiment:**

```powershell
ids-prepare                 # clean + split -> data\processed\flows.parquet (a few minutes)
ids-train --threads 2       # AE + Isolation Forest + Random Forest -> results\
ids-evaluate                # metrics, tables, figures -> results\summary.md, results\figures\
ids-unseen --threads 2      # leave-one-family-out Random Forest -> results\unseen.md
```

**Running alongside another job** (e.g. an Ollama experiment): keep `--threads 2`
(the default). Training will be slower but will not starve the other process.

**Low on RAM (8 GB)?** Downsample the Tuesday–Friday benign flows:

```powershell
ids-prepare --benign-frac 0.3
```

Every command has `--help`.

## What the pipeline does

**Cleaning (`ids-prepare`)**
- Normalises column names (the raw headers have stray leading spaces) and drops the duplicated `Fwd Header Length` column.
- Fixes the broken dash in the web-attack labels (`Web Attack � Brute Force` → `Web Attack - Brute Force`).
- Drops rows containing `NaN` or `Infinity` (mostly in `Flow Bytes/s` and `Flow Packets/s`).
- Drops exact duplicate flows (`--no-dedupe` to keep them) and constant columns.
- Drops `Destination Port` by default, so models learn traffic behaviour rather than memorising which ports were attacked in this testbed (`--keep-port` to keep it).
- Groups the 15 labels into 8 families: Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration (plus Heartbleed, which has too few flows to evaluate).

**Splits**

| Split | Contents | Used by |
|---|---|---|
| `train_benign` | 80% of Monday's benign flows | AE, Isolation Forest, Random Forest, scaler |
| `val_benign` | 20% of Monday's benign flows | AE early stopping, anomaly thresholds |
| `sup_train` | 50% of Tuesday–Friday, stratified by label | Random Forest only |
| `test` | the other 50% of Tuesday–Friday | every model |

Training the unsupervised models on Monday and testing on later days also means
they have to cope with normal day-to-day drift in benign traffic, as a real
deployment would.

**Features:** signed log transform, `sign(x)·log(1+|x|)`, then standardisation
fitted on benign training flows only, clipped at ±10 standard deviations.

**Autoencoder:** dense network `d → 64 → 32 → 8 → 32 → 64 → d`, MSE loss, Adam,
early stopping on benign validation loss. Anomaly score = mean squared
reconstruction error.

**Thresholds:** the unsupervised models flag a flow when its score exceeds the
99th percentile of benign validation scores (`--percentile`), which targets about
a 1% false-positive rate before seeing a single attack. `ids-evaluate` also sweeps
90–99.9 to show the detection/false-alarm trade-off. The Random Forest uses 0.5.

## Outputs

| File | What it holds |
|---|---|
| `results/summary.md` | every table below, ready to paste into the report |
| `results/metrics.json` | precision, recall, F1, FPR, ROC-AUC, PR-AUC per model |
| `results/per_family.csv`, `per_label.csv` | detection rate per attack family and per original label |
| `results/threshold_sweep.csv` | unsupervised models at several thresholds |
| `results/unseen.md` | Random Forest with vs. without each family, next to AE and Isolation Forest |
| `results/ae_top_features.csv` | features with the worst reconstruction per family |
| `results/train_info.json` | training and scoring time, model sizes, Random Forest feature importances |
| `results/figures/*.png` | per-family detection, ROC (log FPR axis), AE error by family, AE training curve, unseen-family chart |
| `data/processed/prep_report.json` | rows dropped at each cleaning step, labels per split |

## Known dataset issues worth discussing in the report

CIC-IDS2017 has documented problems: mislabelled flows, attack flows that never
actually reached their target, and flow-extraction artefacts. Engelen, Rimmer and
Joosen (2021) analysed these and released a corrected version; Lanvin et al.
(2022) showed the errors noticeably change reported detection performance. The
loader normalises column names and handles the corrected version's "Attempted"
labels (`--attempted drop|benign|attack`), so re-running on the corrected data
is a natural extension. Check the column names first, since that release may
differ from the original.

## Ideas for extending it

- Re-run on the corrected CIC-IDS2017 and compare per-family results.
- Variational or denoising autoencoder; a deeper or narrower bottleneck.
- Per-flow explanations: which features pushed one specific flow over the threshold.
- Test on a second dataset (e.g. CSE-CIC-IDS2018 or UNSW-NB15) to check generalisation.

## References

- Sharafaldin, Lashkari, Ghorbani. *Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization.* ICISSP 2018.
- Engelen, Rimmer, Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE Security and Privacy Workshops (WTMC), 2021.
- Lanvin et al. *Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes.* CRiSIS 2022.
- Liu, Ting, Zhou. *Isolation Forest.* ICDM 2008.
