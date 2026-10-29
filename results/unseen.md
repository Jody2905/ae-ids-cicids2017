# Detection of attack families the Random Forest never saw

rf = trained with the family; rf_unseen = retrained without it; rf_unseen_fpr = that retrained model's false-positive rate on benign test flows. The autoencoder and Isolation Forest never see any attacks.

| family | n | Random Forest | Random Forest (family unseen) | Autoencoder | Isolation Forest | RF unseen FPR |
|---|---:|---:|---:|---:|---:|---:|
| DoS | 96,865 | 99.6% | 0.2% | 78.5% | 2.7% | 0.0% |
| DDoS | 64,007 | 99.9% | 63.6% | 83.5% | 0.2% | 0.1% |
| BruteForce | 4,576 | 98.5% | 0.0% | 0.8% | 0.0% | 0.1% |
| WebAttack | 1,071 | 92.9% | 0.6% | 1.3% | 0.0% | 0.1% |
| PortScan | 978 | 97.3% | 7.3% | 92.9% | 13.5% | 0.1% |
| Bot | 718 | 65.7% | 0.0% | 0.4% | 0.0% | 0.1% |
