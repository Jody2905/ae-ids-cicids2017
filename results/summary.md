# Results summary

Test set: 878,342 flows, 168,239 attacks (19.2%). Unsupervised thresholds = 99.0th percentile of benign validation scores; Random Forest threshold = 0.5.

## Overall

| model | precision | recall | f1 | fpr | roc_auc | pr_auc |
|---|---:|---:|---:|---:|---|---|
| Autoencoder | 81.3% | 77.6% | 79.4% | 4.2% | 0.9531 | 0.8368 |
| Isolation Forest | 14.2% | 1.7% | 3.1% | 2.5% | 0.9109 | 0.5629 |
| Random Forest | 99.7% | 99.5% | 99.6% | 0.1% | 0.9999 | 0.9999 |

## Detection rate by attack family

The Benign row is the false-positive rate.

| family | n | Autoencoder | Isolation Forest | Random Forest |
|---|---:|---:|---:|---:|
| Heartbleed | 6 | 100.0% | 100.0% | 83.3% |
| PortScan | 978 | 92.9% | 13.5% | 97.3% |
| Infiltration | 18 | 88.9% | 83.3% | 27.8% |
| DDoS | 64,007 | 83.5% | 0.2% | 99.9% |
| DoS | 96,865 | 78.5% | 2.7% | 99.6% |
| Benign | 710,103 | 4.2% | 2.5% | 0.1% |
| WebAttack | 1,071 | 1.3% | 0.0% | 92.9% |
| BruteForce | 4,576 | 0.8% | 0.0% | 98.5% |
| Bot | 718 | 0.4% | 0.0% | 65.7% |

## Detection rate by original label

| label | n | Autoencoder | Isolation Forest | Random Forest |
|---|---:|---:|---:|---:|
| Heartbleed | 6 | 100.0% | 100.0% | 83.3% |
| DoS slowloris | 2,687 | 96.8% | 21.3% | 99.5% |
| PortScan | 978 | 92.9% | 13.5% | 97.3% |
| Infiltration | 18 | 88.9% | 83.3% | 27.8% |
| DDoS | 64,007 | 83.5% | 0.2% | 99.9% |
| DoS Hulk | 86,423 | 82.3% | 0.5% | 99.7% |
| DoS Slowhttptest | 2,614 | 38.9% | 60.6% | 99.2% |
| DoS GoldenEye | 5,141 | 26.6% | 0.7% | 99.5% |
| BENIGN | 710,103 | 4.2% | 2.5% | 0.1% |
| Web Attack - Brute Force | 735 | 1.5% | 0.0% | 93.7% |
| Web Attack - XSS | 326 | 0.9% | 0.0% | 92.9% |
| SSH-Patator | 1,610 | 0.8% | 0.0% | 96.1% |
| FTP-Patator | 2,966 | 0.8% | 0.0% | 99.8% |
| Bot | 718 | 0.4% | 0.0% | 65.7% |
| Web Attack - Sql Injection | 10 | 0.0% | 0.0% | 30.0% |

## False-positive rate on benign test flows, by day

Unsupervised thresholds were set so about 1% of Monday's benign validation flows are flagged. Higher rates on later days indicate drift in normal traffic.

| day | n | Autoencoder | Isolation Forest | Random Forest |
|---|---:|---:|---:|---:|
| tuesday | 179,952 | 2.8% | 0.9% | 0.0% |
| wednesday | 177,671 | 2.3% | 1.2% | 0.0% |
| thursday | 164,124 | 2.9% | 1.0% | 0.1% |
| friday | 188,356 | 8.5% | 6.4% | 0.1% |

## Threshold sweep (unsupervised models)

| model | val_percentile | threshold | precision | recall | f1 | fpr |
|---|---|---|---:|---:|---:|---:|
| Autoencoder | 90.0 | 0.01965 | 55.9% | 92.4% | 69.7% | 17.2% |
| Autoencoder | 95.0 | 0.03131 | 64.8% | 85.0% | 73.5% | 11.0% |
| Autoencoder | 97.5 | 0.0465 | 72.5% | 81.6% | 76.8% | 7.3% |
| Autoencoder | 99.0 | 0.08855 | 81.3% | 77.6% | 79.4% | 4.2% |
| Autoencoder | 99.5 | 0.1243 | 83.7% | 73.4% | 78.2% | 3.4% |
| Autoencoder | 99.9 | 0.2691 | 85.1% | 50.0% | 62.9% | 2.1% |
| Isolation Forest | 90.0 | 0.5467 | 59.7% | 79.8% | 68.3% | 12.8% |
| Isolation Forest | 95.0 | 0.5698 | 67.1% | 61.4% | 64.1% | 7.1% |
| Isolation Forest | 97.5 | 0.5909 | 54.4% | 20.6% | 29.9% | 4.1% |
| Isolation Forest | 99.0 | 0.612 | 14.2% | 1.7% | 3.1% | 2.5% |
| Isolation Forest | 99.5 | 0.6276 | 23.0% | 1.4% | 2.6% | 1.1% |
| Isolation Forest | 99.9 | 0.6578 | 61.7% | 0.8% | 1.7% | 0.1% |

## Training and scoring time

| model | fit_seconds | score_seconds |
|---|---|---|
| Autoencoder | 194.3 | 0.65 |
| Isolation Forest | 1.7 | 12.2 |
| Random Forest | 105.7 | 2.29 |

## Features the autoencoder reconstructs worst, per family

(relative to its error on benign validation flows)

| family | top 3 features |
|---|---|
| Benign | urg_flag_count, bwd_packet_length_mean, avg_bwd_segment_size |
| Bot | total_length_of_bwd_packets, subflow_bwd_bytes, psh_flag_count |
| BruteForce | bwd_packets_s, bwd_packet_length_mean, avg_bwd_segment_size |
| DDoS | packet_length_mean, average_packet_size, fwd_packet_length_mean |
| DoS | fin_flag_count, idle_mean, idle_max |
| Heartbleed | psh_flag_count, ack_flag_count, bwd_iat_min |
| Infiltration | bwd_packet_length_mean, avg_bwd_segment_size, bwd_packet_length_max |
| PortScan | psh_flag_count, ack_flag_count, fwd_packets_s |
| WebAttack | fwd_iat_min, urg_flag_count, flow_bytes_s |
