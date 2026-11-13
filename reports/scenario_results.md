# Sentinel ML - test scenario results

Generated: 2026-11-13T03:36:52.595281+00:00

Points per dataset: 1200; LSTM epochs: 100; hidden: 32; sequence: 30; tolerance: +/-2 points.

Protocol: one **clean** seasonal baseline; every detector is fitted on its first 70%. Scenario A is the clean held-out 30%; B-H are separate unseen datasets. Training contains no labelled anomaly, and the LSTM calibration holdout is inside the clean training half.

One-to-one matching: each predicted point can match at most one ground-truth point, and vice versa. Precision, recall and F1 are event-level, not raw point overlap.

Scenario A has no anomalies, so its precision/recall/F1 are 0 by construction; the meaningful number there is the false-positive rate.

## A_clean - Clean seasonal holdout, no anomalies

Evaluation data: held-out 30% of the clean baseline. Ground-truth anomaly points: 0.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.04 | 0.03 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.05 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.00 | 0.44 |
| Isolation Forest | 0.000 | 0.000 | 0.000 | 0.042 | 15 | 0 | 15 | 45.48 | 16.06 |
| Spectral Residual | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 0.00 | 0.49 |
| Ensemble | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | 40.72 | 15.82 |
| LSTM Autoencoder | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 0 | - | 41.82 |

## B_isolated_spikes - Isolated spikes

Evaluation data: separate unseen synthetic dataset (seed=143). Ground-truth anomaly points: 36.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.972 | 0.986 | 0.000 | 0 | 1 | 35 | 0.02 | 0.02 |
| MAD | 1.000 | 0.278 | 0.435 | 0.000 | 0 | 26 | 10 | 0.04 | 0.02 |
| Grubbs | 1.000 | 0.417 | 0.588 | 0.000 | 0 | 21 | 15 | 0.00 | 1.16 |
| Isolation Forest | 0.172 | 1.000 | 0.294 | 0.149 | 173 | 0 | 209 | 40.40 | 20.28 |
| Spectral Residual | 1.000 | 0.972 | 0.986 | 0.000 | 0 | 1 | 35 | 0.00 | 0.31 |
| Ensemble | 1.000 | 0.972 | 0.986 | 0.000 | 0 | 1 | 35 | 40.07 | 20.60 |
| LSTM Autoencoder | 0.105 | 1.000 | 0.190 | 0.263 | 306 | 0 | 342 | - | 71.63 |

Recall by anomaly type (one-to-one matching):

| Detector | spike |
|---|---|
| Z-Score | 0.972 |
| MAD | 0.278 |
| Grubbs | 0.417 |
| Isolation Forest | 1.000 |
| Spectral Residual | 0.972 |
| Ensemble | 0.972 |
| LSTM Autoencoder | 1.000 |

## C_dense_spikes - Densely packed spikes and drops

Evaluation data: separate unseen synthetic dataset (seed=144). Ground-truth anomaly points: 96.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.927 | 0.962 | 0.000 | 0 | 7 | 89 | 0.02 | 0.02 |
| MAD | 1.000 | 0.438 | 0.609 | 0.000 | 0 | 54 | 42 | 0.04 | 0.02 |
| Grubbs | 1.000 | 0.083 | 0.154 | 0.000 | 0 | 88 | 8 | 0.00 | 0.90 |
| Isolation Forest | 0.224 | 0.990 | 0.365 | 0.298 | 329 | 1 | 424 | 40.21 | 20.76 |
| Spectral Residual | 1.000 | 0.531 | 0.694 | 0.000 | 0 | 45 | 51 | 0.00 | 0.30 |
| Ensemble | 1.000 | 0.927 | 0.962 | 0.000 | 0 | 7 | 89 | 39.90 | 21.00 |
| LSTM Autoencoder | 0.092 | 1.000 | 0.169 | 0.857 | 946 | 0 | 1042 | - | 71.43 |

Recall by anomaly type (one-to-one matching):

| Detector | drop | spike |
|---|---|---|
| Z-Score | 0.979 | 0.898 |
| MAD | 0.489 | 0.490 |
| Grubbs | 0.128 | 0.061 |
| Isolation Forest | 0.979 | 1.000 |
| Spectral Residual | 0.553 | 0.612 |
| Ensemble | 0.979 | 0.898 |
| LSTM Autoencoder | 1.000 | 1.000 |

## D_level_shift - Sudden level shifts

Evaluation data: separate unseen synthetic dataset (seed=145). Ground-truth anomaly points: 103.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.835 | 0.910 | 0.000 | 0 | 17 | 86 | 0.02 | 0.03 |
| MAD | 1.000 | 0.243 | 0.391 | 0.000 | 0 | 78 | 25 | 0.04 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 103 | 0 | 0.00 | 0.58 |
| Isolation Forest | 0.644 | 1.000 | 0.783 | 0.052 | 57 | 0 | 160 | 40.48 | 19.55 |
| Spectral Residual | 0.909 | 0.097 | 0.175 | 0.001 | 1 | 93 | 11 | 0.00 | 0.29 |
| Ensemble | 1.000 | 0.864 | 0.927 | 0.000 | 0 | 14 | 89 | 40.08 | 19.86 |
| LSTM Autoencoder | 0.873 | 1.000 | 0.932 | 0.014 | 15 | 0 | 118 | - | 71.68 |

Recall by anomaly type (one-to-one matching):

| Detector | level_shift |
|---|---|
| Z-Score | 0.835 |
| MAD | 0.243 |
| Grubbs | 0.000 |
| Isolation Forest | 1.000 |
| Spectral Residual | 0.097 |
| Ensemble | 0.864 |
| LSTM Autoencoder | 1.000 |

## E_drift - Gradual drifts

Evaluation data: separate unseen synthetic dataset (seed=146). Ground-truth anomaly points: 118.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.254 | 0.405 | 0.000 | 0 | 88 | 30 | 0.02 | 0.02 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 118 | 0 | 0.04 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 118 | 0 | 0.00 | 0.57 |
| Isolation Forest | 0.582 | 0.720 | 0.644 | 0.056 | 61 | 33 | 146 | 40.69 | 19.63 |
| Spectral Residual | 1.000 | 0.068 | 0.127 | 0.000 | 0 | 110 | 8 | 0.00 | 0.33 |
| Ensemble | 1.000 | 0.280 | 0.437 | 0.000 | 0 | 85 | 33 | 40.94 | 20.40 |
| LSTM Autoencoder | 1.000 | 0.924 | 0.960 | 0.000 | 0 | 9 | 109 | - | 71.92 |

Recall by anomaly type (one-to-one matching):

| Detector | drift |
|---|---|
| Z-Score | 0.254 |
| MAD | 0.000 |
| Grubbs | 0.000 |
| Isolation Forest | 0.720 |
| Spectral Residual | 0.068 |
| Ensemble | 0.280 |
| LSTM Autoencoder | 0.924 |

## F_noise_burst - Localised noise bursts

Evaluation data: separate unseen synthetic dataset (seed=147). Ground-truth anomaly points: 85.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 85 | 0 | 0.04 | 0.03 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 85 | 0 | 0.05 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 85 | 0 | 0.00 | 0.62 |
| Isolation Forest | 0.625 | 1.000 | 0.769 | 0.046 | 51 | 0 | 136 | 40.43 | 20.22 |
| Spectral Residual | 1.000 | 0.094 | 0.172 | 0.000 | 0 | 77 | 8 | 0.00 | 0.30 |
| Ensemble | 1.000 | 0.094 | 0.172 | 0.000 | 0 | 77 | 8 | 40.26 | 20.45 |
| LSTM Autoencoder | 1.000 | 0.941 | 0.970 | 0.000 | 0 | 5 | 80 | - | 71.24 |

Recall by anomaly type (one-to-one matching):

| Detector | noise_burst |
|---|---|
| Z-Score | 0.000 |
| MAD | 0.000 |
| Grubbs | 0.000 |
| Isolation Forest | 1.000 |
| Spectral Residual | 0.094 |
| Ensemble | 0.094 |
| LSTM Autoencoder | 0.941 |

## G_flatline - Stuck sensor / flatline

Evaluation data: separate unseen synthetic dataset (seed=148). Ground-truth anomaly points: 54.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 54 | 0 | 0.02 | 0.02 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 54 | 0 | 0.04 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 54 | 0 | 0.00 | 0.57 |
| Isolation Forest | 0.279 | 0.444 | 0.343 | 0.054 | 62 | 30 | 86 | 40.62 | 19.81 |
| Spectral Residual | 0.875 | 0.130 | 0.226 | 0.001 | 1 | 47 | 8 | 0.00 | 0.31 |
| Ensemble | 0.800 | 0.074 | 0.136 | 0.001 | 1 | 50 | 5 | 40.04 | 20.26 |
| LSTM Autoencoder | 0.860 | 0.796 | 0.827 | 0.006 | 7 | 11 | 50 | - | 70.86 |

Recall by anomaly type (one-to-one matching):

| Detector | flatline |
|---|---|
| Z-Score | 0.000 |
| MAD | 0.000 |
| Grubbs | 0.000 |
| Isolation Forest | 0.444 |
| Spectral Residual | 0.130 |
| Ensemble | 0.074 |
| LSTM Autoencoder | 0.796 |

## H_mixed_unseen - Mixed anomaly types, unseen seed

Evaluation data: separate unseen synthetic dataset (seed=149). Ground-truth anomaly points: 106.

| Detector | Precision | Recall | F1 | FPR | FP | FN | Flagged | Fit (ms) | Inference (ms) |
|---|---|---|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.349 | 0.517 | 0.000 | 0 | 69 | 37 | 0.02 | 0.03 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 106 | 0 | 0.04 | 0.02 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 106 | 0 | 0.00 | 0.56 |
| Isolation Forest | 0.544 | 0.764 | 0.635 | 0.062 | 68 | 25 | 149 | 40.73 | 19.99 |
| Spectral Residual | 0.933 | 0.132 | 0.231 | 0.001 | 1 | 92 | 15 | 0.00 | 0.29 |
| Ensemble | 1.000 | 0.443 | 0.614 | 0.000 | 0 | 59 | 47 | 40.14 | 20.51 |
| LSTM Autoencoder | 0.971 | 0.953 | 0.962 | 0.003 | 3 | 5 | 104 | - | 71.23 |

Recall by anomaly type (one-to-one matching):

| Detector | drop | level_shift | missing | noise_burst | seasonal | spike |
|---|---|---|---|---|---|---|
| Z-Score | 1.000 | 0.581 | 0.000 | 0.000 | 0.000 | 0.000 |
| MAD | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Grubbs | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Isolation Forest | 1.000 | 1.000 | 0.000 | 1.000 | 0.250 | 1.000 |
| Spectral Residual | 1.000 | 0.097 | 0.000 | 0.364 | 0.083 | 1.000 |
| Ensemble | 1.000 | 0.629 | 0.000 | 0.364 | 0.083 | 1.000 |
| LSTM Autoencoder | 1.000 | 1.000 | 0.571 | 0.818 | 1.000 | 1.000 |

## I_unlabelled_csv - unlabelled CSV inference

Input: `examples/example_unlabelled.csv`.

```json
{
  "status": "ok",
  "input": "examples/example_unlabelled.csv",
  "points": 600,
  "has_labels": false,
  "flagged": 47,
  "flagged_ratio": 0.07833333333333334,
  "score_min": 0.0047049261684765165,
  "score_max": 1.0,
  "threshold": 0.02986891692251608,
  "predictions_csv": "reports/unlabelled_predictions.csv",
  "note": "Unlabelled inference: predictions and scores only. Precision, recall and F1 are deliberately omitted."
}
```

No ground truth exists for this file, so no precision, recall or F1 is reported. Predictions and scores are saved to `reports/unlabelled_predictions.csv`.

