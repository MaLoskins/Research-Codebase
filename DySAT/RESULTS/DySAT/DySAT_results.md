# DySAT - Aggregated Performance Report

This report shows the performance of the model on various event streams when using a single set of 'overall best' hyperparameters.

## Overall Best Hyperparameters Used:
- hidden_dim: 128
- lr: 0.0015361883741367818
- dropout: 0.1
- weight_decay: 2.208070847805232e-06
- scheduler_patience: 18
- early_stop_patience: 20
- num_struct_heads: 2
- num_temporal_heads: 2
- use_temporal_attn: True

## Per-Event Performance (with Overall Best HPs):
| Event Stream         | Test Accuracy | Test F1-score (macro) |
|----------------------|---------------|-----------------------|
| charliehebdo         |        0.9110 |                0.8554 |
| ferguson             |        0.8901 |                0.8615 |
| germanwings-crash    |        0.9021 |                0.9011 |
| ottawashooting       |        0.9110 |                0.9107 |
| sydneysiege          |        0.8417 |                0.8259 |
|----------------------|---------------|-----------------------|
| **Average**          | **    0.8912** | **            0.8709** |
