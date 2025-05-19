# TGN Aggregated Performance Report

## Hyperparameters Used for this Report (from 'all' dataset Optuna or fixed):
- lr: 0.001
- memory_dim: 256
- time_dim: 128
- embedding_dim: 256
- dropout_rate: 0.15
- projector_dropout_rate: 0.2
- grad_clip_norm: 2.0
- batch_size: 256
- project_features: True
- use_layernorm: True
- leaky_relu_slope: 0.1

## Performance on Individual Event Streams (Test Metrics from Best Validation Epoch):

| Event Stream     | Test Accuracy | Test F1-score | Test Precision | Test Recall | Best Val Epoch |
|------------------|---------------|---------------|----------------|-------------|----------------|
| charliehebdo      | 0.7155         | 0.5904         | 0.4261           | 0.9608       | 7              |
| ferguson          | 0.8736         | 0.8317         | 0.8077           | 0.8571       | 24             |
| germanwings-crash | 0.7460         | 0.8049         | 0.8049           | 0.8049       | 3              |
| ottawashooting    | 0.9320         | 0.9247         | 0.9556           | 0.8958       | 21             |
| sydneysiege       | 0.6808         | 0.5699         | 0.5392           | 0.6044       | 4              |

## Average Performance (over 5 successfully processed events with Test Metrics)
- Average Test Accuracy: 0.7896
- Average Test F1: 0.7443
- Average Test Precision: 0.7067
- Average Test Recall: 0.8246
