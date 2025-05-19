# TGN Results for Event: germanwings-crash_PerfReport

## Hyperparameters Used (for this successful run):
- lr: 0.001
- batch_size: 256
- memory_dim: 256
- time_dim: 128
- embedding_dim: 256
- dropout_rate: 0.15
- projector_dropout_rate: 0.2
- grad_clip_norm: 2.0
- project_features: True
- use_layernorm: True
- leaky_relu_slope: 0.1

## Performance Metrics (at best validation epoch):
- Best Epoch: 3
- Validation Accuracy: 0.7500
- Validation F1-Score: 0.7945
- Validation Precision: 0.7250
- Validation Recall: 0.8788
- Validation Confusion Matrix:
```
[[16 11]
 [ 4 29]]
```

### Corresponding Test Metrics (for best validation model state):
- Test Accuracy: 0.7460
- Test F1-Score: 0.8049
- Test Precision: 0.8049
- Test Recall: 0.8049
- Test Confusion Matrix:
```
[[14  8]
 [ 8 33]]
```
