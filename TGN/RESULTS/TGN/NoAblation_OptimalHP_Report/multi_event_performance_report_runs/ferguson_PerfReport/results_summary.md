# TGN Results for Event: ferguson_PerfReport

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
- Best Epoch: 24
- Validation Accuracy: 0.8876
- Validation F1-Score: 0.8905
- Validation Precision: 0.9037
- Validation Recall: 0.8777
- Validation Confusion Matrix:
```
[[115  13]
 [ 17 122]]
```

### Corresponding Test Metrics (for best validation model state):
- Test Accuracy: 0.8736
- Test F1-Score: 0.8317
- Test Precision: 0.8077
- Test Recall: 0.8571
- Test Confusion Matrix:
```
[[151  20]
 [ 14  84]]
```
