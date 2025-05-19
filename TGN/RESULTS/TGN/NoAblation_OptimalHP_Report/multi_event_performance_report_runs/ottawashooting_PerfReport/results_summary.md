# TGN Results for Event: ottawashooting_PerfReport

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
- Best Epoch: 21
- Validation Accuracy: 0.9474
- Validation F1-Score: 0.9474
- Validation Precision: 0.9429
- Validation Recall: 0.9519
- Validation Confusion Matrix:
```
[[99  6]
 [ 5 99]]
```

### Corresponding Test Metrics (for best validation model state):
- Test Accuracy: 0.9320
- Test F1-Score: 0.9247
- Test Precision: 0.9556
- Test Recall: 0.8958
- Test Confusion Matrix:
```
[[106   4]
 [ 10  86]]
```
