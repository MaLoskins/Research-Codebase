# TGN Results for Event: sydneysiege_PerfReport

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
- Best Epoch: 4
- Validation Accuracy: 0.7781
- Validation F1-Score: 0.6635
- Validation Precision: 0.6364
- Validation Recall: 0.6931
- Validation Confusion Matrix:
```
[[179  40]
 [ 31  70]]
```

### Corresponding Test Metrics (for best validation model state):
- Test Accuracy: 0.6808
- Test F1-Score: 0.5699
- Test Precision: 0.5392
- Test Recall: 0.6044
- Test Confusion Matrix:
```
[[122  47]
 [ 36  55]]
```
