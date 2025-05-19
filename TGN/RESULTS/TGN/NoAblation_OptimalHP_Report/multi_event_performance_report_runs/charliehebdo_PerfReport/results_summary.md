# TGN Results for Event: charliehebdo_PerfReport

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
- Best Epoch: 7
- Validation Accuracy: 0.7735
- Validation F1-Score: 0.7579
- Validation Precision: 0.6624
- Validation Recall: 0.8857
- Validation Confusion Matrix:
```
[[183  79]
 [ 20 155]]
```

### Corresponding Test Metrics (for best validation model state):
- Test Accuracy: 0.7155
- Test F1-Score: 0.5904
- Test Precision: 0.4261
- Test Recall: 0.9608
- Test Confusion Matrix:
```
[[244 132]
 [  4  98]]
```
