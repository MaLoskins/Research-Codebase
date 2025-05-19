# Results for sydneysiege

## Best Hyperparameters (from Optuna study on this event):
- lr: 0.006884502001297509
- memory_dim: 256
- time_dim: 64
- embedding_dim: 256
- dropout_rate: 0.20021118382588396
- projector_dropout_rate: 0.21446093073625694
- grad_clip_norm: 4.971233219922118
- batch_size: 128

## Performance with these Hyperparameters:
- Best Validation F1 (during Optuna): 0.7751
- Test Accuracy: 0.7269
- Test F1-score: 0.6667
- Test Precision: 0.5820
- Test Recall: 0.7802
- Test Confusion Matrix:
```
[
  [
    118,
    51
  ],
  [
    20,
    71
  ]
]
```
