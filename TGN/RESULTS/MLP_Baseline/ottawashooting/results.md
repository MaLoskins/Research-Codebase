# MLP Baseline Results for Event: ottawashooting

## Hyperparameters Used:
- results_base_dir: RESULTS/MLP_Baseline
- epochs: 100
- lr: 0.001
- weight_decay: 1e-05
- batch_size: 64
- hidden_dim1: 256
- hidden_dim2: 128
- dropout_rate: 0.5
- test_size: 0.2
- val_size: 0.15
- seed: 42

## Performance Metrics:
- Best Validation F1-Score: 0.8645

### Final Test Metrics (at best validation or last epoch):
- Test Accuracy: 0.8626
- Test F1-Score: 0.8804
- Test Precision: 0.8214
- Test Recall: 0.9485
- Test Confusion Matrix:
```
[[65 20]
 [ 5 92]]
```
