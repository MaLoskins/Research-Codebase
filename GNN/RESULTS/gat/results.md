# Aggregated Results for Architecture: GAT

## Best Overall Hyperparameters (selected based on highest validation accuracy from one event's Optuna study):
- dropout_feat: 0.1
- heads: 2
- hidden_dim: 256
- lr: 0.0031670559815994287
- num_layers: 3
- weight_decay: 5.996033105943871e-06

## Performance with Best Overall Hyperparameters:

| Event             | Test Accuracy | F1 Score (Macro) |
|-------------------|---------------|------------------|
| all               | 0.6976          | 0.0000            |
| charliehebdo      | 0.8329          | 0.3892            |
| ferguson          | 0.9167          | 0.8362            |
| germanwings-crash | 0.7235          | 0.7474            |
| ottawashooting    | 0.7199          | 0.7367            |
| sydneysiege       | 0.7305          | 0.5670            |
| **Average**       | **0.7702**      | **0.5461**         |
