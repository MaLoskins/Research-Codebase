# Aggregated Results for Architecture: GCN

## Best Overall Hyperparameters (selected based on highest validation accuracy from one event's Optuna study):
- dropout_feat: 0.2
- hidden_dim: 256
- lr: 0.0009660598518748213
- num_layers: 3
- weight_decay: 3.913649148498495e-06

## Performance with Best Overall Hyperparameters:

| Event             | Test Accuracy | F1 Score (Macro) |
|-------------------|---------------|------------------|
| all               | 0.7983          | 0.6340            |
| charliehebdo      | 0.8751          | 0.6030            |
| ferguson          | 0.9549          | 0.9154            |
| germanwings-crash | 0.7864          | 0.8106            |
| ottawashooting    | 0.8281          | 0.8392            |
| sydneysiege       | 0.8153          | 0.7359            |
| **Average**       | **0.8430**      | **0.7563**         |
