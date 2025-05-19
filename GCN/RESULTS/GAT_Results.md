# Aggregated Results for GAT

## Best Overall Hyperparameters (selected by average validation accuracy across events):
```json
{
  "dropout": 0.0,
  "hidden_dim": 64,
  "lr": 0.005,
  "num_layers": 3,
  "weight_decay": 0
}
```

## Performance on Each Event (using best overall hyperparameters):
| Event             |   Test Accuracy |   F1 Score |
|:------------------|----------------:|-----------:|
| all               |          0.7829 |     0.7277 |
| charliehebdo      |          0.8541 |     0.723  |
| ferguson          |          0.9311 |     0.9088 |
| germanwings-crash |          0.793  |     0.7889 |
| ottawashooting    |          0.7637 |     0.7612 |
| sydneysiege       |          0.7706 |     0.7421 |

## Average Performance (using best overall hyperparameters):
- Average Test Accuracy: 0.8159
- Average F1 Score (Macro): 0.7753
