# Aggregated Results for GCN

## Best Overall Hyperparameters (selected by average validation accuracy across events):
```json
{
  "dropout": 0.0,
  "hidden_dim": 128,
  "lr": 0.005,
  "num_layers": 3,
  "weight_decay": 0
}
```

## Performance on Each Event (using best overall hyperparameters):
| Event             |   Test Accuracy |   F1 Score |
|:------------------|----------------:|-----------:|
| all               |          0.7613 |     0.6918 |
| charliehebdo      |          0.8206 |     0.619  |
| ferguson          |          0.9196 |     0.8895 |
| germanwings-crash |          0.7649 |     0.7574 |
| ottawashooting    |          0.7795 |     0.7775 |
| sydneysiege       |          0.7562 |     0.7306 |

## Average Performance (using best overall hyperparameters):
- Average Test Accuracy: 0.8004
- Average F1 Score (Macro): 0.7443
