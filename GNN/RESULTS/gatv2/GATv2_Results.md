# Aggregated Results for GATV2

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
| all               |          0.789  |     0.7367 |
| charliehebdo      |          0.8532 |     0.7179 |
| ferguson          |          0.9366 |     0.9166 |
| germanwings-crash |          0.7401 |     0.7389 |
| ottawashooting    |          0.7743 |     0.7718 |
| sydneysiege       |          0.7834 |     0.756  |

## Average Performance (using best overall hyperparameters):
- Average Test Accuracy: 0.8128
- Average F1 Score (Macro): 0.7730
