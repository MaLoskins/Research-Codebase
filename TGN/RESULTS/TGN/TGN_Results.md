# Aggregated TGN Results

Applied Best Overall Hyperparameters (from 'sydneysiege' dataset Optuna study (highest val F1)):
- lr: 0.006884502001297509
- memory_dim: 256
- time_dim: 64
- embedding_dim: 256
- dropout_rate: 0.20021118382588396
- projector_dropout_rate: 0.21446093073625694
- grad_clip_norm: 4.971233219922118
- batch_size: 128

## Performance on Individual Event Streams (using Best Overall HPs):

| Event Stream     | Test Accuracy | Test F1-score | Test Precision | Test Recall |
|------------------|---------------|---------------|----------------|-------------|
| charliehebdo     | 0.7113         | 0.5868         | 0.4224           | 0.9608       |
| ferguson         | 0.8550         | 0.8079         | 0.7810           | 0.8367       |
| germanwings-crash | 0.8254         | 0.8608         | 0.8947           | 0.8293       |
| ottawashooting   | 0.9417         | 0.9388         | 0.9200           | 0.9583       |
| sydneysiege      | 0.7231         | 0.6364         | 0.5888           | 0.6923       |
| **Average**      | **0.8113**     | **0.7661**     | **0.7214**      | **0.8555**   |
