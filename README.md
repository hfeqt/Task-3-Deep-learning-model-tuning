# Task-3: Deep Learning Model Tuning (MNIST)

## Objective
Experiment with training parameters like number of neurons, batch size, and epochs on an MNIST deep learning model.

## Configurations Tried

| Neurons | Batch Size | Epochs |
|---------|------------|--------|
| 64      | 32         | 50     |
| 256     | 64         | 50     |
| 512     | 128        | 100    |

## Observations

- **Accuracy:** Increasing neurons and epochs improved both training and validation accuracy.
- **Batch Size:** Larger batch sizes sped up training but slightly reduced validation accuracy.
- **Best Performance:** 512 neurons, 64 batch size, and 100 epochs gave the best results.

## Loss Graphs
See plots in `/plots/` folder — they show how loss decreases over epochs for each configuration.

## Notes
- Training loss decreases with more epochs.
- Validation loss decreases more consistently with higher neurons and balanced batch size.
