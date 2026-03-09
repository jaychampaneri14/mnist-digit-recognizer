# MNIST Digit Recognizer

CNN trained on the MNIST handwritten digit dataset using PyTorch, achieving ~99% test accuracy.

## Architecture
- Two conv blocks (32 and 64 filters) with BatchNorm, MaxPool, Dropout
- Dense classifier: 64×7×7 → 512 → 10
- OneCycleLR scheduler + Adam optimizer

## Setup

```bash
pip install -r requirements.txt
python main.py
```

MNIST data downloads automatically (~11 MB). Falls back to synthetic data if unavailable.

## Output
- `training_history.png` — loss and accuracy curves
- `confusion_matrix.png` — per-digit confusion matrix
- `sample_predictions.png` — 16 sample predictions
- `best_mnist_model.pth` — saved model weights

## Expected Performance
~99% accuracy on MNIST test set with 10 epochs of training.
