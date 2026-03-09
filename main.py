"""
MNIST Digit Recognizer
CNN trained on MNIST with PyTorch achieving ~99% accuracy.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import warnings
warnings.filterwarnings('ignore')


class DigitCNN(nn.Module):
    """Convolutional Neural Network for digit recognition."""
    def __init__(self):
        super(DigitCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 14x14
            nn.Dropout2d(0.25),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),          # 7x7
            nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def load_mnist(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    try:
        train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"MNIST: {len(train_ds)} train, {len(test_ds)} test samples")
        return train_loader, test_loader, True
    except Exception as e:
        print(f"Could not download MNIST: {e}")
        print("Generating synthetic data instead...")
        return None, None, False


def generate_synthetic_mnist(n_train=6000, n_test=1000):
    """Generate synthetic 28x28 digit-like data for testing."""
    np.random.seed(42)
    X_train = np.random.randn(n_train, 1, 28, 28).astype(np.float32)
    y_train = np.random.randint(0, 10, n_train)
    X_test  = np.random.randn(n_test,  1, 28, 28).astype(np.float32)
    y_test  = np.random.randint(0, 10, n_test)
    # Add class-specific patterns
    for cls in range(10):
        mask_tr = y_train == cls
        mask_te = y_test  == cls
        X_train[mask_tr, 0, cls*2:cls*2+8, :] += 2.0
        X_test[ mask_te, 0, cls*2:cls*2+8, :] += 2.0
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds  = TensorDataset(torch.FloatTensor(X_test),  torch.LongTensor(y_test))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)
    return train_loader, test_loader


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (output.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            output = model(X_batch)
            loss   = criterion(output, y_batch)
            preds  = output.argmax(1)
            total_loss += loss.item() * len(y_batch)
            correct    += (preds == y_batch).sum().item()
            total      += len(y_batch)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def plot_training_history(train_accs, val_accs, train_losses, val_losses, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(train_accs) + 1)
    ax1.plot(epochs, train_accs, 'b-', label='Train Acc')
    ax1.plot(epochs, val_accs,   'r-', label='Val Acc')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy over Epochs')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax2.plot(epochs, val_losses,   'r-', label='Val Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss over Epochs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix — MNIST Digit Recognizer')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def visualize_predictions(model, loader, device, save_path='sample_predictions.png'):
    """Show sample predictions with confidence."""
    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:16].to(device)
    with torch.no_grad():
        outputs = torch.softmax(model(images_dev), dim=1)
        probs, preds = outputs.max(1)
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        img = images[i, 0].numpy()
        ax.imshow(img, cmap='gray')
        color = 'green' if preds[i].item() == labels[i].item() else 'red'
        ax.set_title(f'Pred: {preds[i].item()} ({probs[i].item():.1%})\nTrue: {labels[i].item()}',
                     color=color, fontsize=8)
        ax.axis('off')
    plt.suptitle('MNIST Predictions (green=correct, red=wrong)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def main():
    print("=" * 60)
    print("MNIST DIGIT RECOGNIZER")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, test_loader, real_data = load_mnist()
    if not real_data:
        train_loader, test_loader = generate_synthetic_mnist()
        print("Running with synthetic data (limited accuracy expected)")

    model     = DigitCNN().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                               epochs=10, steps_per_epoch=len(train_loader))

    train_accs, val_accs   = [], []
    train_losses, val_losses = [], []
    n_epochs = 10

    print(f"\n--- Training for {n_epochs} epochs ---")
    best_acc = 0
    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        va_loss, va_acc, _, _ = evaluate(model, test_loader, criterion, device)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), 'best_mnist_model.pth')
        print(f"Epoch {epoch:2d}/{n_epochs}: "
              f"Train Loss={tr_loss:.4f}, Acc={tr_acc:.4f} | "
              f"Val Loss={va_loss:.4f}, Acc={va_acc:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load('best_mnist_model.pth', map_location=device))
    _, final_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal Test Accuracy: {final_acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(10)]))

    plot_training_history(train_accs, val_accs, train_losses, val_losses)
    plot_confusion_matrix(y_true, y_pred)
    visualize_predictions(model, test_loader, device)

    print(f"\nBest accuracy: {best_acc:.4f}")
    print("Model saved to best_mnist_model.pth")
    print("\n✓ MNIST Digit Recognizer complete!")


if __name__ == '__main__':
    main()
