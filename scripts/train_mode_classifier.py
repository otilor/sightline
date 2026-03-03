"""
Train a game mode classifier using auto-labeled frames.

Uses a ResNet18 backbone fine-tuned on the labeled frame dataset.

Usage:
    python3 scripts/train_mode_classifier.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import pathlib
import tempfile
import os
import sys

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = pathlib.Path("data/frames/labeled")
MODEL_OUT = pathlib.Path("data/models/mode_classifier.pt")
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 224
MIN_SAMPLES = 20


def main():
    print(f"Device: {DEVICE}")

    # Filter classes with enough samples
    valid_classes = []
    for cls_dir in sorted(DATA_DIR.iterdir()):
        if cls_dir.is_dir():
            count = len(list(cls_dir.glob("*.png")))
            if count >= MIN_SAMPLES:
                valid_classes.append(cls_dir.name)
                print(f"  {cls_dir.name}: {count} images ✓")
            else:
                print(f"  {cls_dir.name}: {count} images ✗ (skipped, < {MIN_SAMPLES})")

    if len(valid_classes) < 2:
        print("Need at least 2 classes with enough samples!")
        sys.exit(1)

    # Create temp dir with symlinks to valid classes only
    tmp_dir = pathlib.Path(tempfile.mkdtemp())
    for cls in valid_classes:
        os.symlink(str((DATA_DIR / cls).resolve()), str(tmp_dir / cls))

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(str(tmp_dir), transform=transform)
    class_to_idx = full_dataset.class_to_idx

    print(f"\nTotal samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.classes}")

    # Split 80/20
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # Model — ResNet18 with custom head
    num_classes = len(valid_classes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * correct / total
        scheduler.step()

        print(f"  Epoch {epoch+1:>3}/{EPOCHS} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Loss: {train_loss/len(train_loader):.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": valid_classes,
                "class_to_idx": class_to_idx,
                "best_acc": best_acc,
            }, str(MODEL_OUT))

    print(f"\n{'='*50}")
    print(f"Training complete!")
    print(f"Best validation accuracy: {best_acc:.1f}%")
    print(f"Model saved to: {MODEL_OUT}")
    print(f"Classes: {valid_classes}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
