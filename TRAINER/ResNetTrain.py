import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from medmnist import DermaMNIST
from tqdm import tqdm
import numpy as np
import copy

def main():
    # Hyperparameters
    num_epochs = 25  # Increased epochs for better learning
    batch_size = 64  # Increased batch size for stability
    learning_rate = 0.0001
    weight_decay = 1e-4
    num_classes = 7

    # Data Augmentation for Training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Transform for Validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Datasets
    train_dataset = DermaMNIST(split='train', transform=train_transform, download=True)
    val_dataset = DermaMNIST(split='val', transform=val_transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights='IMAGENET1K_V1')

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last convolutional block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace the fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    model = model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # Early Stopping Parameters
    early_stopping_patience = 7
    best_val_loss = np.Inf
    patience_counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            progress_bar.set_postfix({'Loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.squeeze().long().to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = 100 * val_correct / val_total

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Training Loss: {epoch_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), 'best_resnet_model.pth')  # Save the best model
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping")
                break

    # Load Best Model Weights
    model.load_state_dict(best_model_wts)

    # Save the Final Model
    torch.save(model.state_dict(), 'final_resnet_model.pth')

if __name__ == "__main__":
    main()
