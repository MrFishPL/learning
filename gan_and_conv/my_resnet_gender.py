import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random

"""
0.9632 - dokładność po 1. epoce dla resnet18
0.9628 - dokładność po 1. epoce dla resnet34
0.9743 - dokładność po 2. epoce dla resnet34
0.9670 - dokładność po 1. epoce dla resnet50
"""

# Definicja transformacji dla danych treningowych
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Dopasowanie rozmiaru do potrzeb modelu
    transforms.RandomHorizontalFlip(),  # Losowe odwracanie obrazu w poziomie
    transforms.ToTensor(),  # Zamiana obrazu na tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])

# Definicja transformacji dla danych walidacyjnych/testowych
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])

# Ścieżki do danych
train_data_path = "archive/Training"
val_data_path = "archive/Validation"

# Tworzenie Dataset dla danych treningowych i walidacyjnych
train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
val_dataset = datasets.ImageFolder(root=val_data_path, transform=val_transform)

# Tworzenie DataLoader dla danych treningowych i walidacyjnych
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet34(pretrained=True).to(device)

num_ftrs = model.fc.in_features

model.fc = torch.nn.Sequential(
    nn.Linear(num_ftrs, 1),
    nn.Sigmoid(),
).to(device)

# Dostrajamy całą sieć oraz trenujemy nowe warstwy
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

model.train()

num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        labels = labels.view(-1, 1).float()

        optimizer.zero_grad()
        
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)

        if i%20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}; Step: {i + 1}/{len(train_dataloader)} Train Loss: {loss};")

    epoch_loss = running_loss / len(train_dataloader.dataset)
    
# Walidacja modelu
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()
        outputs = model(inputs)
        preds = torch.round(outputs)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

val_accuracy = correct / total
print(f'Validation Accuracy: {val_accuracy:.4f}')

