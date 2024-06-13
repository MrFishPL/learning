import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import random
from torchvision.transforms.functional import to_pil_image
from matplotlib import colormaps
import numpy as np
import PIL
import torch.nn.functional as F
import matplotlib.pyplot as plt

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

def denormalize(tensor, mean, std):
    mean = mean[:, None, None]
    std = std[:, None, None]
    tensor = tensor.cpu() * std.cpu()  + mean.cpu() 
    return tensor.cpu() 

# Definicja transformacji dla danych treningowych
train_transform = transforms.Compose([
    transforms.Resize((120, 80)),  # Dopasowanie rozmiaru do potrzeb modelu
    transforms.RandomHorizontalFlip(),  # Losowe odwracanie obrazu w poziomie
    transforms.ToTensor(),  # Zamiana obrazu na tensor
    # transforms.Normalize(mean=0.5, std=0.5)  # Normalizacja
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizacja
])

# Definicja transformacji dla danych walidacyjnych/testowych
val_transform = transforms.Compose([
    transforms.Resize((120, 80)),  
    transforms.ToTensor(),
    # transforms.Normalize(mean=0.5, std=0.5)
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
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
    def forward(self, x):
        print(x.shape)
        return x

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, [3, 2], padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
 
            nn.Conv2d(16, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, 3, padding=3),

            # Jest w części modeli, ale tu nic nie zmienia
            # nn.Tanh(),
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_encoder(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    g_loss = 0
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)
        # images to tensor obrazów, labels to etykiety klas
        output = model(images)

        labels = labels.view(-1, 1).float()

        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        g_loss += loss.item()

        if i % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}; Step {i}/{len(train_dataloader)}; Loss: {g_loss/len(train_dataloader)}; Train accurancy: {torch.sum(torch.round(output) == labels)/len(images)}")

model.eval()

# FROM: https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569

gradients = None
activations = None

def backward_hook(module, grad_input, grad_output):
  global gradients
  print('Backward hook running...')
  gradients = grad_output
  print(f'Gradients size: {gradients[0].size()}') 

def forward_hook(module, args, output):
  global activations
  print('Forward hook running...')
  activations = output
  print(f'Activations size: {activations.size()}')

backward_hook = model.conv_encoder[-1].register_full_backward_hook(backward_hook, prepend=False)
forward_hook = model.conv_encoder[-1].register_forward_hook(forward_hook, prepend=False)

for i in range(50):
    imgs, labels = next(iter(val_dataloader))
    print(imgs.shape)

    imgs = imgs.to(device)
    labels = labels.to(device)
    model(imgs).backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])

    for i in range(activations.size()[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)

    plt.matshow(heatmap.cpu().detach())

    fig, ax = plt.subplots()
    ax.axis('off') # removes the axis markers

    ax.imshow(to_pil_image(denormalize(imgs, mean, std).reshape(3, 120, 80), mode='RGB'))
    overlay = to_pil_image(heatmap.cpu().detach(), mode='F').resize((120,80), resample=PIL.Image.BICUBIC)

    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)

    # Plot the heatmap on the same axes, 
    # but with alpha < 1 (this defines the transparency of the heatmap)
    ax.imshow(overlay, alpha=0.4, interpolation='nearest', extent=[80, 0, 120, 0])

    # Show the plot
    plt.show()