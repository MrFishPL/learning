import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hiperparametry
image_size = 784 # 28*28
hidden_size = 256
latent_size = 64
batch_size = 100
num_epochs = 200

# Koniecznie bardzo mała wartość!!!
learning_rate = 0.0002

# Ładowanie datasetu MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist = dsets.MNIST(root='./data', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)

# Definicja Generatora
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Wejście: (N, 64)
            nn.Linear(64, 7*7*64),
            nn.ReLU(),
            # Przekształcenie na tensor 3D
            nn.Unflatten(1, (64, 7, 7)),
            # Pierwsza warstwa dekonwolucyjna
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Druga warstwa dekonwolucyjna
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x

# Definicja Dyskryminatora
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Wejście: (N, 1, 28, 28)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # Druga warstwa konwolucyjna
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # Spłaszczenie tensoru
            nn.Flatten(),
            # Warstwa liniowa
            nn.Linear(16*7*7, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        return self.net(x)

# Inicjalizacja sieci
D = Discriminator().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
G = Generator().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Loss i optymalizatory
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = optim.Adam(G.parameters(), lr=learning_rate)

# Trening GANa
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Etykiety prawdziwe i fałszywe
        real_labels = torch.ones(batch_size, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        fake_labels = torch.zeros(batch_size, 1).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        # Trening Dyskryminatora
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        z = torch.randn(batch_size, latent_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        fake_images = G(z)
        # DETACH JEST KLUCZOWE
        outputs = D(fake_images.detach())
        # Tutaj criterion otrzymuje wyjście z modelu i prawidłowe etykiety
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Trening Generatora
        outputs = D(fake_images)
        # Tu etykiety są odrówcone, chcemy żeby generator szedł w kierunku nieprawidłowych etykiet dyskryminatora
        g_loss = criterion(outputs, real_labels)

        # Ponieważ nie rozłączyliśmy fake_images, gradienty popłyną aż do generatora
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')

    # Generowanie i wyświetlanie obrazków co kilka epok
    # (epoch+1) % 20 == 0
    if (epoch+1) % 20 == 0:
        with torch.no_grad():
            z = torch.randn(batch_size, latent_size).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            fake_images = G(z).reshape(-1, 1, 28, 28)
            fake_images = (fake_images + 1) / 2

            fig, ax = plt.subplots(1, 10, figsize=(15, 2))
            for i in range(10):
                ax[i].imshow(fake_images[i][0].cpu().numpy(), cmap='gray')
                ax[i].axis('off')
            plt.show()
