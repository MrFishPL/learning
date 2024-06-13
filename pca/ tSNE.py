import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Wczytanie danych
digits = load_digits()
X = digits.data
y = digits.target

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE z 2 wymiarami
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Tworzenie DataFrame dla łatwiejszej wizualizacji
df_tsne = pd.DataFrame(data=X_tsne, columns=['Dim1', 'Dim2'])
df_tsne['target'] = y

# Wizualizacja wyników t-SNE
plt.figure(figsize=(10, 8))
for target in np.unique(y):
    subset = df_tsne[df_tsne['target'] == target]
    plt.scatter(subset['Dim1'], subset['Dim2'], label=target)
    
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE na zbiorze danych cyfr')
plt.legend()
plt.grid()
plt.show()
