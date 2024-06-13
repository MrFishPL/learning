import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Wczytanie danych
digits = load_digits()
X = digits.data
y = digits.target

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA z 2 składowymi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Tworzenie DataFrame dla łatwiejszej wizualizacji
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y

# Wizualizacja wyników PCA
plt.figure(figsize=(10, 8))
for target in np.unique(y):
    subset = df_pca[df_pca['target'] == target]
    plt.scatter(subset['PC1'], subset['PC2'], label=target)
    
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA na zbiorze danych cyfr')
plt.legend()
plt.grid()
plt.show()
