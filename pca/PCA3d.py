import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie danych
digits = load_digits()
X = digits.data
y = digits.target

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA z 3 składowymi
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# Tworzenie DataFrame dla łatwiejszej wizualizacji
df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
df_pca['target'] = y

# Wizualizacja wyników PCA w 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for target in np.unique(y):
    subset = df_pca[df_pca['target'] == target]
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=target)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA na zbiorze danych cyfr')
ax.legend()
plt.show()
