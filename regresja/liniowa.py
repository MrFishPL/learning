import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Przykładowe dane
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 3, 2, 5, 4])

# Tworzenie modelu regresji liniowej
model = LinearRegression()
model.fit(X, y)

# Przewidywanie wartości
y_pred = model.predict(X)

# Wizualizacja
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Regresja liniowa')
plt.show()
