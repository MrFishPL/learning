import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Przykładowe dane
# X = np.array([[1], [2], [3], [4], [5]])
# y = np.array([1, 3, 2, 5, 4])

X = np.load("X_valid.npy")
y = np.load("y_valid.npy")

# Tworzenie modelu regresji liniowej
model = LinearRegression()
model.fit(X, y)

# Przewidywanie wartości
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# R2 = 1 dla liniowej zależności
# R2 > 90 dobre
print(mse, r2)

# # Wizualizacja
# plt.scatter(X, y, color='blue')
# plt.plot(X, y_pred, color='red')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Regresja liniowa')
# plt.show()
