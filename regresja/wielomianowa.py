import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generowanie przykładowych danych
# np.random.seed(0)
# X = np.sort(5 * np.random.rand(100, 1), axis=0)
# y = (0.5 * X**3 - 2 * X**2 + X + 10 * np.sin(X) + np.exp(0.5 * X) + 2 * np.cos(1.5 * X)).ravel() + np.random.normal(0, 1, X.shape[0])

# Generowanie danych wejściowych X w 7 wymiarach
dimension = 7

# Generowanie danych wejściowych w zależności od liczby wymiarów
X = np.sort(5 * np.random.rand(100, dimension), axis=0)

# Definicja funkcji wyjściowej y w zależności od liczby wymiarów
y = (0.5 * X[:, 0]**3 - 2 * X[:, 1]**2 + X[:, 2] + 10 * np.sin(X[:, 3]) + 
     np.exp(0.5 * X[:, 4]) + 2 * np.cos(1.5 * X[:, 5]) - 3 * X[:, 6]**1.5 + 
     np.sum([np.random.normal(0, 1) for _ in range(max(0, dimension - 7))], axis=0))

# Dodanie szumu do funkcji wyjściowej
y += np.random.normal(0, 1, X.shape[0])

# Tworzenie wielomianowych cech (stopień wielomianu = 10)
# R2 > 90 git
poly = PolynomialFeatures(degree=20)
X_poly = poly.fit_transform(X)

# Trenowanie modelu regresji wielomianowej
model = LinearRegression()
model.fit(X_poly, y)

# Przewidywanie wartości
y_pred = model.predict(X_poly)

# Ocena modelu
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Wizualizacja wyników
# plt.scatter(X, y, color='blue', label='Dane rzeczywiste')
# plt.plot(X, y_pred, color='red', label='Model regresji wielomianowej')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.title('Regresja wielomianowa dla skomplikowanej funkcji')
# plt.legend()
# plt.show()
