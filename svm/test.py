import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Funkcja, która generuje punkty na podstawie wielomianu
def generate_points(poly_func, x_range, num_points):
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    y_values = poly_func(x_values)
    return x_values, y_values

# Funkcja, która dodaje szum do punktów
def add_noise(y_values, noise_level):
    noise = np.random.normal(scale=noise_level, size=y_values.shape)
    return y_values + noise

# Przykładowy wielomian: f(x) = 2*x**5 - 3*x**4 + 4*x**3 - 5*x**2 + 6*x - 7
def polynomial(x):
    return 2*x**5 - 3*x**4 + 4*x**3 - 5*x**2 + 6*x - 7

# Funkcja, która tworzy wzór wielomianu
def polynomial_formula(coefficients):
    terms = []
    for i, coef in enumerate(coefficients):
        if i == 0:
            terms.append(f"{coef:.2f}")
        elif i == 1:
            terms.append(f"{coef:.2f}*x")
        else:
            terms.append(f"{coef:.2f}*x^{i}")
    return " + ".join(terms)

# Zakres x
x_range = (-5, 5)

# Wygeneruj punkty na podstawie wielomianu
num_points = 20
x_values, y_values = generate_points(polynomial, x_range, num_points)

# Dodaj większe szumienie do punktów
noise_level = 500
noisy_y_values = add_noise(y_values, noise_level)

# Przeprowadź regresję wielomianową
poly_features = PolynomialFeatures(degree=5)
x_values_reshaped = x_values.reshape(-1, 1)
x_poly = poly_features.fit_transform(x_values_reshaped)
model = LinearRegression()
model.fit(x_poly, noisy_y_values)
predicted_y_values = model.predict(x_poly)

# Wydobycie współczynników wielomianu
coefficients = model.coef_
coefficients[0] = model.intercept_  # Zastąpienie pierwszego współczynnika wyrazem wolnym
poly_eq = polynomial_formula(coefficients)

# Wyświetl punkty za pomocą Matplotlib
plt.figure(figsize=(8, 6))
plt.scatter(x_values, noisy_y_values, color='blue', label='Noisy Points')
plt.plot(x_values, polynomial(x_values), color='red', label='Original Polynomial')
plt.plot(x_values, predicted_y_values, color='green', linestyle='--', label='Predicted Polynomial')
plt.title('Noisy Points and Polynomial Regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Wyświetlenie wzoru wielomianu
print("Wzór dopasowanego wielomianu:")
print(poly_eq)
