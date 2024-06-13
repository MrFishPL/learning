from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Wczytanie przykładowego zestawu danych Iris
iris = load_iris()
X, y = iris.data, iris.target

print(len(iris.target))

# Inicjalizacja modelu drzewa decyzyjnego
clf = DecisionTreeClassifier()

print("trenowanie")
# Trenowanie modelu
clf.fit(X, y)
print("trenowanie")

# Wizualizacja drzewa decyzyjnego
plt.figure(figsize=(12,8))
tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()

# https://chatgpt.com/share/250a4312-ee66-4110-bac9-00753ddb469e