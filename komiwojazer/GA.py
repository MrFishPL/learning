import random
import math

POPULATION = 100
THREADSHOLD = 0.4
BEST_NUMBER = math.floor(POPULATION * THREADSHOLD)
PAIRS_AMOUNT = POPULATION - BEST_NUMBER
GENERATIONS = 10000

with open("data.txt") as d:
  distances = list(map(lambda x: list(map(float, filter(lambda y: y != "", x[0:-1].split(" ")))), d.readlines()))

def choice(probability):
  return random.random() < probability

# Krzyżowanie permutacji metodą PMX
def pmx(perm1, perm2, p1, p2):
  c1 = [None for _ in range(len(perm1))]
  c2 = [None for _ in range(len(perm2))]

  c1[p1:p2] = perm2[p1:p2]
  c2[p1:p2] = perm1[p1:p2]

  for i in range(0, len(perm1)):
    if i >= p1 and i < p2:
      continue

    el = perm1[i]
    while el in c1:
      idx = c1.index(el)
      el = c2[idx]

    c1[i] = el

  for i in range(0, len(perm2)):
    if i >= p1 and i < p2:
      continue

    el = perm2[i]
    while el in c2:
      idx = c2.index(el)
      el = c1[idx]

    c2[i] = el

  return c1, c2

# Wprowadza mutację przez zamianę miejscami czterech elementów
def mutate(perm):
  perm_len = len(perm)

  i, j = random.randint(0, perm_len - 1), random.randint(0, perm_len - 1)
  perm[i], perm[j] = perm[j], perm[i]

  i, j = random.randint(0, perm_len - 1), random.randint(0, perm_len - 1)
  perm[i], perm[j] = perm[j], perm[i]

# Obliczanie długości ścieżki
def path_length(path):
  global distances
  sum = 0
  l = len(path)
  for i in range(len(path)):
    sum += distances[path[i]][path[(i + 1) % l]]

  return sum

# Tworzenie początkowej losowej populacji
towns = [i for i in range(len(distances))]
perms = [random.sample(towns, len(towns)) for _ in range(POPULATION)]

for i in range(GENERATIONS):
  if (i % 100 == 0):
    print("Pokolenie", i + 1)

  # Sortowanie populacji wg dostosowania
  perms = sorted(perms, key=path_length)[0:BEST_NUMBER]

  # Tworzenie par spośród najlepszych osobników
  pairs = []
  for j in range(PAIRS_AMOUNT):
    x, y = random.randint(0, BEST_NUMBER - 1), random.randint(0, BEST_NUMBER - 1)
    pairs.append((min(x, y), max(x, y)))
  
  # Tworzenie skrzyżowań i mutowanie
  for pair in pairs:
    parent1 = perms[pair[0]]  
    parent2 = perms[pair[1]]  
    perm_len = len(parent1)

    x, y = random.randint(0, perm_len - 1), random.randint(0, perm_len - 1)
    x, y = min(x, y), max(x, y)

    ch1, ch2 = pmx(parent1, parent2, x, y)

    if choice(0.2):
      mutate(ch1)

    if choice(0.2):
      mutate(ch2)

    perms.append(ch1)
    perms.append(ch2)

perms = sorted(perms, key=path_length)
print(path_length(perms[0]))