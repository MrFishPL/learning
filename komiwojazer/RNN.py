with open("data2.txt") as d:
  distances = list(map(lambda x: list(map(float, filter(lambda y: y != "", x[0:-1].split(" ")))), d.readlines()))

N = len(distances)

def path_length(path):
  global distances
  sum = 0
  l = len(path)
  for i in range(len(path)):
    sum += distances[path[i]][path[(i + 1) % l]]

  return sum

best_score_ever = float("inf")
best_path_ever = None

for k in range(N):
  i = k
  path = [i]
  visited_num = 0
  visited = [False for _ in range(N)]

  while visited_num < N:
    visited[i] = True

    last_d = float("inf")
    last_n = None
    for j in range(N):
      if distances[i][j] < last_d and not visited[j]:
        last_d = distances[i][j]
        last_n = j

    visited_num += 1
    if last_n == None:
      continue
    path.append(last_n)
    i = last_n

  p = path_length(path)

  if p < best_score_ever:
    best_score_ever = p
    best_path_ever = path

print(best_score_ever)
print(best_path_ever)