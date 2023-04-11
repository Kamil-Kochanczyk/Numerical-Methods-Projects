import numpy as np
import matplotlib.pyplot as plt

N = 1001
h = 1 / (N - 1)

g = np.zeros(N)
g[0] = 1
g[N - 1] = 1

main_diagonal = np.full(N, (h * h + 2), dtype="float64")
main_diagonal[0] = 1
main_diagonal[N - 1] = 1

diagonal_above = -1 * np.ones(N - 1, dtype="float64")
diagonal_above[0] = 0

diagonal_below = -1 * np.ones(N - 1, dtype="float64")
diagonal_below[N - 2] = 0

diagonal_below[0] = diagonal_below[0] / main_diagonal[0]

for j in range(1, N - 1):
    for i in range(0, 2):
        if i == 0:
            main_diagonal[j] = main_diagonal[j] - diagonal_below[j - 1] * diagonal_above[j - 1]
        else:
            diagonal_below[j] = diagonal_below[j] / main_diagonal[j]

main_diagonal[N - 1] = main_diagonal[N - 1] - diagonal_below[N - 2] * diagonal_above[N - 2]

u = [g[0]]

for k in range(1, N):
    u.append(g[k] - diagonal_below[k - 1] * u[k - 1])

u[N - 1] = u[N - 1] / main_diagonal[N - 1]

for k in range(N - 2, -1, -1):
    u[k] = (u[k] - diagonal_above[k] * u[k + 1]) / main_diagonal[k]

x = [(n - 1) * h for n in range(0, N)]

plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u")
plt.title("(x, u)")
plt.show()
