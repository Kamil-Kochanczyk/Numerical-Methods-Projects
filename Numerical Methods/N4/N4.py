import numpy as np
import matplotlib.pyplot as plt
import math

N = 1000

def f(x):
    return 1 / (1 + 25 * x * x)

def f_k(k):
    return f((2 * k - N) / N)

g = (1.5 * N * N) * np.array([f_k(k) - 2 * f_k(k + 1) + f_k(k + 2) for k in range(0, N - 1)])

u = np.zeros(N - 1)
u[0] = 1
u[N - 2] = 1

v = np.copy(u)

main_diagonal = np.full(N - 1, 4, dtype="float64")
main_diagonal[0] = 3
main_diagonal[N - 2] = 3

diagonal_below = np.full(N - 2, 1, dtype="float64")

main_diagonal[0] = math.sqrt(main_diagonal[0])
diagonal_below[0] = diagonal_below[0] / main_diagonal[0]

for j in range(1, N - 2):
    for i in range(0, 2):
        if i == 0:
            main_diagonal[j] = math.sqrt(main_diagonal[j] - diagonal_below[j - 1] * diagonal_below[j - 1])
        else:
            diagonal_below[j] = diagonal_below[j] / main_diagonal[j]

main_diagonal[N - 2] = math.sqrt(main_diagonal[N - 2] - diagonal_below[N - 3] * diagonal_below[N - 3])

def solve(vector, transpose):
    result = []

    if transpose:
        result.append(vector[vector.size - 1] / main_diagonal[vector.size - 1])
        for k in range(vector.size - 2, -1, -1):
            result.append((vector[k] - diagonal_below[k] * result[(vector.size - 2) - k]) / main_diagonal[k])
        result = result[::-1]

    else:
        result.append(vector[0] / main_diagonal[0])
        for k in range(1, vector.size):
            result.append((vector[k] - diagonal_below[k - 1] * result[k - 1]) / main_diagonal[k])

    return np.array(result)

p = solve(u, False)
y = solve(p, True)

q = solve(g, False)
z = solve(q, True)

solution = z - ((np.dot(v, z) / (1 + np.dot(v, y))) * y)

x_k = np.array([(2 * k - N) / N for k in range(0, N - 1)])

plt.plot(x_k, solution)
plt.xlabel("x_k")
plt.ylabel("solution_k")
plt.title("(x_k, solution_k)")
plt.show()
