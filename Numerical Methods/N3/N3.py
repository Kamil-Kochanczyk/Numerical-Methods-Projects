import numpy as np
import matplotlib.pyplot as plt

N = 1000

def f(x):
    return 1 / (1 + 25 * x * x)

def f_k(k):
    return f((2 * k - N) / N)

g = (1.5 * N * N) * np.array([f_k(i) - 2 * f_k(i + 1) + f_k(i + 2) for i in range(0, N - 1)])

main_diagonal = np.full(N - 1, 4, dtype="float64")
diagonal_above = np.ones(N - 2, dtype="float64")
diagonal_below = np.ones(N - 2, dtype="float64")

diagonal_below[0] = diagonal_below[0] / main_diagonal[0]

for j in range(1, N - 2):
    for i in range(0, 2):
        if i == 0:
            main_diagonal[j] = main_diagonal[j] - diagonal_below[j - 1] * diagonal_above[j - 1]
        else:
            diagonal_below[j] = diagonal_below[j] / main_diagonal[j]

main_diagonal[N - 2] = main_diagonal[N - 2] - diagonal_below[N - 3] * diagonal_above[N - 3]

u = [g[0]]

for k in range(1, N - 1):
    u.append(g[k] - diagonal_below[k - 1] * u[k - 1])

u[N - 2] = u[N - 2] / main_diagonal[N - 2]

for k in range(N - 3, -1, -1):
    u[k] = (u[k] - diagonal_above[k] * u[k + 1]) / main_diagonal[k]

x = [(2 * k - N) / N for k in range(0, N - 1)]

plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u")
plt.title("(x, u)")
plt.show()

"""
import numpy as np
import matplotlib.pyplot as plt

N = 1000

def f(x):
    return 1 / (1 + 25 * x * x)

def f_k(k):
    return f((2 * k - N) / N)

g = (1.5 * N * N) * np.array([f_k(i) - 2 * f_k(i + 1) + f_k(i + 2) for i in range(0, N - 1)])

A = np.diag(np.full(N - 1, 4)) + np.diag(np.ones(N - 2), 1) + np.diag(np.ones(N - 2), -1)

A[1, 0] = A[1, 0] / A[0, 0]

for j in range(1, N - 2):
    for i in range(j, j + 2):
        if i == j:
            A[i, j] = A[i, j] - A[i, j - 1] * A[i - 1, j]
        else:
            A[i, j] = A[i, j] / A[i - 1, j]

A[N - 2, N - 2] = A[N - 2, N - 2] - A[N - 2, N - 3] * A[N - 3, N - 2]

u = [g[0]]

for k in range(1, N - 1):
    u.append(g[k] - A[k, k - 1] * u[k - 1])

u[N - 2] = u[N - 2] / A[N - 2, N - 2]

for k in range(N - 3, -1, -1):
    u[k] = (u[k] - A[k, k + 1] * u[k + 1]) / A[k, k]

x = [(2 * k - N) / N for k in range(0, N - 1)]

plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u")
plt.title("(x, u)")
plt.show()
"""
