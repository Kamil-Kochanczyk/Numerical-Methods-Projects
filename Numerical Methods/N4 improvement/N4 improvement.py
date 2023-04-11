import numpy as np
import matplotlib.pyplot as plt

N = 1000
h = 2 / N

def f(x):
    return 1 / (1 + 25 * x * x)

def fk(k):
    return f(-1 + k * h)

k = np.arange(0, N - 1)
g = (6 / (h * h)) * (fk(k) - 2 * fk(k + 1) + fk(k + 2))

u = np.zeros(N - 1)
u[0] = 1
u[N - 2] = 1

v = np.copy(u)

main_diagonal = np.full(N - 1, 4, dtype="float64")
main_diagonal[0] = 3
main_diagonal[N - 2] = 3

diagonal_below = np.full(N - 2, 1, dtype="float64")

main_diagonal[0] = np.sqrt(main_diagonal[0])
diagonal_below[0] = diagonal_below[0] / main_diagonal[0]

for j in np.arange(1, N - 2):
    main_diagonal[j] = np.sqrt(main_diagonal[j] - diagonal_below[j - 1] * diagonal_below[j - 1])
    diagonal_below[j] = diagonal_below[j] / main_diagonal[j]

main_diagonal[N - 2] = np.sqrt(main_diagonal[N - 2] - diagonal_below[N - 3] * diagonal_below[N - 3])

def solve(vector, transpose):
    size = vector.size
    result = np.empty(size)
    
    if transpose:
        result[0] = vector[size - 1] / main_diagonal[size - 1]
        for k in np.arange(size - 2, -1, -1):
            result[size - 1 - k] = (vector[k] - diagonal_below[k] * result[size - 2 - k]) / main_diagonal[k]
        result = result[::-1]
    
    else:
        result[0] = vector[0] / main_diagonal[0]
        for k in np.arange(1, size):
            result[k] = (vector[k] - diagonal_below[k - 1] * result[k - 1]) / main_diagonal[k]
    
    return result

p = solve(u, False)
y = solve(p, True)

q = solve(g, False)
z = solve(q, True)

solution = z - ((np.dot(v, z) / (1 + np.dot(v, y))) * y)

k = np.arange(0, N - 1)
xk = -1 + k * h

plt.plot(xk, solution)
plt.xlabel("$x_k$")
plt.ylabel("$solution_k$")
plt.title("$(x_k, solution_k)$")
plt.show()
