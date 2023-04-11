import numpy as np
import matplotlib.pyplot as plt

epsilon = 1e-10

N = 100

diagonal = 4
below = -1
above = -1

b = np.zeros(N)
b[0] = 1
b[N - 1] = 1

x = np.zeros(N)
r = b
p = r

Ap = np.zeros(N)

while np.linalg.norm(r) > epsilon:
    Ap[0] = (diagonal * p[0]) + (above * p[1])
    for i in range(1, N - 1):
        Ap[i] = (below * p[i - 1]) + (diagonal * p[i]) + (above * p[i + 1])
    Ap[N - 1] = (below * p[N - 2]) + (diagonal * p[N - 1])
    
    alfa = np.dot(r, r) / np.dot(p, Ap)

    x = x + (alfa * p)

    r_next = r - (alfa * Ap)

    beta = np.dot(r_next, r_next) / np.dot(r, r)

    p = r_next + (beta * p)

    r = r_next

arguments = np.array([i for i in range(0, N)])

plt.plot(arguments, x, label="Conjugate gradient method")
plt.xlabel("n = 0, 1, 2, ..., N - 1")
plt.ylabel("x[n]")
plt.legend()
plt.show()
