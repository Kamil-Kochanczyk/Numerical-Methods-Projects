import numpy as np
import matplotlib.pyplot as plt

N = 1001
h = 1 / (N - 1)

u = []
u.append(1)
u.append(2 / (2 - (h * h)))
u.append((3 * u[0]) - (4 * u[1]))

for i in range(3, N):
    u.append(((h * h + 2) * u[i - 1]) - u[i - 2])

x = [(n - 1) * h for n in range(0, N)]

plt.plot(x, u)
plt.xlabel("x")
plt.ylabel("u")
plt.title("(x, u)")
plt.show()
