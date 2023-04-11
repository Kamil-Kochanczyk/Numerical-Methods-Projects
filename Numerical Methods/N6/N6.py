import numpy as np
import matplotlib.pyplot as plt

def get_error(previous, current):
    return np.linalg.norm(current - previous)

N = 1001
h = 1 / (N - 1)

b = np.zeros(N)
b[0] = 1
b[N - 1] = 1

epsilon = 1e-3

#Metoda relaksacyjna
gamma = 0.25
x_current = np.copy(b)
error = epsilon + 1
counter = 0

while (error > epsilon):
    x_previous = np.copy(x_current)
    for i in range(1, N - 1):
        x_current[i] = x_previous[i] + (gamma * (x_previous[i - 1] - ((h * h + 2) * x_previous[i]) + x_previous[i + 1]))
    error = get_error(x_previous, x_current)
    counter += 1

print("Metoda relaksacyjna: " + str(counter))

#Metoda Jacobiego
x_current_j = np.copy(b)
error = epsilon + 1
counter = 0

while (error > epsilon):
    x_previous_j = np.copy(x_current_j)
    for i in range(1, N - 1):
        x_current_j[i] = (x_previous_j[i - 1] + x_previous_j[i + 1]) / (h * h + 2)
    error = get_error(x_previous_j, x_current_j)
    counter += 1

print("Metoda Jacobiego: " + str(counter))

#Metoda Gaussa-Seidela
x_current_gs = np.copy(b)
error = epsilon + 1
counter = 0

while (error > epsilon):
    x_previous_gs = np.copy(x_current_gs)
    for i in range(1, N - 1):
        x_current_gs[i] = (x_current_gs[i - 1] + x_previous_gs[i + 1]) / (h * h + 2)
    error = get_error(x_previous_gs, x_current_gs)
    counter += 1

print("Metoda Gaussa-Seidela: " + str(counter))

#Successive Over Relaxation
omega = 0.5
x_current_sor = np.copy(b)
error = epsilon + 1
counter = 0

while (error > epsilon):
    x_previous_sor = np.copy(x_current_sor)
    for i in range(1, N - 1):
        x_current_sor[i] = (1 - omega) * x_current_sor[i] + (omega / (h * h + 2)) * (x_current_sor[i - 1] + x_current_sor[i + 1])
    error = get_error(x_previous_sor, x_current_sor)
    counter += 1

print("Successive Over Relaxation: " + str(counter))

x_n = [(n - 1) * h for n in range(0, N)]

plt.plot(x_n, x_current, label="Relaksacyjna")
plt.plot(x_n, x_current_j, label="Jacobiego")
plt.plot(x_n, x_current_gs, label="Gaussa-Seidela")
plt.plot(x_n, x_current_sor, label="Successive Over Relaxation")
plt.xlabel("x_n = (n - 1)h")
plt.ylabel("x_current_n")
plt.title("Metoda")
plt.legend()
plt.show()
