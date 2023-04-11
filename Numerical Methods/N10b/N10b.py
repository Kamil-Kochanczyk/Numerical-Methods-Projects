import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = True

def f1(x):
    return 1 / (1 + x * x)

def f2(x):
    return np.exp(-x * x)

N = 10
X = np.linspace(-5, 5, N)
args_per_interval = 250
total_args = (N - 1) * args_per_interval
x = np.linspace(-5, 5, total_args)
h = 10 / (N - 1)

main_diagonal = np.full(N - 2, 4, dtype="float64")
diagonal_above = np.ones(N - 3, dtype="float64")
diagonal_below = np.ones(N - 3, dtype="float64")

n = np.arange(1, N - 1)

f1n = f1(X)
g1 = (6 / (h * h)) * (f1n[n - 1] - (2 * f1n[n]) + f1n[n + 1])

f2n = f2(X)
g2 = (6 / (h * h)) * (f2n[n - 1] - (2 * f2n[n]) + f2n[n + 1])

diagonal_below[0] = diagonal_below[0] / main_diagonal[0]

for j in np.arange(1, N - 3):
    main_diagonal[j] = main_diagonal[j] - diagonal_below[j - 1] * diagonal_above[j - 1]
    diagonal_below[j] = diagonal_below[j] / main_diagonal[j]

main_diagonal[N - 3] = main_diagonal[N - 3] - diagonal_below[N - 4] * diagonal_above[N - 4]

ksi1 = np.zeros(N)
ksi1[1] = g1[0]
for k in np.arange(2, N - 1):
    ksi1[k] = g1[k - 1] - diagonal_below[k - 2] * ksi1[k - 1]

ksi1[N - 2] = ksi1[N - 2] / main_diagonal[N - 3]
for k in np.arange(N - 3, 0, -1):
    ksi1[k] = (ksi1[k] - diagonal_above[k - 1] * ksi1[k + 1]) / main_diagonal[k - 1]

ksi2 = np.zeros(N)
ksi2[1] = g2[0]
for k in np.arange(2, N - 1):
    ksi2[k] = g2[k - 1] - diagonal_below[k - 2] * ksi2[k - 1]

ksi2[N - 2] = ksi2[N - 2] / main_diagonal[N - 3]
for k in np.arange(N - 3, 0, -1):
    ksi2[k] = (ksi2[k] - diagonal_above[k - 1] * ksi2[k + 1]) / main_diagonal[k - 1]

def get_interval(j):
    return np.linspace(X[j], X[j + 1], args_per_interval)

def get_cubic_spline(j, f, ksi):
    x = get_interval(j)
    A = (X[j + 1] - x) / (X[j + 1] - X[j])
    B = (x - X[j]) / (X[j + 1] - X[j])
    C = (1 / 6) * ((A * A * A) - A) * (X[j + 1] - X[j]) * (X[j + 1] - X[j])
    D = (1 / 6) * ((B * B * B) - B) * (X[j + 1] - X[j]) * (X[j + 1] - X[j])
    return A * f(X[j]) + B * f(X[j + 1]) + C * ksi[j] + D * ksi[j + 1]

def spline_interpolation(f, ksi):
    interpolation = np.empty(0)
    for j in np.arange(0, N - 1):
        interpolation = np.append(interpolation, get_cubic_spline(j, f, ksi))
    return interpolation

f1_interpolation = spline_interpolation(f1, ksi1)
f2_interpolation = spline_interpolation(f2, ksi2)

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, f1_interpolation, label="spline interpolation")
plt.scatter(X, f1(X), c="red")
plt.legend()
plt.show()

plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, f2_interpolation, label="spline interpolation")
plt.scatter(X, f2(X), c="red")
plt.legend()
plt.show()





N = 10
X = np.linspace(-5, 5, N)
x = np.linspace(-10, 10, 1000)
h = 10 / (N - 1)

def sinc(n):
    result = np.empty(x.size)
    result[x != X[n]] = np.sin(np.pi * (x[x != X[n]] - X[n]) / h) / (np.pi * (x[x != X[n]] - X[n]) / h)
    result[x == X[n]] = 1
    result[x < -5] = 0
    result[x > 5] = 0
    return result

def sinc_interpolation(f):
    interpolation = np.zeros(x.size)
    for j in np.arange(0, N):
        interpolation += f(X[j]) * sinc(j)
    return interpolation

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, sinc_interpolation(f1), label="sinc interpolation")
plt.scatter(X, f1(X), c="red")
plt.legend()
plt.show()

plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, sinc_interpolation(f2), label="sinc interpolation")
plt.scatter(X, f2(X), c="red")
plt.legend()
plt.show()
