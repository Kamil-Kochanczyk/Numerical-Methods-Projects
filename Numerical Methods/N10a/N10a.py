import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["axes.grid"] = True

def f1(x):
    return 1 / (1 + x * x)

def f2(x):
    return np.exp(-x * x)

def l(i, x, X):
    result = np.ones(x.size)
    j = np.arange(0, X.size)
    j = j[j != i]
    for number in j:
        result *= (x - X[number]) / (X[i] - X[number])
    return result

def interpolate(f, x, X):
    interpolation = np.zeros(x.size)
    for number in np.arange(0, X.size):
        interpolation += f(X[number]) * l(number, x, X)
    return interpolation

def get_max_error(N, f, equidistance):
    x_local = np.linspace(-5, 5, 500)
    if equidistance:
        X_local = np.linspace(-5, 5, N)
    else:
        n = np.arange(0, N)
        X_local = 5 * np.cos((n * np.pi) / (N - 1))
    return np.abs(f(x_local) - interpolate(f, x_local, X_local)).max()





N = 20
X = np.linspace(-5, 5, N)
x = np.linspace(-5, 5, 500)

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, interpolate(f1, x, X), label="interpolated $f_1(x)$")
plt.scatter(X, f1(X), c="red")
plt.legend()
plt.show()

plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, interpolate(f2, x, X), label="interpolated $f_2(x)$")
plt.scatter(X, f2(X), c="red")
plt.legend()
plt.show()

Ns = np.arange(1, 26)

f1_errors = np.array([get_max_error(value, f1, True) for value in Ns])
f2_errors = np.array([get_max_error(value, f2, True) for value in Ns])

best_N_f1 = f1_errors.argmin() + 1
best_N_f2 = f2_errors.argmin() + 1

print("Smallest sigma for equidistant nodes")
print("f1: N = {0}, sigma = {1}".format(best_N_f1, f1_errors[best_N_f1 - 1]))
print("f2: N = {0}, sigma = {1}\n".format(best_N_f2, f2_errors[best_N_f2 - 1]))

x = np.linspace(-6, 6, 1000)
X1 = np.linspace(-5, 5, best_N_f1)
X2 = np.linspace(-5, 5, best_N_f2)

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, interpolate(f1, x, X1), label="interpolated $f_1(x)$")
plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, interpolate(f2, x, X2), label="interpolated $f_2(x)$")
plt.legend()
plt.title("Interpolation, best N, smallest sigma")
plt.show()





N = 20
n = np.arange(0, N)
X = 5 * np.cos((n * np.pi) / (N - 1))
x = np.linspace(-5, 5, 500)

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, interpolate(f1, x, X), label="interpolated $f_1(x)$")
plt.scatter(X, f1(X), c="red")
plt.legend()
plt.show()

plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, interpolate(f2, x, X), label="interpolated $f_2(x)$")
plt.scatter(X, f2(X), c="red")
plt.legend()
plt.show()

Ns = np.arange(2, 26)

f1_errors = np.array([get_max_error(value, f1, False) for value in Ns])
f2_errors = np.array([get_max_error(value, f2, False) for value in Ns])

best_N_f1 = f1_errors.argmin() + 2
best_N_f2 = f2_errors.argmin() + 2

print("Smallest sigma for nodes dense on the edges")
print("f1: N = {0}, sigma = {1}".format(best_N_f1, f1_errors[best_N_f1 - 2]))
print("f2: N = {0}, sigma = {1}\n".format(best_N_f2, f2_errors[best_N_f2 - 2]))

x = np.linspace(-5.25, 5.25, 1000)

N = best_N_f1
n = np.arange(0, N)
X1 = 5 * np.cos((n * np.pi) / (N - 1))

N = best_N_f2
n = np.arange(0, N)
X2 = 5 * np.cos((n * np.pi) / (N - 1))

plt.plot(x, f1(x), label="$f_1(x)$")
plt.plot(x, interpolate(f1, x, X1), label="interpolated $f_1(x)$")
plt.plot(x, f2(x), label="$f_2(x)$")
plt.plot(x, interpolate(f2, x, X2), label="interpolated $f_2(x)$")
plt.legend()
plt.title("Interpolation, best N, smallest sigma")
plt.show()
