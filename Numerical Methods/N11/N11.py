import numpy as np

epsilon = 1e-6
a = 0
b = np.pi

def f(y):
    c = np.cos(y)
    return np.exp(c * c)

def trapezoidal_rule(h):
    Y = np.arange(a, b + h, h)
    i = np.arange(1, Y.size - 1)
    return (1 / 2) * h * (f(Y[0]) + 2 * np.sum(f(Y[i])) + f(Y[-1]))

def s(k, h):
    i = np.arange(0, k)
    Y = a + (2 * i + 1) * h
    return np.sum(f(Y))

def trapezoidal_error(k):
    return (2 * np.e * np.pi * np.pi * np.pi) / (12 * k * k)

k = 2
h = (b - a) / k
N = k + 1
I = trapezoidal_rule(h)
error = trapezoidal_error(k)

while error > epsilon:
    h = h / 2
    I = (1 / 2) * I + h * s(k, h)
    k = 2 * k
    N = k + 1
    error = trapezoidal_error(k)

print("I = {0}, N = {1}\n".format(I, N))

def simpsons_rule(h):
    Y = np.arange(a, b + h, h)
    k = (b - a) / (2 * h)
    i = np.arange(1, k + 1, dtype="int32")
    j = np.arange(1, k, dtype="int32")
    return (h / 3) * (f(Y[0]) + 4 * np.sum(f(Y[2 * i - 1])) + 2 * np.sum(f(Y[2 * j])) + f(Y[-1]))

def simpsons_error(h):
    return (1 / 180) * h * h * h * h * (b - a) * 20 * np.e

k = 1
N = 2 * k + 1
h = (b - a) / (2 * k)
I = simpsons_rule(h)
error = simpsons_error(h)

while error > epsilon:
    h = h / 2
    I = simpsons_rule(h)
    error = simpsons_error(h)
    k = (b - a) / (2 * h)
    N = 2 * k + 1

print("I = {0}, N = {1}\n".format(I, N))
