import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sin(x)
f_prime = lambda x: np.cos(x)

def dha(x, h):
    return (f(x + h) - f(x)) / h

def dhb(x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def dhc(x, h):
    return (8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h) - f(x + 2 * h)) / (12 * h)

step = 1e-5
start = 1e-16
stop = 1 + step
h_values = np.arange(start, stop, step)

error = lambda x, h, dh: abs(dh(x, h) - f_prime(x))

x = 1

dha_errors = np.array([error(x, h, dha) for h in h_values])
dhb_errors = np.array([error(x, h, dhb) for h in h_values])
dhc_errors = np.array([error(x, h, dhc) for h in h_values])

indexes = [np.argmin(dha_errors), np.argmin(dhb_errors), np.argmin(dhc_errors)]

a = ("dha", h_values[indexes[0]], dha_errors[indexes[0]])
b = ("dhb", h_values[indexes[1]], dhb_errors[indexes[1]])
c = ("dhc", h_values[indexes[2]], dhc_errors[indexes[2]])

triplets = [a, b, c]

for triplet in triplets:
    print("{0}: {1:e}, {2:e}".format(triplet[0], triplet[1], triplet[2]))

plt.plot(h_values, dha_errors, label="dha")
plt.plot(h_values, dhb_errors, label="dhb")
plt.plot(h_values, dhc_errors, label="dhc")

plt.xlabel("h")
plt.xscale("log")
plt.ylabel("|Dhf(x) - f'(x)|")
plt.yscale("log")
plt.title("Errors")
plt.legend()
plt.show()
