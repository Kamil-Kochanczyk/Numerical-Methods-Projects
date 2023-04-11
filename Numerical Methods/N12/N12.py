import numpy as np

In = np.empty(31)
In[0] = 1 - (1 / np.e)

for n in range(1, 31):
    In[n] = 1 - (n * In[n - 1])

print(In)
print()

def f(x, n):
    return np.exp(x) * np.float_power(x, n)

def trapezoidal_rule(n, i):
    h = 1 / i
    X = np.arange(0, 1 + h, h)
    j = np.arange(1, X.size - 1)
    return (1 / 2) * h * (f(X[0], n) + 2 * np.sum(f(X[j], n)) + f(X[-1], n))

In = np.array([(1 / np.e) * trapezoidal_rule(n, 15000) for n in range(0, 31)])
print(In)
print()

"""
def f_prim_prim(x, n):
    return np.exp(x) * (np.float_power(x, n) + 2 * n * np.float_power(x, n - 1) + n * (n - 1) * np.float_power(x, n - 2))

def get_max_error(n, i):
    return f_prim_prim(1, n) / (12 * i * i)

epsilon = 1e-6

k = np.arange(0, 31)
max_errors = get_max_error(k, 15000)
print(max_errors)
print(np.all(max_errors < epsilon))
print()
"""
