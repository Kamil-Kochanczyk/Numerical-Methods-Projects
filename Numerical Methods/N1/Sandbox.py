"""
This is a sandbox. The main program is N1.py.
I used this program to test whether functions in N1.py work correctly or not.
I also used it to test if optimized function f in N1.py runs quicker than not-optimized function g
"""

from math import sin, cos, exp
import time, N1

def g(x, N):
    result = 0
    for n in range(0, N + 1):
        result += ((sin(n * (x**4)))**2) * exp(-1 * n) + cos(n * (x**4)) * exp(-4 * n)
    return result

x = 1
last_n, N = N1.find(x)

print("x = " + str(x))
print("N = " + str(N) + "\n")

for n in range(last_n - 2, last_n + 3):
    result_f = N1.f(x, n)
    print("f({0}, {1}) = {2}".format(x, n, result_f), end="")
    if (n == last_n):
        print(" <--- last_n", end="")
    print()
print()

for n in range(N - 2, N + 3):
    result_f = N1.f(x, n)
    print("f({0}, {1}) = {2}".format(x, n, result_f), end="")
    if (n == N):
        print(" <--- N", end="")
    print()
print()

start = time.time()
result_f = N1.f(x, N)
end = time.time()
time_f = end - start
print("f({0}, {1}) = {2}".format(x, N, result_f))

start = time.time()
result_g = g(x, N)
end = time.time()
time_g = end - start
print("g({0}, {1}) = {2}\n".format(x, N, result_g))

print("time_g / time_f = " + str(time_g / time_f) + "\n")
