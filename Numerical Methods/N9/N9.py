import numpy as np

epsilon = 1e-8

A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, -1]], dtype="float64")

def power_method(found_eigenvectors = None):
    guess = np.random.rand(3)
    error = epsilon + 1
    previous_eigenvector = None
    eigenvector = guess
    eigenvalue = None
    z = None

    while (error > epsilon):
        previous_eigenvector = eigenvector
        z = A @ eigenvector

        if (found_eigenvectors != None):
            for e in found_eigenvectors:
                z -= np.dot(z, e) * e

        eigenvector = z / np.linalg.norm(z)
        error = np.linalg.norm(np.abs(previous_eigenvector) - np.abs(eigenvector))

    eigenvalue = np.linalg.norm(z)

    if (eigenvector[0] * previous_eigenvector[0] < 0):
        eigenvalue *= -1

    return (eigenvector, eigenvalue)

found_eigenvectors = []
e1, lambda1 = power_method(found_eigenvectors)

found_eigenvectors.append(e1)
e2, lambda2 = power_method(found_eigenvectors)

found_eigenvectors.append(e2)
e3, lambda3 = power_method(found_eigenvectors)

print("Power method:")
print(lambda1, lambda2, lambda3, sep="\n")
print()

def rayleigh_method(lambda_guess, e_guess):
    error = epsilon + 1
    eigenvalue = lambda_guess
    eigenvector = e_guess

    while (error > epsilon):
        previous_eigenvalue = eigenvalue
        z = np.linalg.solve(A - (eigenvalue * np.identity(eigenvector.size)), eigenvector)
        eigenvector = z / np.linalg.norm(z)
        numerator = np.dot(eigenvector, (A @ eigenvector))
        denominator = np.dot(eigenvector, eigenvector)
        eigenvalue = numerator / denominator
        error = eigenvalue - previous_eigenvalue

    return eigenvalue

lambda1 = rayleigh_method(8, np.array([4, 5, 7]))
lambda2 = rayleigh_method(-4, np.array([-3, -4, 8]))
lambda3 = rayleigh_method(0, np.array([8, -5, 0]))

print("Rayleigh method:")
print(lambda1, lambda2, lambda3, sep="\n")
print()

a = A
error = epsilon + 1

while (error > epsilon):
    a_previous = a
    q, r = np.linalg.qr(a)
    a = r @ q
    error = np.abs(a - a_previous).max()

lambda1, lambda2, lambda3 = tuple(np.diag(a))

print("QR method:")
print(lambda1, lambda2, lambda3, sep="\n")
print()
