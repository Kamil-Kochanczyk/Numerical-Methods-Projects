from math import cos, exp



def f(x, N):
    result = 1
    
    c = cos(x * x * x * x)
    e_inverse = 1 / exp(1)
    
    alfa_n_1 = 1
    alfa_n_2 = c
    beta = e_inverse
    
    for n in range(1, N + 1):
        alfa_n = 2 * alfa_n_1 * c - alfa_n_2
        
        result += beta * (1 + alfa_n * (beta * beta * beta - alfa_n))
        
        alfa_n_2 = alfa_n_1
        alfa_n_1 = alfa_n
        beta *= e_inverse
    
    return result



def find(x):
    last_n = 0
    values = []
    
    while (f(x, last_n + 1) != (current_value := f(x, last_n))):
        values.append(current_value)
        last_n += 1
    
    values.append(f(x, last_n))
    
    N = 0
    
    while (abs(values[N] - values[last_n]) >= 1e-10):
        N += 1
    
    return (last_n, N)



if __name__ == "__main__":
    x = 1
    last_n, N = find(x)
    print(last_n)
    print(N)
    print(f(x, last_n))
    print(f(x, N))
