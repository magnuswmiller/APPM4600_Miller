# Code for Homework 08 - APPM 4600

# Importing libraries
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb

# Define function plot points
def function_eval(fun,a,b,n):
    x_eval = np.linspace(a,b,n)
    y_eval_fine = fun(x_eval)
    print(y_eval_fine)
    return x_eval, y_eval_fine

# Define Equispaced Nodes
def equispace_node(fun,a,b,n):
    x_pts = np.linspace(a,b,n)
    y_eval = fun(x_pts)
    return x_pts, y_eval

# Define Chebyshev Nodes
def cheb_node(a,b,n):
    j = np.arange(1, n+1)
    x_cheb = np.cos((2*j - 1) * np.pi / (2*n))
    x_cheb = 0.5 * (b - a) * x_cheb + 0.5 * (a + b)
    
    y_eval = fun1(x_cheb)
    return x_cheb, y_eval

# Lagrange Interpolation Helper
def lagrange_interp(fun, x_node, x_target):
    n = len(x_node)
    m = len(x_target)
    
    lagrange = np.ones((n,m))
    weights = np.ones(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                weights[i] *= (x_node[i]-x_node[j])
    weights = 1/weights

    psi = np.ones(m)
    for i in range(n):
        psi *= (x_target-x_node[i])

    fj = 1 / (np.transpose(np.tile(x_target, (n, 1))) - np.tile(x_node, (m, 1)))
    L = fj * np.transpose(np.tile(psi, (n, 1))) * np.tile(weights, (m, 1))

    g = L @ fun(x_node)

    return g
    return -1

# Hermite Interpolation Helper
def hermite_interp(fun, dfun, x_node, x_target):
    n = len(x_node)
    m = len(x_target)

    x_herm = np.repeat(x_node, 2)
    y_herm = np.repeat(fun(x_node),2)

    table = np.zeros((2*n, 2*n))
    table[:, 0] = y_herm

    for i in range(n):
        table[2*i+1, 1] = dfun(x_node[i])
        if i > 0:
            table[2*i, 1] = (table[2*i, 0] - table[2*i-1, 0]) / (x_herm[2*i] - 
                                                                 x_herm[2*i-1])

    for j in range(2, 2 * n):
        for i in range(j, 2 * n):
            table[i, j] = (table[i, j-1] - table[i-1, j-1]) / (x_herm[i] - 
                                                               x_herm[i-j])

    g = np.zeros(m)
    for k in range(m):
        term = 1
        g[k] = table[0, 0]  # First term of the polynomial
        for j in range(1, 2 * n):
            term *= (x_target[k] - x_herm[j-1])
            g[k] += table[j, j] * term  # Add next term

    return g

# Tridiagonal System Solver
def solve_tridiag_system(A, d):
    n = len(d)
    c_prime = np.zeros(n-1)
    d_prime = np.zeros(n)

    c_prime[0] = A[0,1] / A[0,0]
    d_prime[0] = d[0] / A[0,0]

    for i in range(1, n-1):
        denom = A[i, i] - A[i, i-1] * c_prime[i-1]
        c_prime[i] = A[i, i+1] / denom
        d_prime[i] = (d[i] - A[i, i-1] * d_prime[i-1]) / denom

    d_prime[n-1] = (d[n-1] - A[n-1, n-2] * d_prime[n-2]) / (A[n-1, n-1] - 
                                                            A[n-1, n-2] * 
                                                            c_prime[n-2])

    x = np.zeros(n)
    x[-1] = d_prime[-1]

    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x

# Natural Cubic Spline Intepolation Helper
def nat_cubic_spline_interp(fun, x_node, x_target):
    n = len(x_node) - 1  
    h = np.diff(x_node)  
    y_node = fun(x_node)  

    d = np.zeros(n-1)
    for i in range(1, n):
        d[i-1] = (3/h[i]) * (y_node[i+1] - y_node[i]) - (3/h[i-1]) * (y_node[i] - 
                                                                      y_node[i-1])

    A = np.zeros((n-1, n-1))
    for i in range(n-1):
        if i > 0:
            A[i, i-1] = h[i]  
        A[i, i] = 2 * (h[i] + h[i+1])  
        if i < n-2:
            A[i, i+1] = h[i+1]  

    c = np.zeros(n+1)
    c[1:n] = solve_tridiag_system(A, d)

    a = y_node[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y_node[i+1] - y_node[i]) / h[i] - (h[i] / 3) * (2 * c[i] + c[i+1])
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    g = np.zeros(len(x_target))
    for j in range(len(x_target)):
        i = np.searchsorted(x_node[:-1], x_target[j]) - 1
        i = max(0, min(i, n-1))

        dx = x_target[j] - x_node[i]
        g[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return g

# Clamped Cubic Spline Interpolation Helper
def clamp_cubic_spline_interp(fun, dfun, x_node, x_target):
    n = len(x_node) - 1
    h = np.diff(x_node)
    y_node = fun(x_node)

    d = np.zeros(n+1)
    d[0] = (3/h[0]) * (y_node[1] - y_node[0]) - 3 * dfun(x_node[0])
    d[n] = 3 * dfun(x_node[n]) - (3/h[n-1]) * (y_node[n] - y_node[n-1])

    for i in range(1, n):
        d[i] = (3/h[i]) * (y_node[i+1] - y_node[i]) - (3/h[i-1]) * (y_node[i] - 
                                                                    y_node[i-1])

    A = np.zeros((n+1, n+1))
    A[0, 0] = 2 * h[0]
    A[0, 1] = h[0]

    A[n, n] = 2 * h[n-1]
    A[n, n-1] = h[n-1]

    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    c = solve_tridiag_system(A, d)

    a = y_node[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y_node[i+1] - y_node[i]) / h[i] - (h[i] / 3) * (2 * c[i] + c[i+1])
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    g = np.zeros(len(x_target))
    for j in range(len(x_target)):
        i = np.searchsorted(x_node[:-1], x_target[j]) - 1
        i = max(0, min(i, n-1))

        dx = x_target[j] - x_node[i]
        g[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return g

def periodic_cubic_spline_interp(fun, x_node, x_target):
    n = len(x_node) - 1
    h = np.diff(x_node)
    y_node = fun(x_node)

    d = np.zeros(n+1)
    for i in range(1, n):
        d[i] = (3/h[i]) * (y_node[i+1] - y_node[i]) - (3/h[i-1]) * (y_node[i] - y_node[i-1])

    # Modify for periodicity: Ensure S''(0) = S''(2π)
    d[0] = (3/h[0]) * (y_node[1] - y_node[0]) - (3/h[n-1]) * (y_node[n] - y_node[n-1])
    d[n] = d[0]  # Ensure periodicity

    A = np.zeros((n+1, n+1))
    for i in range(1, n):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]

    # Modify for periodicity
    A[0, 0] = 2 * (h[0] + h[n-1])
    A[0, 1] = h[0]
    A[0, n-1] = h[n-1]
    A[n, :] = A[0, :]

    c = solve_tridiag_system(A, d)

    a = y_node[:-1]
    b = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        b[i] = (y_node[i+1] - y_node[i]) / h[i] - (h[i] / 3) * (2 * c[i] + c[i+1])
        d[i] = (c[i+1] - c[i]) / (3 * h[i])

    g = np.zeros(len(x_target))
    for j in range(len(x_target)):
        i = np.searchsorted(x_node[:-1], x_target[j]) - 1
        i = max(0, min(i, n-1))

        dx = x_target[j] - x_node[i]
        g[j] = a[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3

    return g


# Defining Functions
def fun1(x):
    return 1/(1+x**2)

def dfun1(x):
    return -(2 * x) / (x**2 + 1)**2

def fun2(x):
    return np.sin(10*x)

def dfun2(x):
    return 10*np.cos(10*x)

# Driver function for question 1
def driver1():
    # Define Interval
    a = -5
    b = 5

    # Interp Nodes:
    interp_deg_low = [5,10,15,20]

    # Lagrange Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(fun1,a,b,N)
        poly_eval = lagrange_interp(fun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Lagrange Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


    # Hermite Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(fun1,a,b,N)
        poly_eval = hermite_interp(fun1, dfun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Hermite Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Natural Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(fun1,a,b,N)
        poly_eval = nat_cubic_spline_interp(fun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Natural Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Clamped Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(fun1,a,b,N)
        poly_eval = clamp_cubic_spline_interp(fun1, dfun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Clamped Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return -1

def driver2():
    # Define Interval
    a = -5
    b = 5

    # Interp Nodes:
    interp_deg_low = [5,10,15,20]

    # Lagrange Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = cheb_node(a,b,N)
        poly_eval = lagrange_interp(fun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Lagrange Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


    # Hermite Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = cheb_node(a,b,N)
        poly_eval = hermite_interp(fun1, dfun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Hermite Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Natural Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = cheb_node(a,b,N)
        poly_eval = nat_cubic_spline_interp(fun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Natural Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Clamped Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun1,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = cheb_node(a,b,N)
        poly_eval = clamp_cubic_spline_interp(fun1, dfun1, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1,2)
    plt.xlim(-5.5,5.5)
    plt.title("Clamped Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    return -1

# Driver function for question 3
def driver3():
    '''
    # Define Interval
    a = 0 
    b = 2*np.pi 

    # Interp Nodes:
    interp_deg_low = [5, 10, 15, 20]

    # Periodic Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun2,a,b,1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(a,b,N)
        poly_eval = periodic_cubic_spline_interp(fun2, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1.5,1.5)
    plt.xlim(-0.6283185307,2.2*np.pi)
    plt.title("Natural Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    '''
# Define Interval
    a = 0 
    b = 2*np.pi 

    # Interp Nodes:
    interp_deg_low = [5, 10, 15, 20]

    # Periodic Cubic Spline Interpolation
    x_eval, y_eval_fine = function_eval(fun2, a, b, 1001)
    for N in interp_deg_low:
        x_pts, y_eval = equispace_node(fun2,a, b, N)
        poly_eval = periodic_cubic_spline_interp(fun2, x_pts, x_eval)

        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")
        plt.scatter(x_pts, y_eval, marker='o', s=20)
    
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    plt.ylim(-1.5, 1.5)
    plt.xlim(-0.5, 2.2*np.pi)
    plt.title("Periodic Cubic Spline Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    return -1

# Run driver functions
driver1()
driver2()
driver3()
