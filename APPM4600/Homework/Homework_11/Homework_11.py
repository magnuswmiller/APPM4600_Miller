# Code for Homework 11 - APPM 4600

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sympy as sp
import math

# Eval composite trapezoidal rule
def eval_comp_trap(f,a,b,n):
    h = (b-a)/n
    result = 0.5 * (f(a) + f(b))

    for i in range(1, n):
        x = a + (i * h)
        result += f(x)

    return result * h

# Eval composite simpson's rule
def eval_comp_simp(f,a,b,n):
    if n % 2 != 0:
        raise ValueError("n must be even to use Simpson's")

    h = (b-a)/n
    result = f(a) + f(b)

    for i in range(1,n):
        x = a + (i * h)
        if i % 2 == 0:
            result += 2 * f(x)
        else:
            result += 4 * f(x)

    return result * (h/3)

# Defining function
def fun(x):
    return 1/(1+x**2)

def fun2(x):
    return np.cos(1/x)*x

# Calculating error stuff
def err_est_trap():
    s = sp.Symbol('s')
    f = 1 / (1 + s**2)
    f2 = sp.diff(f,s,2)
    max_f2 = sp.Max(*[sp.Abs(f2.subs(s, val)) for val in range(-5, 6)])
    return f2, max_f2

def err_est_simp():
    s = sp.Symbol('s')
    f = 1 / (1 + s**2)
    f4 = sp.diff(f,s,4)
    max_f4 = sp.Max(*[sp.Abs(f4.subs(s, val)) for val in range(-5, 6)])
    return f4, max_f4

def abs_error(approx):
    return abs(2 * math.atan(5) - approx)

# Driver function
def main():
    # Question 1 code
    result_trap = eval_comp_trap(fun,-5,5,1291)
    trap_err = abs_error(result_trap)
    print(result_trap)
    print(trap_err)

    result_simp = eval_comp_simp(fun,-5,5,108)
    simp_err = abs_error(result_simp)
    print(result_simp)
    print(simp_err)

    result_quad_6 = scipy.integrate.quad(fun,-5,5)[0]
    quad_6_err = abs_error(result_quad_6)
    print(result_quad_6)
    print(quad_6_err)

    result_quad_4 = scipy.integrate.quad(fun,-5,5,epsabs=1e-4)[0]
    quad_4_err = abs_error(result_quad_4)
    print(result_quad_4)
    print(quad_4_err)

    f2, max_f2 = err_est_trap()
    print(f2)
    print(max_f2)
    f4, max_f4 = err_est_simp()
    print(f4)
    print(max_f4)

    # Question 2 Code
    epsilon = 1e-10
    b = 1
    result_2 = eval_comp_trap(fun2,epsilon,b,4)
    print(result_2)

    return -1

if __name__ == '__main__':
    main()
