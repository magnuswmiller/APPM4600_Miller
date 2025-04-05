# Code for Homework 10 - APPM 4600

# importing libraries
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt

# function that calculates the coefficients of maclaurin expansion of sin(x)
def mac_exp_sin(deg):
    coeffs = np.zeros(deg+1)
    for i in range(deg+1):
        if i % 2 == 1:
            coeffs[i] = ((-1)**((i-1)//2))/factorial(i)
    return coeffs

# Function that computes R[m/n] Pade approximation
def pade_approx(coeffs, m, n):
    assert(len(coeffs) >= m + n + 1)

    # build system of eqns
    A = np.zeros((n,n))
    rhs = np.zeros(n)
    for i in range(n):
        rhs[i] = -coeffs[m+i+1]
        for j in range(n):
            A[i,j] = coeffs[m+i-j]

    # solve for b_n in the denominator
    b = np.linalg.solve(A, rhs)
    b_full = np.concatenate(([1.0],b))

    # solve for a_n in the numerator
    a = np.zeros(m + 1)
    for i in range(m + 1):
        for j in range(min(i + 1, n + 1)):
            if i - j >= 0:
                a[i] += b_full[j] * coeffs[i - j]

    return a, b_full

# Function that evaluates the pade approximation at the value x provided
def pade_eval(x, a, b):
    num = np.polyval(a[::-1], x)
    denom = np.polyval(b[::-1], x)
    return num/denom

# function that evaluates the maclaurin polynomial at the value x provided
def mac_eval(x, coeffs):
    poly = np.polyval(coeffs[::-1], x)
    return poly

def error_eval(y_eval_sin, y_eval):
    error = np.abs(y_eval_sin - y_eval)
    return error

# function that handles plotting
def overlay_plotter(x_eval, y_eval_sin, y_eval_pade, y_eval_mac, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x_eval, y_eval_sin, label='sin(x)', linewidth=2)
    plt.plot(x_eval, y_eval_pade, '--', label='Padé', linewidth=2)
    plt.plot(x_eval, y_eval_mac, '--', label='Maclaurin deg. 6', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return -1

# function that handles error plotting
def error_plotter(x_eval, y_pade_err, y_mac_err, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x_eval, y_pade_err, label='Padé', linewidth=2)
    plt.plot(x_eval, y_mac_err, '--', label='Maclaurin deg. 6', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return -1

# function that plots all errors
def error_plotter_all(x_eval, y_mac_err, y_pade_3, y_pade_2, y_pade_4, title):
    plt.figure(figsize=(8, 4))
    plt.plot(x_eval, y_pade_3, '-.', label='Padé [3/3]', linewidth=2)
    plt.plot(x_eval, y_pade_2, ':',label='Padé [2/4]', linewidth=2)
    plt.plot(x_eval, y_pade_4, '--',label='Padé [4/2]', linewidth=2)
    plt.plot(x_eval, y_mac_err, label='Maclaurin deg. 6', linewidth=2)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    return -1

# function that handles each pade
def handler(deg, m, n):
    print(" * * * * * * * * * [" + str(m) + "/" + str(n)+ "] * * * * * * * * * ")
    # define integral:
    init = 0
    final = 5
    # determine maclaurin coefficients
    coeffs = mac_exp_sin(deg)
    print(" * Coefficients of Maclaurin Polynomial:")
    print(coeffs)
    print("----------------------------------------------")

    # compute pade approximation [3/3]
    a1, b1 = pade_approx(coeffs, m, n)
    print(" * Coefficients of the numerator: ")
    print(a1)
    print("----------------------------------------------")
    print(" * Coefficients of the denominator: ")
    print(b1)
    print("----------------------------------------------")

    # Evaluate [3/3]
    x_eval = np.linspace(init, final, 400)
    y_eval_sin = np.sin(x_eval)
    y_eval_pade = pade_eval(x_eval, a1, b1)
    y_eval_mac = mac_eval(x_eval, coeffs)

    # plot pade vs sin
    overlay_plotter(x_eval, y_eval_sin, y_eval_pade, y_eval_mac, 'Plots of Pade Approx and Maclaurin Poly. of Sin(x)')

    # evaluate error
    y_pade_err = error_eval(y_eval_sin, y_eval_pade)
    y_mac_err = error_eval(y_eval_sin, y_eval_mac)

    # plot error
    error_plotter(x_eval, y_pade_err, y_mac_err, 'Error Plots of Pade Approx and Maclaurin Poly. of Sin(x)')
    return x_eval, y_mac_err, y_pade_err

# main driver function
def main():
    DEGREE = 6

    # Part (a)
    x_eval, y_mac_err, y_pade_3 = handler(DEGREE, 3, 3)
    
    # Part (b)
    e_eval, y_mac_err, y_pade_2 = handler(DEGREE, 2, 4)

    # Part (c)
    x_eval, y_mac_err, y_pade_4 = handler(DEGREE, 4, 2)

    # Plot Error
    error_plotter_all(x_eval, y_mac_err, y_pade_3, y_pade_2, y_pade_4, 'Error Plots of Pade Approx and Maclaurin Poly. of Sin(x)')
    return -1

if __name__ == "__main__":
    main()
