# Code for Homework_08 - APPM 4600
# Applied Mathematics Department
# College of Engineering and Applied Science
# University of Colorado at Boulder

# Importing Libraries
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import matplotlib
from scipy.special import comb

# Global Constants/Set Defaults
plt.rcParams['figure.figsize'] = [10, 5];

# Function definitions
def fun(x):
    return ((1)/(1+(10*x)**2))

# Helper functions
def chebyshev_nodes(N, a=-1, b=1):
    j = np.arange(1, N + 1)
    x_cheb = np.cos((2 * j - 1) * np.pi / (2 * N))
    
    # Scale to [a, b] if needed
    return 0.5 * (a + b) + 0.5 * (b - a) * x_cheb

# --------- Vandermonde Constructor --------
def construct_vander(x_pts):
    num_pts = len(x_pts)
    vander = np.vander(x_pts, N=num_pts, increasing=True)
    return vander 

# ------ Monomial Expansion/Polynomia Interpolation ------
def mono_interp(x_pts, y_eval, x_eval):
    vander = construct_vander(x_pts)
    coefs = np.linalg.solve(vander, y_eval)
    poly_eval = np.polyval(coefs[::-1], x_eval)
    return poly_eval

# -------- Barycentric Weights Constructor --------
def construct_bary_weight(x_pts):
    num_pts = len(x_pts)
    weights = np.ones(num_pts)
    for j in range(num_pts):
        weights[j] = 1.0 / np.prod([x_pts[j] - x_pts[i] for i in range(num_pts) if i != j])
    return weights

# ------ Barycentric Lagrange Interpolation ------
def bary_interp(x_pts, y_eval, weights, x_eval):
    poly_eval = np.zeros_like(x_eval)
    for i, x_i in enumerate(x_eval):
        if np.any(np.isclose(x_i, x_pts, atol=1e-12)):  # Use np.isclose() for floating-point safety
            poly_eval[i] = y_eval[np.argmin(np.abs(x_pts - x_i))] 
        else:
            num = np.sum((weights/(x_i-x_pts))*y_eval)
            denom = np.sum(weights/(x_i-x_pts)) 
            poly_eval[i] = num/denom
    return poly_eval 

# Driver Function for Barycentric Lagrange Interpolation
def driver_bary():
    # evaluation points for f(x) plot
    x_eval = np.linspace(-1,1,1001)
    # evaluations of f(x)
    y_eval_fine = fun(x_eval)


    # Interpolation degrees
    interp_deg_low = [2,3,4,5,6,7,8,9]
    interp_deg_high = [12,14,16,18,20]
    interp_deg_super = [70, 90]

    # Interpolations (low)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label='f(x)')
    for N in interp_deg_low:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # find bary weights
        weights = construct_bary_weight(x_pts)

        # calculate polynomial
        poly_eval = bary_interp(x_pts, y_eval, weights, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.ylim(-1, 2)
    plt.title("Barycentric Langrange Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Interpolations (high)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label="f(x)")
    for N in interp_deg_high:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # find bary weights
        weights = construct_bary_weight(x_pts)

        # calculate polynomial
        poly_eval = bary_interp(x_pts, y_eval, weights, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.ylim(-1, 2)
    plt.title("Barycentric Lagrange Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    
    # Interpolations (super)
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label="f(x)")
    for N in interp_deg_super:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # find bary weights
        weights = construct_bary_weight(x_pts)

        # calculate polynomial
        poly_eval = bary_interp(x_pts, y_eval, weights, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.ylim(-1, 2)
    plt.title("Barycentric Lagrange Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    return -1

# Driver Function for Monomial Interpolation
def driver_mono():
    # evaluation points for f(x) plot
    x_eval = np.linspace(-1,1,1001)
    # evaluations of f(x)
    y_eval_fine = fun(x_eval)

    # Interpolation degrees
    interp_deg_low = [2,3,4,5,6,7,8,9]
    interp_deg_high = [12,14,16,18,20]
    interp_deg_super = [70,90]

    # Interpolations (low)
    for N in interp_deg_low:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # Compute polynomial
        poly_eval = mono_interp(x_pts, y_eval, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label="f(x)")
    plt.ylim(-1, 2)
    plt.title("Polynomial Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Interpolations (high)
    for N in interp_deg_high:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # Compute polynomial
        poly_eval = mono_interp(x_pts, y_eval, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label="f(x)")
    plt.ylim(-1, 2)
    plt.title("Polynomial Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    # Interpolations (super)
    for N in interp_deg_super:
        # Equispaced
        # x_pts = np.linspace(-1,1,N)
        # Chebyshev
        x_pts = chebyshev_nodes(N)
        y_eval = fun(x_pts)

        # Compute polynomial
        poly_eval = mono_interp(x_pts, y_eval, x_eval)

        # Plot the interpolating polynomial
        plt.plot(x_eval, poly_eval, linestyle='--', label=f"(N={N})")

        # Scatter plot the interpolation points
        plt.scatter(x_pts, y_eval, marker='o', s=30)

    # Format the plot
    plt.plot(x_eval, y_eval_fine, linestyle='solid', label="f(x)")
    plt.ylim(-1, 2)
    plt.title("Polynomial Interpolation with Different N Values")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
    return -1

driver_mono()
driver_bary()
