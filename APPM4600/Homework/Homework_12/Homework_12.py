# Code for Homework 12 - APPM 4600

# importing libraries
import numpy as np
from scipy.linalg import hilbert
from scipy.linalg import solve

# function for output clarification
def print_divider(title=None):
    print("\n" + "-" * 50)
    if title:
        print(f"{title}")
        print("-" * 80)

# helper function for 1a
def question1a():
    print_divider("Question 1a")
    # Define the coefficient matrix A
    A = np.array([[6, 2, 2],
                  [2, 2/3, 1/3],
                  [1, 2, -1]],
                 dtype=float)

    # Right-hand side vector b
    b = np.array([-2, 1, 0], dtype=float)

    # Proposed solution vector
    x = np.array([2.6, -3.8, -5.0], dtype=float)

    # Compute A @ x
    Ax = A @ x

    # Print A @ x and b for comparison
    print("A @ x =", Ax)
    print("b =", b)

    # Check if A @ x is close to b
    if np.allclose(Ax, b, atol=1e-10):
        print("The solution (2.6, -3.8, -5) satisfies the system.")
    else:
        print("The solution does NOT satisfy the system.")
    return -1

# helper functions for question 1b/1c
# Custom rounding function to 4 significant digits
def round4(x):
    return float(f"{x:.5g}")

# Apply rounding to all elements of a matrix or vector
def round4_array(arr):
    return np.vectorize(round4)(arr)

# subroutine for question 1b
def question1b():
    print_divider("Question 1b")
    # Original matrix A and vector b
    A = np.array([[6.0, 2.0, 2.0],
                  [2.0, 2/3, 1/3],
                  [1.0, 2.0, -1.0]])
    b = np.array([-2.0, 1.0, 0.0])

    # Round initial A and b
    A = round4_array(A)
    b = round4_array(b)

    # Number of equations
    n = 3

    # Forward elimination (no pivoting)
    for i in range(n - 1):
        for j in range(i + 1, n):
            m = round4(A[j, i] / A[i, i])
            A[j, i:] = round4_array(A[j, i:] - m * A[i, i:])
            b[j] = round4(b[j] - m * b[i])

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = round4(sum(A[i, j] * x[j] for j in range(i + 1, n)))
        x[i] = round4((b[i] - s) / A[i, i])

    # Print results
    print("Upper triangular matrix A:\n", A)
    print("Modified right-hand side b:", b)
    print("Solution [x, y, z]:", x)
    return -1

# subroutine for question 1c
def question1c():
    print_divider("Question 1c")
    # Original matrix A and vector b
    A = np.array([[6.0, 2.0, 2.0],
                  [2.0, 2/3, 1/3],
                  [1.0, 2.0, -1.0]])
    b = np.array([-2.0, 1.0, 0.0])

    # Round initial A and b
    A = round4_array(A)
    b = round4_array(b)

    # Number of equations
    n = 3

    # Forward elimination WITH partial pivoting
    for i in range(n - 1):
        # Find row with largest absolute value in column i
        max_row = i + np.argmax(np.abs(A[i:, i]))
        
        # Swap rows in A and b
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Elimination
        for j in range(i + 1, n):
            m = round4(A[j, i] / A[i, i])
            A[j, i:] = round4_array(A[j, i:] - m * A[i, i:])
            b[j] = round4(b[j] - m * b[i])
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = round4(sum(A[i, j] * x[j] for j in range(i + 1, n)))
        x[i] = round4((b[i] - s) / A[i, i])

    # Print results
    print("Upper triangular matrix A:\n", A)
    print("Modified right-hand side b:", b)
    print("Solution [x, y, z]:", x)
    return -1

# helper function for question 2
def question2():
    print_divider("Question 2")
    # Original symmetric matrix A
    A = np.array([[12, 10, 4],
                  [10, 8, -5],
                  [4, -5, 3]], dtype=float)

    # Extract the vector x (below first diagonal element)
    x = A[1:, 0]  # vector [10, 4]

    # Compute the norm and Householder vector
    norm_x = np.linalg.norm(x)
    sign = np.sign(x[0]) if x[0] != 0 else 1
    e1 = np.array([1.0, 0.0])
    v = x + sign * norm_x * e1
    v = v / np.linalg.norm(v)

    # Build Householder matrix H (3x3) using v = [0, v1, v2]
    v_full = np.zeros(3)
    v_full[1:] = v
    H = np.eye(3) - 2 * np.outer(v_full, v_full)

    # Apply similarity transformation
    A1 = H @ A @ H  # This is the tridiagonalized form

    # zero out very small values for clean display
    #A1[np.abs(A1) < 1e-10] = 0

    # Print results
    print("Householder matrix H:\n", H)
    print("\nTransformed matrix A1 = H A H:\n", A1)
    return -1

# hilbert matrix constructor
def true_hilbert(n):
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])

# helper function for the power method for question 3
def power_method(A, max_iter=1000, tol=1e-16):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    for i in range(max_iter):
        x_next = A @ x
        x_next = x_next / np.linalg.norm(x_next)
        
        # Rayleigh quotient for dominant eigenvalue estimate
        lambda_approx = x_next.T @ A @ x_next

        if np.linalg.norm(x_next - x) < tol:
            return lambda_approx, x_next, i+1
        
        x = x_next
    
    return lambda_approx, x, max_iter

# inverse power method for smallest
def inverse_power_method(A, tol=1e-10, max_iter=5000):
    n = A.shape[0]
    x = np.random.rand(n)
    x = x / np.linalg.norm(x)
    
    lambda_old = 0

    for k in range(max_iter):
        y = solve(A, x)  # Solve A y = x
        x_next = y / np.linalg.norm(y)
        
        lambda_new = x_next.T @ A @ x_next
        
        if abs(lambda_new - lambda_old) < tol:
            return lambda_new, x_next, k+1
        
        lambda_old = lambda_new
        x = x_next

    return lambda_new, x, max_iter

# subroutine for question 3
def question3():
    print_divider("Question 3a")
    for n in range(4, 21, 4):
        A = true_hilbert(n)
        lam, v, num_iter = power_method(A)
        print(f"n = {n}: Dominant eigenvalue ≈ {lam:.6f}, iterations = {num_iter}")
    print_divider("Question 3b")
    A = true_hilbert(16)
    lam, v, num_iter = inverse_power_method(A)
    print(f"n = 16: Smallest eigenvalue ≈ {lam:.6e}, iterations = {num_iter}")
    return -1

# main driver function
def main():
    question1a()
    question1b()
    question1c()
    question2()
    question3()
    return -1

if __name__ == '__main__':
    main()
