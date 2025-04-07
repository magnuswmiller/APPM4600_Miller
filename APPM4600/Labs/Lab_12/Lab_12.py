# Code for Lab 12 - APPM 4600

# Importing Libraries
import numpy as np

# Interval converter
def interval_convert(x,a,b):
    t = 0.5*(x + 1)*(b - a) + a
    gauss = sum(w * f(t)) * 0.5*(b - a)
    return guass

# Adaptive Quadrature Routines - the following three can be passed as the method parameter to the main adaptive_quad() function
# Composite Trapezoidal
def eval_composite_trap(N,a,b,f):
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)
    y = f(x)
    I_hat = h * (0.5 * y[0] + np.sum(y[1:N-1]) + 0.5 * y[-1])
    return I_hat, x, None

# Composite Simpson's
def eval_composite_simpsons(N,a,b,f):
    if N % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of points.")
    
    x = np.linspace(a, b, N)
    h = (b - a) / (N - 1)
    y = f(x)

    I_hat = h/3 * (y[0] + 
                   4 * np.sum(y[1:N-1:2]) + 
                   2 * np.sum(y[2:N-2:2]) + 
                   y[-1])
    return I_hat, x, None

# Gaussian Quadrature
def eval_gauss_quad(N,a,b,f):
    degree = 3
    x, w = np.polynomial.legendre.leggaus(degree)
    x = interval_convert(x,a,b)
    I_hat = np.sum(f(x)*w)

    return I_hat, x, w

# Adaptive Quadrature method - uses the above routines for the sub-intervals
def adaptive_quad(a,b,f,tol,N,method):
    # 1/2^50 ~ 1e-15
    maxit = 50
    left_p = np.zeros((maxit,))
    right_p = np.zeros((maxit,))
    s = np.zeros((maxit,1))
    left_p[0] = a; right_p[0] = b;

    # initial approx and grid
    s[0],x,_ = method(N,a,b,f);

    # save grid
    X = []
    X.append(x)
    j = 1;
    I = 0;
    nsplit = 1;

    while j < maxit:
        # get midpoint to split interval into left and right
        c = 0.5*(left_p[j-1]+right_p[j-1]);

        # compute integral on left and right spilt intervals
        s1,x,_ = method(N,left_p[j-1],c,f); X.append(x)
        s2,x,_ = method(N,c,right_p[j-1],f); X.append(x)

        if np.max(np.abs(s1+s2-s[j-1])) > tol:
            left_p[j] = left_p[j-1]
            right_p[j] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j] = s1
            left_p[j-1] = 0.5*(left_p[j-1]+right_p[j-1])
            s[j-1] = s2
            j = j+1
            nsplit = nsplit+1
        else:
            I = I+s1+s2
            j = j-1
            if j == 0:
                j = maxit
    return I, np.unique(X), nsplit

# driver function
def main():
    return -1

if __name__ == '__main__':
    main()
