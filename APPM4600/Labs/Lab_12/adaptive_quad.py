# get lgwts routine and numpy
import numpy as np

# adaptive quad subroutines
# the following three can be passed
# as the method parameter to the main adaptive_quad() function

def eval_composite_trap(M,a,b,f):
    """
    Composite Trapezoidal Rule on interval [a,b] with M subintervals
    Returns:
        I_trap - approximated integral
        xnode  - quadrature nodes used
        None   - placeholder for weights (not used)
    """
    h = (b - a) / M
    xnode = a + np.arange(M + 1) * h

    I_trap = 0.5 * f(xnode[0]) + 0.5 * f(xnode[-1])
    I_trap += np.sum(f(xnode[1:-1]))
    I_trap *= h

    return I_trap, xnode, None

def eval_composite_simpsons(M,a,b,f):
    """
    Composite Simpson's Rule on interval [a,b] with M subintervals (M must be even)
    Returns:
        I_simp - approximated integral
        xnode  - quadrature nodes used
        None   - placeholder for weights (not used)
    """
    h = (b - a) / M
    xnode = a + np.arange(M + 1) * h

    I_simp = f(xnode[0]) + f(xnode[-1])
    I_simp += 4 * np.sum(f(xnode[1:-1:2]))  # odd-indexed nodes
    I_simp += 2 * np.sum(f(xnode[2:-1:2]))  # even-indexed nodes
    I_simp *= h / 3

    return I_simp, xnode, None

def eval_gauss_quad(M,a,b,f):
    """
    Non-adaptive numerical integrator for \int_a^b f(x)w(x)dx
    Input:
    M - number of quadrature nodes
    a,b - interval [a,b]
    f - function to integrate

    Output:
    I_hat - approx integral
    x - quadrature nodes
    w - quadrature weights

    Currently uses Gauss-Legendre rule
    """
    x, w = np.polynomial.legendre.leggauss(M)

    # Map from [-1,1] to [a,b]
    x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
    w_mapped = 0.5 * (b - a) * w

    I_hat = np.sum(f(x_mapped) * w_mapped)
    return I_hat, x_mapped, w_mapped

def adaptive_quad(a,b,f,tol,M,method):
  """
  Adaptive numerical integrator for \int_a^b f(x)dx
  
  Input:
  a,b - interval [a,b]
  f - function to integrate
  tol - absolute accuracy goal
  M - number of quadrature nodes per bisected interval
  method - function handle for integrating on subinterval
         - eg) eval_gauss_quad, eval_composite_simpsons etc.
  
  Output: I - the approximate integral
          X - final adapted grid nodes
          nsplit - number of interval splits
  """
  # 1/2^50 ~ 1e-15
  maxit = 50
  left_p = np.zeros((maxit,))
  right_p = np.zeros((maxit,))
  s = np.zeros((maxit,1))
  left_p[0] = a; right_p[0] = b;
  # initial approx and grid
  s[0],x,_ = method(M,a,b,f);
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
    s1,x,_ = method(M,left_p[j-1],c,f); X.append(x)
    s2,x,_ = method(M,c,right_p[j-1],f); X.append(x)
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
  return I,np.unique(X),nsplit

