# Code for APPM 4600 Homework 05
# Completed by Magnus Miller

# Importing Libraries:
import numpy as np
import math

# Defining functions
def fxn1(x):
    return np.asarray([3*x[0]**2-x[1]**2,3*x[0]*x[1]**2-x[0]**3-1])

def dfxn1(x):
    return np.array([[6*x[0],-3*x[1]**2],[-3*x[0]**2+3*x[1]**2,6*x[0]*x[1]]]);

def fun(x):
    return np.asarray([3*x[0]**2-x[1]**2,3*x[0]*x[1]**2-x[0]**3-1])

def Jfun(x):
    M = np.array([[6*x[0],-3*x[1]**2],[-3*x[0]**2+3*x[1]**2,6*x[0]*x[1]]]);
    return M;

s = np.array([[(1/6),(1/18)],[0,(1/6)]]);

def gfun(x):
    v = x-s@fun(x);
    return v;
def Jgfun(x):
    M = s@Jfun(x);
    return M;

def fixed_point_method_nd(G,JG,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Gn = G(xn); #function value vector
    n=0;
    nf=1;  #function evals

    if verb:
        print("|--n--|----xn----|---|G(xn)-xn|---|");

    while np.linalg.norm(Gn-xn)>tol and n<=nmax:

        if verb:
            rhoGn = np.max(np.abs(np.linalg.eigvals(JG(xn))));
            print("|--%d--|%1.7f|%1.15f|%1.2f|" %(n,np.linalg.norm(xn),np.linalg.norm(Gn-xn),rhoGn));

        # Fixed Point iteration step
        xn = Gn;

        n+=1;
        rn = np.vstack((rn,xn));
        Gn = G(xn);
        nf+=1;

        if np.linalg.norm(xn)>1e15:
            n=nmax+1;
            nf=nmax+1;
            break;

    r=xn;

    if verb:
        if n>=nmax:
            print("Fixed point iteration failed to converge, n=%d, |G(xn)-xn|=%1.1e\n" % (nmax,np.linalg.norm(Gn-r)));
        else:
            print("Fixed point iteration converged, n=%d, |G(xn)-xn|=%1.1e\n" % (n,np.linalg.norm(Gn-r)));

    return (r,rn,n);

def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if (len(x0)<100):
        if (np.linalg.cond(Jf(x0)) > 1e16):
            print("Error: matrix too close to singular");
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
            r=x0;
            return (r,rn,nf,nJ);

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.15f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

def find_point_on_surface(x0, y0, z0, tol=1e-10, max_iter=50):
    approx = []
    def f(x, y, z):
        return x**2 + 4*y**2 + 4*z**2 - 16

    def grad_f(x, y, z):
        return np.array([2*x, 8*y, 8*z])
    
    x, y, z = x0, y0, z0

    print(f"Iteration {0}: x = {x:.10f}, y = {y:.10f}, z = {z:.10f}, f = {f(x,y,z):.10e}")
    for i in range(max_iter):
        f_val = f(x, y, z)
        approx.append(float(f_val))
        grad = grad_f(x, y, z)
        norm_grad = np.dot(grad, grad)

        if abs(f_val) < tol:
            print(f"Iteration {i+1}: x = {x:.10f}, y = {y:.10f}, z = {z:.10f}, f = {f_val:.10e}")
            break  # Converged
        
        d = f_val / norm_grad
        x -= d * grad[0]
        y -= d * grad[1]
        z -= d * grad[2]

        print(f"Iteration {i+1}: x = {x:.10f}, y = {y:.10f}, z = {z:.10f}, f = {f_val:.10e}")
    return approx

def compute_order_of_convergence(approximations):
    orders = []
    print(approximations)
    for k in range(2, len(approximations) - 1):
        numerator = np.log(abs((approximations[k+1] - approximations[k]) / (approximations[k] - approximations[k-1])))
        denominator = np.log(abs((approximations[k] - approximations[k-1]) / (approximations[k-1] - approximations[k-2])))
        
        if denominator != 0:
            orders.append(numerator / denominator)
        else:
            orders.append(float('nan'))  # Prevent division by zero issues

    return orders  # Returns a list of computed orders


# Initial guess
x0, y0, z0 = 1, 1, 1

# Find point on the surface
solution = find_point_on_surface(x0, y0, z0)
print("Final solution:", solution)


def driver():
    # question 1 parameters
    x0 = np.asarray([1,1])
    nmax = 300
    tol = 1e-10
    print("Newton Iteration Approximations")
    newtonApp = newton_method_nd(fxn1, dfxn1, x0, tol, nmax, True)
    print("FPI Iteration Approximations")
    fpiApp = fixed_point_method_nd(gfun,Jgfun,x0,1e-15,500,verb=True);
    ellipseApp = find_point_on_surface(1,1,1,tol,nmax)
    ellipseOrder = compute_order_of_convergence(ellipseApp)
    print("Order of Convergence: ")
    for i in range(len(ellipseOrder)):
        print(ellipseOrder[i])
    return -1

driver()

