# Code for Lab 07 - APPM 4600

# Import libraries
import numpy as np
from scipy.interpolate import lagrange
import matplotlib.pyplot as plt
import matplotlib;
from scipy.special import comb
plt.rcParams['figure.figsize'] = [10, 5];

def driver():
    # defining functions
    def fun(x):
        return 1/(1+10*x**2); # Runge

    # N = 2,3,...,10
    # Equispaced Points
    # Vandermonde (monomial) interp error
    nn = np.arange(2,11,1);
    yl = [-6.5,1.5];
    poly_interp_logerror(fun,nn);
    plt.ylim(yl);
    plt.title('Vandermonde matrix interpolation log error');
    plt.show();

    #Lagrange function interp error (building Lagrange polynomial basis in a matrix)
    yl = [-6.5,1.5];
    lagrange_interp_logerror(fun,nn); #uncomment when you write this method
    plt.ylim(yl);
    plt.title('Lagrange Equispaced interpolation log error');
    plt.show();

    # Newton function interp error. Try turning "sortpts" on and off (True vs False
    #to see instability due to the points getting too close)
    nn = np.arange(2,11,1);
    yl = [-4.5,1.5];
    srtpts=True;
    newton_interp_logerror(fun,nn,srtpts); #uncomment when you write this method
    plt.ylim(yl);
    plt.title('Newton Equispaced interpolation log error');
    plt.show();

    # N = 11, 12,...,20
    # Equispaced Points
    # Vandermonde (monomial) interp error
    nn = np.arange(11,20,1);
    yl = [-15,1.5];
    poly_interp_logerror(fun,nn);
    plt.ylim(yl);
    plt.title('Vandermonde matrix interpolation log error');
    plt.show();

    #Lagrange function interp error (building Lagrange polynomial basis in a matrix)
    yl = [-16.5,1.5];
    lagrange_interp_logerror(fun,nn); #uncomment when you write this method
    plt.ylim(yl);
    plt.title('Lagrange Equispaced interpolation log error');
    plt.show();

    # Newton function interp error. Try turning "sortpts" on and off (True vs False
    #to see instability due to the points getting too close)
    nn = np.arange(2,11,1);
    yl = [-4.5,1.5];
    srtpts=True;
    newton_interp_logerror(fun,nn,srtpts); #uncomment when you write this method
    plt.ylim(yl);
    plt.title('Newton Equispaced interpolation log error');
    plt.show();
    # Define interpolation nodes
    N = 10  # Number of interpolation points
    x_nodes = np.linspace(-1, 1, N+1)  # Equally spaced interpolation nodes
    y_nodes = fun(x_nodes)  # Function values at nodes

    # Define evaluation points (dense grid)
    x_eval = np.linspace(-1, 1, 1000)  # Fine grid for smooth curve
    y_true = fun(x_eval)  # True function values

    # Compute interpolations
    y_vander = poly_interp(fun, x_nodes, x_eval)  # Monomial (Vandermonde)
    y_lagrange = lagrange_interp(fun, x_nodes, x_eval)  # Lagrange
    y_newton, _ = newton_interp(fun, x_nodes, x_eval, srt=False)  # Newton

    # Plot actual function
    plt.figure(figsize=(8,6))
    plt.plot(x_eval, y_true, 'k-', label="True Function $f(x)$", linewidth=2)

    # Plot interpolations
    plt.plot(x_eval, y_vander, '--', label="Monomial (Vandermonde)", linewidth=1.5)
    plt.plot(x_eval, y_lagrange, '-.', label="Lagrange Interpolation", linewidth=1.5)
    plt.plot(x_eval, y_newton, ':', label="Newton Interpolation", linewidth=1.5)

    # Plot interpolation nodes
    plt.scatter(x_nodes, y_nodes, color='red', zorder=3, label="Interpolation Nodes")

    # Formatting
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Interpolation Approximation vs. True Function")
    plt.grid(True)
    plt.show()


    ############################################################################
    # Now interpolate using chebyshev nodes

################################################################################

# Vandermonde interpolation routine (for comparison only, this is very unstable)
def poly_interp(f,xint,xtrg):
    # polynomial interpolation of function f on n+1 equispaced points on interval [-1,1].
    # output g is the interpolant evaluated at target points y.

    # standard xint are equispaced nodes
    #xi = np.linspace(-1,1,n+1);

    n = len(xint)-1;
    fi = f(xint); #evaluate function at interpolation points

    # Create Vandermonde matrix
    V = np.zeros((n+1,n+1));
    for i in range(n+1):
        V[:,i] = xint**i;

    # Solve for polynomial coefficients
    c = np.linalg.solve(V,fi);
    c = c[::-1];
    g = np.polyval(c,xtrg);
    return g;
################################################################################
# Vandermonde interpolation error plot given an array of n
def poly_interp_logerror(f,nn):
    y = np.linspace(-0.999,0.999,1000);
    for n in nn:
        xi = np.linspace(-1,1,n+1);
        g1 = poly_interp(f,xi,y);
        plt.plot(y,np.log10(np.abs(f(y)-g1)+1e-16),'-.',label='N ='+str(n));

    plt.legend(bbox_to_anchor=(1.1, 1.05));
    return;
################################################################################
# Chebyshev nodes definition
def chebpts(n):
    theta = np.linspace(0,np.pi,n+1);
    return -np.cos(theta);
################################################################################
# Lagrange interpolation
def lagrange_interp(f,xint,xtrg):

    n = len(xint);
    mtrg = len(xtrg);

    # Evaluate a matrix L of size mtrg x n+1 where L[:,j] = Lj(xtrg)
    L = np.ones((mtrg,n));
    w=np.ones(n); psi=np.ones(mtrg);

    for i in range(n):
        for j in range(n):
            if np.abs(j-i)>0:
                w[i] = w[i]*(xint[i]-xint[j]);
        psi = psi*(xtrg-xint[i]);
    w = 1/w;

    fj = 1/(np.transpose(np.tile(xtrg,(n,1))) - np.tile(xint,(mtrg,1)));
    L = fj*np.transpose(np.tile(psi,(n,1)))*np.tile(w,(mtrg,1));

    # Polynomial interpolant is L*y, where y = f(xint)
    g = L@f(xint);
    return g;

# Lagrange interpolation log error for a given array of n
def lagrange_interp_logerror(f, nn):
    y = np.linspace(-0.999, 0.999, 1000)
    for n in nn:
        xi = np.linspace(-1, 1, n+1)
        g1 = lagrange_interp(f, xi, y)
        plt.plot(y, np.log10(np.abs(f(y) - g1) + 1e-16), '-.', label='N ='+str(n))

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    return;

################################################################################
# Newton interpolation

# Function that performs Newton interpolation, evaluating with Horner
def newton_interp(f,xint,xtrg,srt=False):

    n = len(xint)-1;

    # option: sort points
    if srt:
        xint=sortxi(xint);

    fi=f(xint);                 #function values

    D = np.zeros((n+1,n+1)); #matrix of divided differences
    D[0]=fi;
    for i in np.arange(1,n+1):
        # Compute divided differences at row i using row i-1.
        D[i,0:n+1-i]=(D[i-1,1:n+2-i] - D[i-1,0:n+1-i])/(xint[i:n+1] - xint[0:n+1-i]);

    cN = D[:,0]; #Interpolation coefficients are stored on the first column of D.

    # Evaluation (Horner's rule)
    g = cN[n]*np.ones(len(xtrg)); #constant term
    for i in np.arange(n-1,-1,-1):
        g = g*(xtrg-xint[i]) + cN[i];

    return (g,cN);

def sortxi(xi):
    n=len(xi)-1;
    a=xi[0]; b=xi[n];
    xi2=np.zeros(n+1);
    xi2[0]=a; xi2[1]=b;
    xi = np.delete(xi,[0,n]);
    for j in np.arange(2,n+1):
        dj = np.abs(xi2[j-1]-xi);
        mj = np.argmax(dj);
        xi2[j] = xi[mj];
        xi=np.delete(xi,[mj]);

    return xi2;

# Newton interpolation log error for an array of n
def newton_interp_logerror(f,nn,srt=False):
    y = np.linspace(-0.999, 0.999, 100)
    for n in nn:
        xi = np.linspace(-1, 1, n+1)
        g1, _ = newton_interp(f, xi, y, srt)
        plt.plot(y, np.log10(np.abs(f(y) - g1) + 1e-16), '-.', label='N ='+str(n))

    plt.legend(bbox_to_anchor=(1.1, 1.05))
    return;

################################################################################

driver()
