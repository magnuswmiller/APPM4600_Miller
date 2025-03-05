# Code for Lab 08 - APPM 4600

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.linalg import inv

# Line-Eval Helper Function
def line_eval(x0,f_x0,x1,f_x1, a):
    return f_x0 + ((f_x1 - f_x0) / (x1 - x0)) * (a - x0)

# Linear Helper Functions
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    
    for jint in range(Nint):
        a1 = xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''

        '''temporarily store your info for creating a line in the interval of 
         interest'''
        ind = np.where((xeval >= a1) & (xeval <= b1))[0]  # Get indices
        n = len(ind)  # Number of points in this interval

        for kk in range(n):
            yeval[ind[kk]] = line_eval(a1, fa1, b1, fb1, xeval[ind[kk]])  # Use correct indices
    return yeval

# Cubic Helper Functions
def create_natural_spline(yint,xint,N):

#    create the right  hand side for the linear system
    b = np.zeros(N-1);
#  vector values
    h = np.zeros(N);
    h[0] = xint[1]-xint[0]
    for i in range(1,N):
       h[i] = xint[i+1] - xint[i]
       b[i-1] = ((yint[i+1]-yint[i])/h[i] - (yint[i]-yint[i-1])/h[i-1])/(h[i-1]+h[i]);

#  create the matrix M so you can solve for the A values
    M = np.zeros((N-1,N-1));
    for i in np.arange(N-1):
        M[i,i] = 4/12;

        if i<(N-2):
            M[i,i+1] = h[i+1]/(6*(h[i]+h[i+1]));

        if i>0:
            M[i,i-1] = h[i]/(6*(h[i]+h[i+1]));

# Solve system M*A = b to find coefficients (a[1],a[2],...,a[N-1]).
    A = np.zeros(N+1);
    A[1:N] = np,linalg.solve(M,b)

#  Create the linear coefficients
    B = np.zeros(N)
    C = np.zeros(N)
    for j in range(N):
        h_j = x[j+1] - x[j]  # Interval width

        B[j] = (A[j+1] - A[j]) / h_j - (h_j / 3) * (2 * M[j] + M[j+1])  # First derivative term
        C[j] = M[j]  # Second derivative term
        D[j] = (M[j+1] - M[j]) / (3 * h_j)  # Third derivative term

    return(A,B,C)

def eval_local_spline(xeval,xi,xip,Ai,Aip,B,C):
# Evaluates the local spline as defined in class
# xip = x_{i+1}; xi = x_i
# Aip = A_{i}; Ai = A_{i-1}

    hi = xip-xi;

    yeval =
    return yeval;


def  eval_cubic_spline(xeval,Neval,xint,Nint,A,B,C):

    yeval = np.zeros(Neval+1);

    for j in range(Nint):
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        atmp = xint[j];
        btmp= xint[j+1];

#   find indices of values of xeval in the interval
        ind= np.where((xeval >= atmp) & (xeval <= btmp));
        xloc = xeval[ind];

# evaluate the spline
        yloc = eval_local_spline(xloc,atmp,btmp,A[j],A[j+1],B[j],C[j])
#   copy into yeval
        yeval[ind] = yloc

    return(yeval)

# Driver Function
def driver():
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1

    # Points to evaluate at
    num_eval = 100
    x_eval = np.linspace(a,b,num_eval)

    # Number of intervals
    num_int = 10

    # Evaluate the linear spline
    y_eval = eval_lin_spline(x_eval, num_eval, a, b, f, num_int)

    f_x_eval = f(x_eval)

    # Plotting
    plt.figure()
    plt.plot(x_eval,f_x_eval, 'r',label='xeval, fex')
    plt.plot(x_eval,y_eval, 'b',label='xeval, yeval')
    plt.legend()
    plt.show()

    err = abs(y_eval-f_x_eval)
    plt.figure()
    plt.plot(x_eval,err,'r')
    plt.show()

    def f(x):
        return np.exp(x);
    a = 0;
    b = 1;

    ''' number of intervals'''
    Nint = 3;
    xint = np.linspace(a,b,Nint+1);
    yint = f(xint);

    ''' create points you want to evaluate at'''
    Neval = 100;
    xeval =  np.linspace(xint[0],xint[Nint],Neval+1);

#   Create the coefficients for the natural spline
    (A,B,C) = create_natural_spline(yint,xint,Nint);

#  evaluate the cubic spline
    yeval = eval_cubic_spline(xeval,Neval,xint,Nint,A,B,C);


    ''' evaluate f at the evaluation points'''
    fex = f(xeval)

    nerr = norm(fex-yeval)
    print('nerr = ', nerr)

    plt.figure()
    plt.plot(xeval,fex,'ro-',label='exact function')
    plt.plot(xeval,yeval,'bs--',label='natural spline')
    plt.legend
    plt.show()

    err = abs(yeval-fex)
    plt.figure()
    plt.semilogy(xeval,err,'ro--',label='absolute error')
    plt.legend()
    plt.show()
    return -1

driver()
