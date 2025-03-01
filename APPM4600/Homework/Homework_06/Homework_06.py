# Code produced for Homework 06 - APPM 4600

#importing libraries
import numpy as np
from numpy import random as rand
import time
import math
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.ticker import LinearLocator
#from matplotlib.animation import FuncAnimation
#from IPython.display import HTML, Video
#from mpl_toolkits.mplot3d import Axes3D
#from timeit import default_timer as timer


################################################################################
# Sub routines and helper functions listed here
################################################################################
# Newton method in n dimensions implementation
def newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1; nJ=0; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        # compute n x n Jacobian matrix
        Jn = Jf(xn);
        nJ+=1;

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

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

################################################################################
# Lazy Newton method (chord iteration) in n dimensions implementation
def lazy_newton_method_nd(f,Jf,x0,tol,nmax,verb=False):

    # Initialize arrays and function value
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    # compute n x n Jacobian matrix (ONLY ONCE)
    Jn = Jf(xn);

    # Use pivoted LU factorization to solve systems for Jf. Makes lusolve O(n^2)
    lu, piv = lu_factor(Jn);

    n=0;
    nf=1; nJ=1; #function and Jacobian evals
    npn=1;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:

        if verb:
            print("|--%d--|%1.7f|%1.12f|" %(n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        # Newton step (we could check whether Jn is close to singular here)
        pn = -lu_solve((lu, piv), Fn); #We use lu solve instead of pn = -np.linalg.solve(Jn,Fn);
        xn = xn + pn;
        npn = np.linalg.norm(pn); #size of Newton step

        n+=1;
        rn = np.vstack((rn,xn));
        Fn = f(xn);
        nf+=1;

    r=xn;

    if verb:
        if np.linalg.norm(Fn)>tol:
            print("Lazy Newton method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Lazy Newton method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return (r,rn,nf,nJ);

################################################################################
# Implementation of Broyden method. B0 can either be an approx of Jf(x0) (Bmat='fwd'),
# an approx of its inverse (Bmat='inv') or the identity (Bmat='Id')
def broyden_method_nd(f,B0,x0,tol,nmax,Bmat='Id',verb=False):

    # Initialize arrays and function value
    d = x0.shape[0];
    xn = x0; #initial guess
    rn = x0; #list of iterates
    Fn = f(xn); #function value vector
    n=0;
    nf=1;
    npn=1;

    #####################################################################
    # Create functions to apply B0 or its inverse
    if Bmat=='fwd':
        #B0 is an approximation of Jf(x0)
        # Use pivoted LU factorization to solve systems for B0. Makes lusolve O(n^2)
        lu, piv = lu_factor(B0);
        luT, pivT = lu_factor(B0.T);

        def Bapp(x): return lu_solve((lu, piv), x); #np.linalg.solve(B0,x);
        def BTapp(x): return lu_solve((luT, pivT), x) #np.linalg.solve(B0.T,x);
    elif Bmat=='inv':
        #B0 is an approximation of the inverse of Jf(x0)
        def Bapp(x): return B0 @ x;
        def BTapp(x): return B0.T @ x;
    else:
        Bmat='Id';
        #default is the identity
        def Bapp(x): return x;
        def BTapp(x): return x;
    ####################################################################
    # Define function that applies Bapp(x)+Un*Vn.T*x depending on inputs
    def Inapp(Bapp,Bmat,Un,Vn,x):
        rk=Un.shape[0];

        if Bmat=='Id':
            y=x;
        else:
            y=Bapp(x);

        if rk>0:
            y=y+Un.T@(Vn@x);

        return y;
    #####################################################################

    # Initialize low rank matrices Un and Vn
    Un = np.zeros((0,d)); Vn=Un;

    if verb:
        print("|--n--|----xn----|---|f(xn)|---|");

    while npn>tol and n<=nmax:
        if verb:
            print("|--%d--|%1.7f|%1.12f|" % (n,np.linalg.norm(xn),np.linalg.norm(Fn)));

        #Broyden step xn = xn -B_n\Fn
        dn = -Inapp(Bapp,Bmat,Un,Vn,Fn);
        # Update xn
        xn = xn + dn;
        npn=np.linalg.norm(dn);

        ###########################################################
        ###########################################################
        # Update In using only the previous I_n-1
        #(this is equivalent to the explicit update formula)
        Fn1 = f(xn);
        dFn = Fn1-Fn;
        nf+=1;
        I0rn = Inapp(Bapp,Bmat,Un,Vn,dFn); #In^{-1}*(Fn+1 - Fn)
        un = dn - I0rn;                    #un = dn - In^{-1}*dFn
        cn = dn.T @ (I0rn);                # We divide un by dn^T In^{-1}*dFn
        # The end goal is to add the rank 1 u*v' update as the next columns of
        # Vn and Un, as is done in, say, the eigendecomposition
        Vn = np.vstack((Vn,Inapp(BTapp,Bmat,Vn,Un,dn)));
        Un = np.vstack((Un,(1/cn)*un));

        n+=1;
        Fn=Fn1;
        rn = np.vstack((rn,xn));

    r=xn;

    if verb:
        if npn>tol:
            print("Broyden method failed to converge, n=%d, |F(xn)|=%1.1e\n" % (nmax,np.linalg.norm(Fn)));
        else:
            print("Broyden method converged, n=%d, |F(xn)|=%1.1e\n" % (n,np.linalg.norm(Fn)));

    return(r,rn,nf)

################################################################################
# Steepest descent algorithm
def steepest_descent(f,Gf,x0,tol,nmax,type='swolfe',verb=True):
    # Set linesearch parameters
    c1=1e-3; c2=0.9; mxbck=10;
    # Initialize alpha, fn and pn
    alpha=1;
    xn = x0; #current iterate
    rn = x0; #list of iterates
    fn = f(xn); nf=1; #function eval
    pn = -Gf(xn); ng=1; #gradient eval

    # if verb is true, prints table of results
    if verb:
        print("|--n--|-alpha-|----|xn|----|---|f(xn)|---|---|Gf(xn)|---|");

    # while the size of the step is > tol and n less than nmax
    n=0;
    while n<=nmax and np.linalg.norm(pn)>tol:
        if verb:
            print("|--%d--|%1.5f|%1.7f|%1.7f|%1.7f|" %(n,alpha,np.linalg.norm(xn),np.abs(fn),np.linalg.norm(pn)));

        # Use line_search to determine a good alpha, and new step xn = xn + alpha*pn
        (xn,alpha,nfl,ngl)=line_search(f,Gf,xn,pn,type,mxbck,c1,c2);

        nf=nf+nfl; ng=ng+ngl; #update function and gradient eval counts
        fn = f(xn); #update function evaluation
        pn = -Gf(xn); # update gradient evaluation
        n+=1;
        rn=np.vstack((rn,xn)); #add xn to list of iterates

    r = xn; # approx root is last iterate

    return (r,rn,nf,ng);

################################################################################
# Backtracking line-search algorithm (to find an for the step xn + an*pn)
def line_search(f,Gf,x0,p,type,mxbck,c1,c2):
    alpha=2;
    n=0;
    cond=False; #condition (if True, we accept alpha)
    f0 = f(x0); # initial function value
    Gdotp = p.T @ Gf(x0); #initial directional derivative
    nf=1;ng=1; # number of function and grad evaluations

    # we backtrack until our conditions are met or we've halved alpha too much
    while n<=mxbck and (not cond):
        alpha=0.5*alpha;
        x1 = x0+alpha*p;
        # Armijo condition of sufficient descent. We draw a line and only accept
        # a step if our function value is under this line.
        Armijo = f(x1) <= f0 + c1*alpha*Gdotp;
        nf+=1;
        if type=='wolfe':
            #Wolfe (Armijo sufficient descent and simple curvature conditions)
            # that is, the slope at new point is lower
            Curvature = p.T @ Gf(x1) >= c2*Gdotp;
            # condition is sufficient descent AND slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        elif type=='swolfe':
            #Symmetric Wolfe (Armijo and symmetric curvature)
            # that is, the slope at new point is lower in absolute value
            Curvature = np.abs(p.T @ Gf(x1)) <= c2*np.abs(Gdotp);
            # condition is sufficient descent AND symmetric slope reduction
            cond = Armijo and Curvature;
            ng+=1;
        else:
            # Default is Armijo only (sufficient descent)
            cond = Armijo;

        n+=1;

    return(x1,alpha,nf,ng);

################################################################################
def order_calc(approx, converged):
    print(approx)
    orders = []
    if converged:
        for k in range(2, len(approx)-1):
            num = np.log(abs((approx[k+1] - approx[k]) / (approx[k] - approx[k-1])))
            den = np.log(abs((approx[k] - approx[k-1]) / (approx[k-1] - approx[k-2])))
            if den != 0:
                orders.append(num/den)
                print(orders)
        return orders
    else:
        return orders

################################################################################
# Driver Function
################################################################################
def driver():
    # Driver for Question 1 - Lazy Newton and Broyden
    # question 1 functions
    def F1(x):
        print(x[0])
        return np.array([x[0]**2+x[1]**2-4, math.exp(x[0])+x[1]-1]);
    def JF1(x):
        return np.array([[2*x[0], 2*x[1]],[math.exp(x[0]),1]]);


    # defining x0 inital guesses
    x0_1 = np.array([1.0,1.0])
    x0_2 = np.array([1.0,-1.0])
    x0_3 = np.array([0.0,0.0])

    # defining stopping tolerance
    tol_1 = 1.5e-7

    # defining nmax
    nmax_1 = 100
    nmax_2=1500;

    # convergence status
    converged = False

    # Apply Newton Method x0_1
    print("-------------------- Newton x0_1 --------------------")
    try:
        (rN,rnN,nfN,nJN) = newton_method_nd(F1,JF1,x0_1,tol_1,nmax_1,'swolfe', True);
        print(rN)
        converged = True
#    order = order_calc(rnN, converged)
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Newton Method x0_2
    try:
        print("-------------------- Newton x0_2 --------------------")
        (rN,rnN,nfN,nJN) = newton_method_nd(F1,JF1,x0_2,tol_1,nmax_1,True);
        print(rN)
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Newton Method x0_3
    try:
        print("-------------------- Newton x0_3 --------------------")
        (rN,rnN,nfN,nJN) = newton_method_nd(F1,JF1,x0_3,tol_1,nmax_1,True);
        print(rN)
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Lazy Newton (chord iteration) x0_1
    try:
        print("-------------------- Lazy x0_1 --------------------")
        (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F1,JF1,x0_1,tol_1,nmax_2,True);
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Lazy Newton (chord iteration) x0_2
#    try:
    print("-------------------- Lazy x0_2 --------------------")
    (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F1,JF1,x0_2,tol_1,nmax_2,True);
    converged = True
#    except:
    '''
    print("error "+str(IOError))
    print("Method failed to converge")
    converged = False
    '''

    # Apply Lazy Newton (chord iteration) x0_3
    try:
        print("-------------------- Lazy x0_3 --------------------")
        (rLN,rnLN,nfLN,nJLN) = lazy_newton_method_nd(F1,JF1,x0_3,tol_1,nmax_2,True);
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Broyden Method x0_1
    try:
        print("-------------------- Broyden x0_1 --------------------")
        Bmat='fwd';
        B0 = JF1(x0_1); 
        (rB,rnB,nfB) = broyden_method_nd(F1,B0,x0_1,tol_1,nmax_1,Bmat,True);
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Broyden Method x0_2
    try:
        print("-------------------- Broyden x0_2 --------------------")
        Bmat='fwd';
        B0 = JF1(x0_2); 
        (rB,rnB,nfB) = broyden_method_nd(F1,B0,x0_2,tol_1,nmax_1,Bmat,True);
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Apply Broyden Method x0_3
    try:
        print("-------------------- Broyden x0_3 --------------------")
        Bmat='fwd';
        B0 = JF1(x0_3); 
        (rB,rnB,nfB) = broyden_method_nd(F1,B0,x0_3,tol_1,nmax_1,Bmat,True);
        converged = True
    except:
        print("error "+str(IOError))
        print("Method failed to converge")
        converged = False

    # Question 2 Driver
    # question 2 functions
    def F2(x):
        return np.array([x[0] + np.cos(x[0] * x[1] * x[2]) - 1,
                         (1 - x[0])**(1/4) + x[1] + 0.05 * x[2]**2 - x[2]*0.15 - 1,
                         -x[0]**2 - 0.1 * x[1]**2 + 0.01 * x[1] + x[2] - 1])
    def JF2(x):
        return np.array([[1 - x[1] * x[2] * np.sin(x[0] * x[1] * x[2]),
                          -x[0] * x[2] * np.sin(x[0] * x[1] * x[2]),
                          -x[0] * x[1] * np.sin(x[0] * x[1] * x[2])],
                         [-0.25 * (1 - x[0])**(-3/4), 1, 0.1 * x[2] - 0.15],
                         [-2 * x[0], -0.2 * x[1] + 0.01, 1]])
    def q(x):
        Fun = F2(x)
        return 0.5 * np.dot(Fun, Fun)
    def Gq(x):
        return JF2(x).T @ F2(x)

    x0_4 = np.asarray([0,1,1])
    tol_3 = 1e-6
    tol_4 = 5e-2

    print("-------------------- Newton pt.2 --------------------")
    (rN,rnN,nfN,nJN) = newton_method_nd(F2,JF2,x0_4,tol_3,nmax_1,True);
    print(rnN[len(rnN)-1])

    print("-------------------- Steepest pt.2 --------------------")
    (r,rn,nf,ng)=steepest_descent(q,Gq,x0_4,tol_3,nmax_2,True);
    print(rn[len(rn)-1])

    print("-------------------- Newton -> Steepest pt.2 --------------------")
    (r,rn,nf,ng)=steepest_descent(q,Gq,x0_4,tol_4,nmax_2,True);
    (rN,rnN,nfN,nJN) = newton_method_nd(F2,JF2,rN,tol_3,nmax_1,True);
    print(rnN[len(rnN)-1])

    return -1

driver()
