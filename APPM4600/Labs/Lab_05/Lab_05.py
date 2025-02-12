################################################################################
# This python script presents examples regarding the newton method and its
# application to 1D nonlinear root-finding, as presented in class.
# APPM 4650 Fall 2021
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# First, we define a function we will test the Newton method with. For each
# function we define, we also define its derivative.
# Our test function from previous sections
def fun(x):
    return x + np.cos(x)-3;
    return x**2
def dfun(x):
    return 1 - np.sin(x);

################################################################################
# We now implement the Newton method
def newton_method(f,df,x0,tol,nmax,verb=False):
    #newton method to find root of f starting at guess x0

    #Initialize iterates and iterate list
    xn=x0;
    rn=np.array([x0]);
    # function evaluations
    fn=f(xn); dfn=df(xn);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)
    # defining FP

    if abs(dfn)<dtol:
        #If derivative is too small, Newton will fail. Error message is
        #displayed and code terminates.
        if verb:
            fprintf('\n derivative at initial guess is near 0, try different x0 \n');
    elif abs(dfn)>=1:
        print("Function doesn't converge.")
    else:
        n=0;
        if verb:
            print("\n|--n--|----xn----|---|f(xn)|---|---|f'(xn)|---|");

        #Iteration runs until f(xn) is small enough or nmax iterations are computed.

        while n<=nmax:
            if verb:
                print("|--%d--|%1.8f|%1.8f|%1.8f|" %(n,xn,np.abs(fn),np.abs(dfn)));

            pn = - fn/dfn; #Newton step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;
            if np.abs(dfn)>=1:
                break;

            #Update guess adding Newton step
            xn = xn + pn;

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            dfn=df(xn);
            fn=f(xn);
            nfun+=2;

        r=xn;

        if n>=nmax:
            print("Newton method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Newton method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

# Newton’s method condition check
def newton_convergence_condition(x):
    """ Check if Newton's method will converge at x """
    return abs(dfun(x)) < 1

# Modified bisection method with Newton’s basin check
def bisect_method_with_newton(f, df, a, b, tol, nmax, vrb=False):
    """Bisection method that terminates when the midpoint satisfies Newton’s convergence condition"""

    an, bn = a, b
    xn = (an + bn) / 2
    n = 0
    rn = np.array([xn])

    if f(a) * f(b) >= 0:
        print("\n Interval is inadequate, f(a)*f(b) >= 0. Try again")
        return None

    while n <= nmax:
        if newton_convergence_condition(xn):  # Check Newton’s convergence
            print(f"Bisection terminated at iteration {n} since midpoint {xn} lies in Newton's basin")
            (r,rn,nfun)=newton_method(f,df,xn,tol,nmax,vrb)
            return r,rn,nfun

        if (bn - an) < 2 * tol:
            break

        if f(an) * f(xn) < 0:
            bn = xn
        else:
            an = xn

        xn = (an + bn) / 2
        rn = np.append(rn, xn)
        n += 1

    return xn, rn

################################################################################
# Now, we apply this method to our test function
(r,rn)=bisect_method_with_newton(fun,dfun,3,4,1e-14,100,True);

# We plot n against log10|f(rn)|
plt.plot(np.arange(0,rn.size),np.log10(np.abs(fun(rn))),'r-o');
plt.xlabel('n'); plt.ylabel('log10|f(rn)|');
plt.suptitle("Newton method results");
plt.show();
input();
################################################################################
# Modes of 'failure' of the Newton method
# We now demonstrate that the Newton method can fail in the following ways:
# (1) Converge linearly due to repeated / multiple roots (Can be fixed)
# (2) Take a while to reach the basin of quadratic convergence
# (3) Fail to converge at all (cycle or diverge)
# Newton methods have to be paired with safeguards to become fully robust
