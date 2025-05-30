################################################################################
# This python script presents an example and the application of methods
# (bisection and fixed point) to 1D nonlinear root-finding, as presented in class.
# APPM 4650 Spring 2025
################################################################################
# Import libraries
import numpy as np;
import matplotlib.pyplot as plt;

# The driver function is the code that will be executed when running this .py file
# on the command line.
def driver():
    # First, we define a function we will test our methods with. For each
    # function we define, we also define its derivative.

    '''
    # (1) Bisection method test(s). We run the bisect_method function for a defined
    # fun in interval [a,b] such that f(a)f(b)<0. We run the method until our
    # error estimate (b-a)/2 is smaller than TOL or n is bigger than nmax.

    def fun(x):
        return x + np.cos(x)-3;
    def dfun(x):
        return 1 - np.sin(x);

    (r1,r1n)=bisect_method(fun,3,4,1e-14,100,True);
    plt.show();

    # (2) Fixed point iteration method test(s): Given f(x) = 0, we run tests to
    # find the fixed point of g(x) = x - cf(x) for c a non-zero constant.
    # We run the method until |xn-x_{n-1}| is smaller than TOL
    # or n is bigger than nmax.
    '''

    '''
    #Example 1: c = 2
    def g1(x):
        return x - 2*fun(x);
    def dg1(x):
        return 1 - 2*dfun(x);

    (rf,rfn)=fixed_point_method(g1,dg1,3.5,3,4,1e-14,100,True);
    plt.show();

    # Example 2: c = (1/3)
    def g2(x):
        return x - (1/3)*fun(x);
    def dg2(x):
        return 1 - (1/3)*dfun(x);

    (rf2,rf2n)=fixed_point_method(g2,dg2,3.5,3,4,1e-14,100,True);
    plt.show();
    '''

    # Excersise 1:
    def fun1(x):
        return (x**2)*(x-1)
    def dfun1(x):
        return x*(3*x-2)
    
    # Input A:
    try:
        print("Exercise 1a:")
        (rf1,rf1n)=bisect_method(fun1,0.5,2,1e-14,100,True);
        plt.show();
    except:
        pass

    # Input B:
    try:
        print("Exercise 1b:")
        (rf12,rf12n)=bisect_method(fun1,-1,0.5,1e-14,100,True);
        plt.show();
    except:
        pass

    # Input C:
    try:
        print("Exercise 1c:")
        (rf13,rf13n)=bisect_method(fun1,-1,2,1e-14,100,True);
        plt.show();
    except:
        pass

    # Exercise 2:
    # A
    def fun2(x):
        return (x-1)*(x-3)*(x-5)
    def dfun2(x):
        return (3*x**2)-(18*x)+23 
    try:
        print("Exercise 2a:")
        (rf2,rf2n)=bisect_method(fun2,0,2.4,1e-5,100,True);
        plt.show();
    except:
        pass

    # B
    def fun3(x):
        return (x-1)**2*(x-3)
    def dfun3(x):
        return (x-1)*((3*x)-7)
    try:
        print("Exercise 2b:")
        (rf3,rf3n)=bisect_method(fun3,0,2,1e-5,100,True);
        plt.show()
    except:
        pass

    # C
    def fun4(x):
        return np.sin(x) 
    def dfun4(x):
        return np.cos(x)
    try:
        print("Exercise 2c:")
        (rf4,rf4n)=bisect_method(fun4,0,0.1,1e-5,100,True);
        plt.show;
    except:
        pass

    # Exercise 3:
    # A
    def fun5(x):
        return x*(1+((7-x**5)/(x**2)))**3
    def dfun5(x):
        return -(((x**5-x**2-7)**2)*(10*x**5-x**2+35))/x**6
    try:
        print("exercise 3a:")
        (rf5,rf5n)=fixed_point_method(fun5,dfun5,5,0,10,1e-14,100,True);
        plt.show();
    except:
        pass

    # B
    def fun6(x):
        return -1
    def dfun6(x):
        return -1
    try:
        print("exercise 3b:")
        (rf6,rf6n)=fixed_point_method(fun6,dfun6,5,0,10,1e-14,100,True);
        plt.show();
    except:
        pass

    # C
    def fun7(x):
        return -1
    def dfun7(x):
        return -1
    try:
        print("exercise 3c:")
        (rf7,rf7n)=fixed_point_method(fun7,dfun7,5,0,10,1e-14,100,True);
        plt.show();
    except:
        pass

    # D
    def fun8(x):
        return -1
    def dfun8(x):
        return -1
    try:
        print("exercise 3d:")
        (rf8,rf8n)=fixed_point_method(fun8,dfun8,5,0,10,1e-14,100,True);
        plt.show();
    except:
        pass

################################################################################
# Here we write any functions to be used by driver. They can be in any order we
# want.
def bisect_method(f,a,b,tol,nmax,vrb=False):
    #Bisection method applied to f between a and b

    # Initial values for interval [an,bn], midpoint xn
    an = a; bn=b; n=0;
    xn = (an+bn)/2;
    # Current guess is stored at rn[n]
    rn=np.array([xn]);
    r=xn;
    ier=0;

    if vrb:
        print("\n Bisection method with nmax=%d and tol=%1.1e\n" % (nmax, tol));

    # The code cannot work if f(a) and f(b) have the same sign.
    # In this case, the code displays an error message, outputs empty answers and exits.
    if f(a)*f(b)>=0:
        print("\n Interval is inadequate, f(a)*f(b)>=0. Try again \n")
        #print("f(a)*f(b) = %1.1f \n" % f(a)*f(b));
        r = None;
        return r
    else:
        #If f(a)f(b), we proceed with the method.
        if vrb:
            print("\n|--n--|--an--|--bn--|----xn----|-|bn-an|--|---|f(xn)|---|");

        while n<=nmax:
            if vrb:
                print("|--%d--|%1.4f|%1.4f|%1.8f|%1.8f|%1.8f|" % (n,an,bn,xn,bn-an,np.abs(f(xn))));

            # Bisection method step: test subintervals [an,xn] and [xn,bn]
            # If the estimate for the error (root-xn) is less than tol, exit
            if (bn-an)<2*tol: # better test than np.abs(f(xn))<tol
                ier=1;
                break;

            # If f(an)*f(xn)<0, pick left interval, update bn
            if f(an)*f(xn)<0:
                bn=xn;
            else:
                #else, pick right interval, update an
                an=xn;

            # update midpoint xn, increase n.
            n += 1;
            xn = (an+bn)/2;
            rn = np.append(rn,xn);

    # Set root estimate to xn.
    r=xn;

    if vrb:
        ########################################################################
        # Approximate error log-log plot
        logploterr(rn,r);
        plt.title('Bisection method: Log error vs n');
        ########################################################################

    return r, rn;

def fixed_point_method(g,dg,x0,a,b,tol,nmax,vrb=False):
     # Fixed point iteration method applied to find the fixed point of g from starting point x0

     # Initial values
     n=0;
     xn = x0;
     # Current guess is stored at rn[n]
     rn=np.array([xn]);
     r=xn;

     if vrb:
         print("\n Fixed point method with nmax=%d and tol=%1.1e\n" % (nmax, tol));
         print("\n|--n--|----xn----|---|g(xn)|---|---|g'(xn)---|");

     while n<=nmax:
         if vrb:
             print("|--%d--|%1.8f|%1.8f|%1.4f|" % (n,xn,np.abs(g(xn)),np.abs(dg(xn))));

         # If the estimate is approximately a root, get out of while loop
         if np.abs(g(xn)-xn)<tol:
             #(break is an instruction that gets out of the while loop)
             break;

         # update iterate xn, increase n.
         n += 1;
         xn = g(xn); #apply g (fixed point step)
         rn = np.append(rn,xn); #add new guess to list of iterates

     # Set root estimate to xn.
     r=xn;

     if vrb:
         ########################################################################
         # Approximate error log-log plot
         logploterr(rn,r);
         plt.title('Fixed Point Iteration: Log error vs n');
         ########################################################################

     return r, rn;

# This auxiliary function plots approximate log error for a list of iterates given
# the list (in array rn) and the exact root r (or our best guess)
def logploterr(rn,r):
    n = rn.size-1;
    e = np.abs(r-rn[0:n]);
    #length of interval
    nn = np.arange(0,n);
    #log plot error vs iteration number
    plt.plot(nn,np.log2(e),'r--');
    plt.xlabel('n'); plt.ylabel('log2(error)');
    return;

################################################################################
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()
