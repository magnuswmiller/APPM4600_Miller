# Code For Lab 10 - APPM 4600

# installing libraries
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import math
from scipy.integrate import quad

# Helper Functions:
def eval_legendre(order, x_val):
    poly_vector = np.ones(order+1)
    for i in range(len(poly_vector)):
        if(i == 0):
            poly_vector[i] = 1
        elif(i == 1):
            poly_vector[i] = x_val
        else:
            coef = 1/(i)
            t1 = ((2*(i-1))+1)*x_val*poly_vector[i-1]
            t2 = (i-1)*poly_vector[i-2]
            poly_vector[i] = coef*(t1-t2)
    return poly_vector

def legendre_poly(order, x):
    n_poly = eval_legendre(order, x)[-1]
    return n_poly

def eval_cheb(order, x):
    poly_vector = np.ones(order+1)
    for i in range(len(poly_vector)):
        if(i == 0):
            poly_vector[i] = 1
        elif(i == 1):
            poly_vector[i] = x
        else:
            poly_vector[i] = 2*x*poly_vector[i-1]-poly_vector[i-2]
    return poly_vector

def cheb_poly(order, x):
    n_poly = eval_cheb(order,x)[-1]
    return n_poly

def eval_legendre_expansion(f,a,b,w,n,x): 
    # This subroutine evaluates the Legendre expansion
    # Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab 
    p = eval_legendre(n,x)
    # initialize the sum to 0 
    pval = 0.0    
    for j in range(0,n+1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: legendre_poly(j,x)
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x)**2*w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x)*w(x)*f(x)/norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*p[j] 
    return pval

def eval_cheb_expansion(f,a,b,w,n,x): 
    # This subroutine evaluates the Legendre expansion
    # Evaluate all the Legendre polynomials at x that are needed
    # by calling your code from prelab 
    p = eval_cheb(n,x)
    # initialize the sum to 0 
    pval = 0.0    
    for j in range(0,n+1):
        # make a function handle for evaluating phi_j(x)
        phi_j = lambda x: cheb_poly(j,x)
        # make a function handle for evaluating phi_j^2(x)*w(x)
        phi_j_sq = lambda x: phi_j(x)**2*w(x)
        # use the quad function from scipy to evaluate normalizations
        norm_fac,err = quad(phi_j_sq, a, b)
        # make a function handle for phi_j(x)*f(x)*w(x)/norm_fac
        func_j = lambda x: phi_j(x)*w(x)*f(x)/norm_fac
        # use the quad function from scipy to evaluate coeffs
        aj,err = quad(func_j, a, b)
        # accumulate into pval
        pval = pval+aj*p[j] 
    return pval

# Driver Function
def driver():
    #  function you want to approximate
    f = lambda x: 1/(1+x**2)

    # Interval of interest    
    a = -1
    b = 1

    # weight function    
    w = lambda x: 1.
    w2 = lambda x: 1/np.sqrt(1-x**2)

    # order of approximation
    n = 2

    #  Number of points you want to sample in [a,b]
    N = 1000

    # equispaced nodes
    xeval = np.linspace(a,b,N+1)
    pval = np.zeros(N+1)

    for kk in range(N+1):
      pval[kk] = eval_legendre_expansion(f,a,b,w,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    print("hellow")
    plt.figure()    
    plt.plot(xeval,fex,'-', label= 'f(x)')
    plt.plot(xeval,pval,'--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval-fex)
    plt.semilogy(xeval,err,'ro--',markersize=4, label='error')
    plt.legend()
    plt.show()

    # Chebyshev nodes
    #j = np.arange(1, N+2)
    #xeval_cheb = np.cos((2*j - 1) * np.pi / (2*(N)))
    #print(xeval_cheb)
    pval_cheb = np.zeros(N+1)

    for kk in range(N+1):
      pval_cheb[kk] = eval_cheb_expansion(f,a,b,w2,n,xeval[kk])
      
    ''' create vector with exact values'''
    fex = np.zeros(N+1)
    for kk in range(N+1):
        fex[kk] = f(xeval[kk])
        
    plt.figure()    
    plt.plot(xeval,fex,'-', label= 'f(x)')
    plt.plot(xeval,pval_cheb,'--',label= 'Expansion') 
    plt.legend()
    plt.show()    
    
    err = abs(pval_cheb-fex)
    plt.semilogy(xeval,err,'ro--',markersize=4, label='error')
    plt.legend()
    plt.show()

    return -1
    
if __name__ == '__main__':
  # run the drivers only if this is called from the command line
  driver()         
