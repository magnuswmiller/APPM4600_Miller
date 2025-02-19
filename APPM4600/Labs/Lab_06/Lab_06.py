# Code for Lab 06

# importing libraries
import numpy as np
from scipy import io, integrate, linalg, signal
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
import math

# driver fxn
def driver():
    # pre-lab:
    x = np.pi/2
    h = 0.01*2.**(-np.arange(0,10))
    forDifApp = forwardDif(fxn1, x, h)
    forAlphaApp = orderApprox(forDifApp)
    print("Approximations of the Forward Difference:")
    printApprox(forDifApp)
    print("Order of convergence: ", forAlphaApp[len(forAlphaApp)-1])

    cenDifApp = centeredDif(fxn1, x, h)
    cenAlphaApp = orderApprox(cenDifApp)
    print("Approximations of the Forward Difference:")
    printApprox(cenDifApp)
    print("Order of convergence: ", cenAlphaApp[len(cenAlphaApp)-1])

    print("Lazy Newton")
    nmax=100;
    x0 = np.asarray([1,0])
    tol = 1e-10
    lazyNewtonApp = lazyNewton(fxn2, jfxn2, x0, tol, nmax)
    print(lazyNewtonApp)

    return -1

# defining functions:
def fxn1(x):
    return np.cos(x)

def fxn2(x):
    return np.asarray([(4*(x[0])**2)+(x[1])**2-4,(x[0])+(x[1])-np.sin(x[0]-x[1])])

def jfxn2(x):
    return np.asarray([[8*x[0],2*x[1]],[1-np.cos(x[0]-x[1]),np.cos(x[0]-x[1])+1]])

# helper fxns:
def forwardDif(fxn, s, h):
    approx = []
    for i in range(len(h)-1):
        deriv = float((fxn1(s+h[i])-fxn1(s))/h[i])
        approx.append(deriv)
    return approx

def centeredDif(fxn, s, h):
    approx = []
    for i in range(len(h)-1):
        deriv = float((fxn1(s+h[i])-fxn1(s-h[i]))/(2*h[i]))
        approx.append(deriv)
    return approx 

def orderApprox(approx):
    alphaApp = []
    for i, item in enumerate(approx[2:len(approx)-1],2):
        num = np.log(np.abs((approx[i+1]-approx[i])/(approx[i]-approx[i-1])))
        den = np.log(np.abs((approx[i]-approx[i-1])/(approx[i-1]-approx[i-2])))
        alpha = num/den
        alphaApp.append(alpha)
    return alphaApp

# Building Lazy Newton Method
def lazyNewton(fxn, jfxn, x0, tol, nmax):
    xn = x0
    rn = x0
    f_xn = fxn(xn)
    jf_xn = jfxn(xn)

    lu, piv = lu_factor(jf_xn)

    iter = 0
    fxnIter = 1
    jfxnIter = 1
    eps = 1

    while eps > tol and iter <= nmax:
        if len(rn) >= 3 and np.sqrt((rn[(iter,0)]-rn[(iter-1,0)])**2+(rn[(iter,1)]-rn[(iter-1,1)])**2) > 1e-5:
            jf_xn = jfxn(xn)
            jfxnIter += 1

        pn = -lu_solve((lu, piv), f_xn)
        xn = xn + pn
        eps = np.linalg.norm(pn)

        iter += 1
        f_xn = fxn(xn)
        fxnIter += 1
        rn = np.vstack((rn,xn))
        print(eps)
    approx = xn
    return (approx, rn, fxnIter, jfxnIter)

def printApprox(approx):
    print("|---Iteration---|---Approximation---|")
    for i, item in enumerate(approx[0:],0):
        print("|-------%i-------|%.16f|"%(i,item))
    return -1

driver()
