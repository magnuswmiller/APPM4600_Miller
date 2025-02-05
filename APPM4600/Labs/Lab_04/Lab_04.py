# import libraries
import numpy as np
import math
    
def driver():

    # functions:
     f1 = lambda x: ((10)/(x+4))**(0.5)

     Nmax = 150
     tol = 1e-10

     
     x0 = 1.5
     [xstar,ier,approx] = fixedpt(f1,x0,tol,Nmax)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print('Error message reads:',ier)
     print('Number of iterations: ', len(approx))
     alpha = order(approx)
     print('Alpha value: ', alpha[len(alpha)-1])
     aitken = aitkenSeq(approx)
     print(aitken)
     print(order(aitken))

# define routines
def fixedpt(f,x0,tol,Nmax):

    ''' x0 = initial guess''' 
    ''' Nmax = max number of iterations'''
    ''' tol = stopping tolerance'''

    count = 0
    approx = [x0]
    while (count <Nmax):
       count = count +1
       x1 = f(x0)
       if (abs(x1-x0) <tol):
          xstar = x1
          ier = 0
          return [xstar,ier,approx]
       x0 = x1
       print(float(x0))
       approx.append(float(x0))

    xstar = x1
    ier = 1
    return [xstar, ier, approx]

def order(approx):
    alpha = []
    for i in range(len(approx)-1):
        a = (math.log(abs((approx[i+1]-approx[i])/(approx[i]-approx[i-1]))))/(math.log(abs((approx[i]-approx[i-1])/(approx[i-1]-approx[i-2]))))
        alpha.append(a)
    return alpha

def aitkenSeq(approx):
    aitken = []
    for i in range(len(approx)-2):
        phat = approx[i]-((approx[i+1]-approx[i])**2/(approx[i+2]-(2*approx[i+1])+approx[i]))
        aitken.append(phat)
    return aitken
    

driver()
