# code for homework 03

# import libraries
import numpy as np
import matplotlib.pyplot as plt

# driver fxn
def driver():

    # Question 1 calling script   
    print("---------- Question 1 Output ----------")
    f1 = lambda x: 2*x-1-np.sin(x)
    a1 = 0
    b1 = 1

    tol1 = 1e-8

    [astar1,ier1, count1] = bisection(f1,a1,b1,tol1)
    print('the approximate root is',astar1)
    print('number of iterations: ', count1)
    print('the error message reads:',ier1)
    print('f(astar1) =', f1(astar1))

    # Question 2 calling script
    print("---------- Question 2a Output ----------")
    f2 = lambda x: (x-5)**9
    a2 = 4.82
    b2 = 5.2

    tol2 = 1e-4

    [astar2,ier2,count2] = bisection(f2,a2,b2,tol2)
    print('the approximate root is',astar2)
    print('number of iterations: ', count2)
    print('the error message reads:',ier2)
    print('f(astar2) =', f2(astar2))

    print("---------- Question 2b Output ----------")
    f3 = lambda x: (x**9)-(45*x**8)+(900*x**7)-(10500*x**6)+(78750*x**5)-(393750*x**4)+(1312500*x**3)-(2812500*x**2)+(3515625*x)-1953125
    [astar3,ier3,count3] = bisection(f3,a2,b2,tol2)
    print('the approximate root is',astar3)
    print('number of iterations: ', count3)
    print('the error message reads:',ier3)
    print('f(astar3) =', f3(astar3))

    print("---------- Question 3b Output ----------")
    f4 = lambda x: x**3+x-4
    a4 = 1
    b4 = 4

    tol4 = 1e-3

    [astar4,ier4,count4] = bisection(f4,a4,b4,tol4)
    print('the approximate root is',astar4)
    print('number of iterations: ', count4)
    print('the error message reads:',ier4)
    print('f(astar3) =', f4(astar4))

    print("---------- Question 5 Output ----------")
    plotting()

# bisection method
def bisection(f,a,b,tol):
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
    if (fa*fb>0):
       ier = 1
       astar = a
       count = 0
       return [astar, ier, count]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      count = 0
      return [astar, ier, count]

    if (fb ==0):
      astar = b
      ier = 0
      count = 0
      return [astar, ier, count]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        count = 1
        return [astar, ier, count]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier, count]

def plotting():
    print("---------- Question 5 Output ----------")

    def f(x):
        return x - 4 * np.sin(2 * x) - 3

    x = np.linspace(-10, 10, 1000)  # Range from -10 to 10

    y = f(x)

    plt.plot(x, y, label="f(x) = x - 4/sin(2x) - 3", color="blue")
    plt.axhline(0, color="red", linestyle="--", label="Zero Line")

    plt.title("Plot of f(x) = x - 4/sin(2x) - 3", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)

    plt.show()
      

driver()               

