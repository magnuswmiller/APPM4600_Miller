# Lab 01 Code
# Part 4: Practical Code Design
# Importing Libraries
import numpy as np
import numpy.linalg as la
import math

'''
driver()
Param:
Return:
This function serves as the driver for the dot product subroutine
'''
def driver():
    n = 100
    x = np.linspace(0,np.pi,n)
    # this is a function handle. You can use it to define
    # functions instead of using a subroutine like you
    # have to in a true low level language.
    f = lambda x: x**2 + 4*x + 2*np.exp(x)
    g = lambda x: 6*x**3 + 2*np.sin(x)
    y = f(x)
    w = g(x)
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product is : ', dp)
    return

'''
driver1()
Param:
Return:
This function serves as the driver for the dot product subroutine
'''
def driver1():
    n = 3
    y = [1,0,0]
    w = [0,1,0]
    # evaluate the dot product of y and w
    dp = dotProduct(y,w,n)
    # print the output
    print('the dot product is : ', dp)
    return

'''
driver2()
Param:
Return:
This function serves as the driver for the dot product subroutine for matrix
multiplication.
'''
def driver2():
    small_matrix = np.array([[1,2],[3,4]])
    small_vector = np.array([5,6])

    large_matrix = np.random.randint(0,10, (100,100))
    large_vector = np.random.randint(0,10, 100)
    
    dps = dotProduct(small_matrix, small_vector, 2)
    print('the matrix multiplication is: ', dps)

    dpl = dotProduct(large_matrix, large_vector, 100)
    print('the matrix multiplication is: ', dpl)
    return

'''
dotProduct()
Param:
    x: np array of 100 equally spaced entries between 0 and 2pi
    y: array of integer values
    n: integer value representing the length of the vectors x, y
This function calculates the dot product of two vectors x and y.
'''
def dotProduct(x,y,n):
    # Computes the dot product of the n x 1 vectors x and y
    dp = 0.
    for j in range(n):
        dp = dp + x[j]*y[j]
    return dp

# Call to driver function.
driver()
driver1()
driver2()
