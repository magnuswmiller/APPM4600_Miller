# Code For Lab 10 - APPM 4600

# installing libraries
import numpy as np
import matplotlib.pyplot as plt

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

# Driver Function
def driver():
    poly_vector = eval_legendre(4,2.0)
    print(poly_vector)
    return -1

driver()
