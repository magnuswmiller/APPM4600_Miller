# Lab 01 Code:

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

# Part 3.1.3 Code: Plotting
X = np.linspace(0, 2 * np.pi, 100)
Ya = np.sin(X)
Yb = np.cos(X)

plt.plot(X, Ya)
plt.plot(X, Yb)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Part 3.2 Code: Exercises (The Basics)
# Question 1:
x = np.linspace(0, 9, 10)
y = np.arange(0, 10, 1)
print(x)
print(y)

# Question 2:
print(x[:3])
print(y[:3])

# Question 3:
print("The first three elements of x are: ", x[:3])

# Question 4:
w = 10**(-np.linspace(1,10,10))
x1 = np.arange(0, len(w), 1)
print(w)
plt.semilogy(x,w)

# Question 5:
s = 3*w
plt.semilogy(x,s)
plt.show()
