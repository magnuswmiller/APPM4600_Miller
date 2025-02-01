# Code for Homework 02

import math
import numpy as np
import matplotlib.pyplot as plt

# Code for Question 3
x = 9.999999995000000e-10

print(math.e**x-1)

y = math.e**x
z = y-1
print(z)

# Code for Question 4a
t = np.arange(0, np.pi, np.pi/30)
y = np.cos(t)

s = np.sum(t*y)
print("The sum is: %16f"%(s))

# Code for Question 4b
# Part 1:
theta = np.linspace(0, 2 * np.pi, 500)
R = 1.2
delta_r = 0.1
f = 15
p = 0

x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)

plt.figure()
plt.plot(x, y)
plt.axis("equal")
plt.title("Wavy Circle - Single Curve")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()

# Part 2:
plt.figure()
for i in range(10):
    delta_r = 0.05
    f = 2 + i
    p = np.random.uniform(0, 2 * np.pi)
    x = R * (1 + delta_r * np.sin(f * theta + p)) * np.cos(theta)
    y = R * (1 + delta_r * np.sin(f * theta + p)) * np.sin(theta)
    plt.plot(x, y, label=f"Curve {i+1}")

plt.axis("equal")
plt.title("Wavy Circles - Multiple Curves")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

