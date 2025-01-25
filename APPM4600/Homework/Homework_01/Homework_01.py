# APPM 4600 - Homework
# Code for Homework 01
# Written by Magnus Miller

# Code Layout and Information:
# - Code uses main function as driver
# - Each homework question that requires coding will have a corresponding
#   subroutine
# - Subroutines are called by the main function

# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

# Helper function for Question 5 part b
def cos_diff(x, delta):
    return np.cos(x + delta) - np.cos(x)

# helper function for Question 5 part b
def rewritten_diff(x, delta):
    return -2 * np.sin(delta / 2) * np.sin((2*x + delta) / 2)

def question1():
    # defining x-values
    x_val = np.arange(1.920, 2.080, 0.001)

    # defining coefficients of polynomial
    coef = [1,-18,144,-672,2016,-4032,5376,-4608,2304,-512]

    # calculate using coefs
    p_coef = np.polyval(coef, x_val)

    # calculate using expression
    p_expression = (x_val - 2) ** 9

    # plot p_coef
    plt.plot(x_val, p_coef, label="Plot of (x-2)^9 using coefficients")
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title("Plot of p(x) = (x-2)^9")
    plt.show()

    # plot p_expression
    plt.plot(x_val, p_expression, label="Plot of (x-2)^9 using formula")
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title("Plot of p(x) = (x-2)^9")
    plt.show()

    # plot both overlayed
    plt.plot(x_val, p_coef, label="Using Coefficients")
    plt.plot(x_val, p_expression, label="Using Expression")
    plt.xlabel("$x$")
    plt.ylabel("$p(x)$")
    plt.title("Plot of p(x) = (x-2)^9")
    plt.show()
    return -1

def question5():
    # Define values for x1, x2, and their errors
    x1_small = 0.005
    x2_small = 0.000999999
    x1_large = 1000000.0
    x2_large = 9999999.0
    delta_x1 = 1e-7
    delta_x2 = -1e-7

    # Exact and approximate values
    y_small = x1_small - x2_small
    y_large = x1_large - x2_large
    delta_y = delta_x1 - delta_x2
    tilde_y_small = y_small + delta_y
    tilde_y_large = y_large + delta_y

    # Errors
    absolute_error_small = abs(delta_y)
    relative_error_small = abs(delta_y / y_small)
    absolute_error_large= abs(delta_y)
    relative_error_large= abs(delta_y / y_large)

    print(f"Absolute Error Small: {absolute_error_small}")
    print(f"Relative Error Small: {relative_error_small}")
    print(f"Absolute Error Large: {absolute_error_large}")
    print(f"Relative Error Large: {relative_error_large}")
    
    # Values for x and delta
    x_values = [np.pi, 10**6]
    delta_values = np.logspace(-16, 0, 100)  # Logarithmic spacing for delta

    # Plot the results
    for x in x_values:
        diff = [cos_diff(x, delta) for delta in delta_values]
        rewritten = [rewritten_diff(x, delta) for delta in delta_values]
        difference = np.subtract(diff, rewritten)
        
        plt.figure(figsize=(8, 6))
        plt.plot(delta_values, difference, label='cos(x + delta) - cos(x)')
#        plt.plot(delta_values, rewritten, label='Rewritten Expression')
        plt.xscale('log')
        plt.xlabel('Delta (log scale)')
        plt.ylabel('Difference')
        plt.title(f'Comparison for x = {x}')
        plt.legend()
        plt.grid(True)
        plt.show()
    return -1


# Driver that calls subroutines
def main():
    print("Output for Question #1:")
    question1()

    print("Output for Question #5:")
    question5()

    return -1

# Automatically calls the main function
if __name__ == '__main__':
    main()
