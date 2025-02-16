'''
Code for APPM 4600 Homework 04 due 02/14/25
Written and Completed by Magnus Miller
'''

# Importing Libraries
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf, erfinv

#--------------- Questions Code ---------------
#----- Fxns for Question 1 -----
def errorInv(Ts,Ti,Alpha,Tsec):
    z = erfinv(-Ts/(-Ts+Ti))
    x = z * 2 * np.sqrt(Alpha * Tsec)
    return x

# Function f(x) based on the given equation
def f(x):
    return erf(x / (2 * np.sqrt(0.138e-6 * (60*24*60*60)))) - ((15 - 0) / (20 - (-15)))

# Derivative f'(x)
def df(x):
    return (2 / np.sqrt(np.pi)) * np.exp(-(x / (2 * np.sqrt(0.138e-6 * (60*24*60*60))))**2) * (1 / (2 * np.sqrt(0.138e-6 * (60*24*60*60))))

# Plot
def plot1():
    # Plot the function to visualize the root-finding problem
    x_values = np.linspace(0, 6, 300)
    y_values = [f(x) for x in x_values]

    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, label=r"$f(x) = \text{erf}(\frac{x}{2\sqrt{\alpha t}}) - \frac{T_s - 0}{T_i - T_s}$")
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xlabel("Depth (meters)")
    plt.ylabel("f(x)")
    plt.title("Root Finding for Freezing Depth")
    plt.legend()
    plt.grid()
    plt.show()
    return -1

# Bisection method function
def bisection_method(f, a, b, tol=1e-13, max_iter=100):
    if f(a) * f(b) > 0:
        raise ValueError("Bisection method requires f(a) and f(b) to have opposite signs.")

    for _ in range(max_iter):
        c = (a + b) / 2  # Midpoint
        fc = f(c)

        if abs(fc) < tol or (b - a) / 2 < tol:
            return c  # Converged

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    raise ValueError("Bisection method did not converge")


# Newton's method function
def newton_method(f, df, x0, tol=1e-13, nmax=100):
    x = x0
    for _ in range(nmax):
        fx = f(x)
        dfx = df(x)
        if abs(fx) < tol:
            return x  # Converged
        x -= fx / dfx  # Newton-Raphson step
    raise ValueError("Newton's method did not converge")

#----- Fxns for Question 4 -----
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x), its derivative f'(x), and f''(x)
def f2(x):
    return np.exp(3*x) - 27*x**6 + 27*x**4*np.exp(-x) - 9*x**2*np.exp(2*x)

def df2(x):
    return 3*np.exp(3*x) - 162*x**5 + 108*x**3*np.exp(-x) - 18*x*np.exp(2*x)

def ddf2(x):
    return 9*np.exp(3*x) - 810*x**4 + 324*x**2*np.exp(-x) - 18*np.exp(2*x) - 18*x*np.exp(2*x)

# Standard Newton's Method
def newton_method2(x0, tol=1e-10, max_iter=100):
    iterates = [x0]
    for _ in range(int(max_iter)):
        if abs(f2(x0)) < tol:
            break
        x0 = x0 - f2(x0) / df2(x0)
        iterates.append(x0)
    return x0, iterates

# Modified Newton's Method for multiplicity m
def modified_newton(x0, m, tol=1e-10, max_iter=100):
    iterates = [x0]
    for _ in range(max_iter):
        if abs(f2(x0)) < tol:
            break
        x0 = x0 - m * f2(x0) / df2(x0)
        iterates.append(x0)
    return x0, iterates

# Newton’s Method Applied to g(x) = f(x)/f'(x)
def newton_g_method(x0, tol=1e-10, max_iter=100):
    iterates = [x0]
    for _ in range(max_iter):
        g_x = f2(x0) / df2(x0)
        g_prime_x = (df2(x0)*df2(x0) - f2(x0)*ddf2(x0)) / (df2(x0)**2)
        if abs(g_x) < tol:
            break
        x0 = x0 - g_x / g_prime_x
        iterates.append(x0)
    return x0, iterates

# Function to calculate order of convergence p
def order_of_convergence(iterates):
    errors = [abs(x - iterates[-1]) for x in iterates[:-1]]
    p_values = []
    for k in range(1, len(errors) - 1):
        p = np.log(errors[k+1] / errors[k]) / np.log(errors[k] / errors[k-1])
        p_values.append(p)
    return np.mean(p_values)
    '''
    alpha = []
    for i in range(len(iterates)-1):
        if (abs((iterates[i]-iterates[i-1])/(iterates[i-1]-iterates[i-2]))) != 1:
            a = (math.log(abs((iterates[i+1]-iterates[i])/(iterates[i]-iterates[i-1]))))/(math.log(abs((iterates[i]-iterates[i-1])/(iterates[i-1]-iterates[i-2]))))
            alpha.append(a)
    return alpha[len(alpha)-1]
    '''

#----- Fxns for Question 5 -----
def f3(x):
    return x**6 - x - 1

def df3(x):
    return 6*x**5 - 1

def newton_method3(x0, tol=1e-10, max_iter=50):
    iterates = [x0]
    errors = []
    
    for _ in range(max_iter):
        x1 = x0 - f3(x0) / df3(x0)
        iterates.append(x1)
        error = abs(x1 - x0)
        errors.append(error)
        
        if error < tol:
            break
        
        x0 = x1
    
    return x1, iterates, errors

'''
def secant_method(x0, x1, tol=1e-10, max_iter=50):
    iterates = [x0, x1]
    errors = []
    
    for _ in range(max_iter):
        f_x0, f_x1 = f3(x0), f3(x1)
        
        # Compute next iterate
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        iterates.append(x2)
        error = abs(x2 - x1)
        errors.append(error)
        
        if error < tol:
            break
        
        x0, x1 = x1, x2
    
    return x2, iterates, errors
'''
def secant_method(x0,x1,tol,nmax):
    xnm=x0; xn=x1;
    rn=np.array([x1]);
    # function evaluations
    fn=f3(xn); fnm=f3(xnm);
    msec = (fn-fnm)/(xn-xnm);
    nfun=2; #evaluation counter nfun
    dtol=1e-10; #tolerance for derivative (being near 0)

    if np.abs(msec)<dtol:
        #If slope of secant is too small, secant will fail. Error message is
        print("fail")
        #displayed and code terminates.
    else:
        n=0;

        while n<=nmax:
            pn = - fn/msec; #Secant step
            if np.abs(pn)<tol or np.abs(fn)<2e-15:
                break;

            #Update guess adding Newton step, update xn-1
            xnm = xn; #xn-1 is now xn
            xn = xn + pn; #xn is now xn+pn

            # Update info and loop
            n+=1;
            rn=np.append(rn,xn);
            fnm = fn; #Note we can re-use this function evaluation
            fn=f(xn); #So, only one extra evaluation is needed per iteration
            msec = (fn-fnm)/(xn-xnm); # New slope of secant line
            nfun+=1;

        r=xn;

        if n>=nmax:
            print("Secant method failed to converge, niter=%d, nfun=%d, f(r)=%1.1e\n'" %(n,nfun,np.abs(fn)));
        else:
            print("Secant method converged succesfully, niter=%d, nfun=%d, f(r)=%1.1e" %(n,nfun,np.abs(fn)));

    return (r,rn,nfun)

# Solve for x using Newton's method
def driver():
    # Constants
    Ti = 20  # Initial temperature in Celsius
    Ts = -15  # Surface temperature in Celsius
    Alpha = 0.138e-6  # Thermal diffusivity in m^2/s
    Tsec = 60 * 24 * 60 * 60  # Time in seconds (60 days)


    # 1a
    print(errorInv(Ts,Ti,Alpha,Tsec))

    plot1()

    # 1b
    a, b = 0, 6
    x_solution_bisect = bisection_method(f, a, b)
    print(f"Depth at which freezing occurs (Bisection Method): {x_solution_bisect:.16f} meters")

    # 1c
    x0 = 0.01
    x_solution = newton_method(f, df, x0)
    print(f"Depth at which freezing occurs (Newton's Method): {x_solution:.16f} meters")


    # 4a
    x00 = 4.5
    root_newton, iterates_newton = newton_method2(x0)
    p_newton = order_of_convergence(iterates_newton)
    print(f"Newton's Method Root: {root_newton}, Order of Convergence: {p_newton:.4f}")

    # 4b
    root_modified, iterates_modified = modified_newton(x0, m=2)  # Assuming multiplicity m=2
    p_modified = order_of_convergence(iterates_modified)
    print(f"Modified Newton's Method Root: {root_modified}, Order of Convergence: {p_modified:.4f}")

    # 4c
    root_g, iterates_g = newton_g_method(x0)
    p_g = order_of_convergence(iterates_g)
    print(f"Newton's Method on g(x) Root: {root_g}, Order of Convergence: {p_g:.4f}")

    # 5
    x_0newt = 2.0
    x_0sec = 1.0

    root_newton, iterates_newton, errors_newton = newton_method3(x_0newt)
    root_secant, iterates_secant, errors_secant = secant_method(2, x_0sec,1e-13,100)

    iterates_newton = np.array(iterates_newton)
    iterates_secant = np.array(iterates_secant)

    from scipy.optimize import fsolve
    actual_root = fsolve(f, 2)[0]  # Largest root near 2

    errors_newton = np.abs(iterates_newton - actual_root)
    errors_secant = np.abs(iterates_secant - actual_root)

    print(actual_root)
    print("Iteration | Newton's Error | Secant's Error")
    print("-" * 40)
    for i in range(min(len(errors_newton), len(errors_secant))):
        print(f"{i+1:<9} | {errors_newton[i]:.2e} | {errors_secant[i]:.2e}")

    plt.figure(figsize=(8,6))
    plt.loglog(errors_newton[:-1], errors_newton[1:], label="Newton's Method", marker='o')
    plt.loglog(errors_secant[:-1], errors_secant[1:], label="Secant Method", marker='s')
    plt.xlabel(r"$|x_k - \alpha|$")
    plt.ylabel(r"$|x_{k+1} - \alpha|$")
    plt.title("Convergence Order Estimation")
    plt.legend()
    plt.grid()
    plt.show()

    return -1

driver()

