import numpy as np
import matplotlib.pyplot as plt
import random
import time
import gc

# Gradient descent algorithm
def gradient_descent(f, df, x0, alpha=0.1, epsilon=1e-10, max_iter=1000000):
    """
    Gradient descent algorithm to find minimum of a function f.

    Args:
        f: a function to minimize.
        df: the derivative of the function f.
        x0: initial value.
        alpha: learning rate.
        epsilon: accuracy stopping criterion.
        max_iter: maximum number of iterations.

    Returns:
        x_min: the value of x that minimizes the function f.
        f_min: the minimum value of the function f.
        x_history: the sequence of x values during optimization (following steps of optimalization).
    """

    x = x0
    iter = 0
    x_history = [x0]
    while iter < max_iter:
        grad = df(x)
        x_next = x - alpha * grad
        different = x_next - x
        if np.linalg.norm(different) < epsilon:
            break
        x = x_next
        x_history.append(x)
        iter += 1
    return x, f(x), np.array(x_history), iter

def show2D(x_min_f, f_min_f, x_history_f):
    x = np.arange(-5.0, 5.0, 0.1)
    y = f(x)
    fig, ax = plt.subplots()
    ax.plot(x, y, label="f(x)")
    ax.plot(x_history_f, f(x_history_f), color='red', marker='o', label="Optimization path")
    ax.scatter(x_min_f, f_min_f, marker='o', s=100, c='green', label="Minimum")
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_title('Gradient Descent Optimization of 2D Function')
    ax.legend()
    plt.show()

def show3D(x_min_g, f_min_g, x_history_g):
    x = y = np.arange(-5.0, 5.0, 0.1)
    X, Y = np.meshgrid(x, y)
    Z = g([X, Y])
    fig = plt.figure(figsize=(8,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='coolwarm')
    ax.plot(x_history_g[:, 0], x_history_g[:, 1], g(x_history_g.T), color='red', marker='o', label="Optimization path")
    ax.scatter(x_min_g[0], x_min_g[1], f_min_g, marker='*', s=200, c='green', label="Minimum")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('g(x,y)')
    ax.set_title('Gradient Descent Optimization of 3D Function')
    ax.legend()
    plt.show()

def random_number(start, stop, step):
    numbers = []
    for i in range(step):
        number = round(random.uniform(start, stop), 2)
        numbers.append(number)
    return numbers


# One-dimensional quadratic function
def f(x):
    return 2*x**2 + 3*x + 1

def df(x):
    return 4*x + 3

def optimize_f (x0):
    # Find the minimum of f
    print(f"FUNKCJA f(x)")
    print(f"Losowy punkt starowy x0 = {x0}")

    gc_old = gc.isenabled()
    gc.disable()
    start = time.time()
    x_min_f, f_min_f, x_history_f,iteraction_f  = gradient_descent(f, df, x0=x0)
    stop = time.time()
    if gc_old:
        gc.enable()
    function_time_f = stop - start

    print(f"Minimum funkcji f: {f_min_f:.4f} dla x = {x_min_f:.4f}")
    print(f"Czas działania algorytmu gradnientu prosteg: {function_time_f}")
    print(f"Liczba wykonanych iteracji: {iteraction_f} \n")

    # Visualize the flow of the algorithm for f
    show2D(x_min_f, f_min_f, x_history_f)

# Two-dimensional quadratic function
def g(x):
    return 1-0.6*np.exp(-x[0]**2-x[1]**2)-0.4*np.exp(-(x[0]+1.75)**2-(x[1]-1)**2)

def dg(x):
    return np.array([1.2*x[0]*np.exp(-x[0]**2-x[1]**2)+0.8*(x[0]+1.75)*np.exp(-(x[0]+1.75)**2-(x[1]-1)**2),
                    1.2*x[1]*np.exp(-x[0]**2-x[1]**2)+0.8*(x[1]-1)*np.exp(-(x[0]+1.75)**2-(x[1]-1)**2)])

def optimize_g(x0,y0):

    # Find the minimum of g
    print(f"FUNKCJA g(x)")
    print(f"Losowy punkt starowy (x0,y0) = {(x0,y0)}")

    gc_old = gc.isenabled()
    gc.disable()
    start = time.time()
    x_y_min_g, f_min_g, x_history_g,iteraction_g  = gradient_descent(g, dg, x0=np.array([x0, y0]))
    stop = time.time()
    if gc_old:
        gc.enable()
    function_time_g = stop - start

    print(f"Minimum funkcji g: {f_min_g:.4f} dla (x,y) = {x_y_min_g}")
    print(f"Czas działania algorytmu gradnientu prosteg: {function_time_g}")
    print(f"Liczba wykonanych iteracji: {iteraction_g}")
    # Visualize the flow of the algorithm for g
    show3D(x_y_min_g, f_min_g, x_history_g)



# Start random value
random_numbers = random_number(-5, 5, 2)
x0 = random_numbers[0]
y0 = random_numbers[1]
optimize_f(x0)
optimize_g(x0,y0)









