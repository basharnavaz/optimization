# Code by Basharnavaz Khan
# Project submission for ECSE 507
# Optimization codes

import numpy as np
from examples import *


# Steepest Descent Algorithm
def steepest_descent(cost, diff_cost, x0, w=0.01, report_print=False):
    x, i = x0, 0
    while np.linalg.norm(diff_cost(x)) > 10**-6:
        i = i + 1
        x = x - w*diff_cost(x)
        if i > 10000:
            break
    if report_print:
        print("Iterations: ", i, ",   Grad norm = ", np.linalg.norm(diff_cost(x)))
    return x


# Armijo Step size rule selection
def armijo_step(cost, diff_cost, x0, gamma=1.2, mu=0.8):
    for _ in range(10):
        v, diff_v, x = cost, diff_cost, x0
        # Calculate the step size
        p, q = 0, 0
        # print("1)  ", v(x - diff_v(x) * (gamma ** p)), " <<<   >>>>", vbar((gamma ** p), v, diff_v, x))
        # Search direction is set as negative gradient
        s = - diff_v(x)
        while v(x - diff_v(x)*(gamma**p)) < vbar((gamma**p), v, diff_v, x, s):
            p = p + 1
        # print("2)  ", v(x - diff_v(x)*(mu**q*gamma**p)), " <<<   >>>>", vbar((mu**q*gamma**p), v, diff_v, x, s))
        while v(x - diff_v(x)*(mu**q*gamma**p)) > vbar((mu**q*gamma**p), v, diff_v, x, s):
            q = q + 1
        # print("p: ", p, "q: ", q)
        w = mu**q*gamma**p

        # Set the new xj
        x = x - w*diff_v(x)
        print(v(x))


# Conjugate Gradient Algorithm
def conjugate_gradient(cost, diff_cost, x0, gamma=1.2, mu=0.8):
    x, v, diff_v = x0, cost, diff_cost
    s = - diff_v(x)
    print(v(x))
    for _ in range(10):
        if np.linalg.norm(diff_v(x)) < 10**-4:
            return
        # Calculate optimal step size using Armijo step size rule
        p, q = 0, 0
        while v(x - diff_v(x) * (gamma ** p)) < vbar((gamma ** p), v, diff_v, x, s):
            p = p + 1
        while v(x - diff_v(x) * (mu ** q * gamma ** p)) > vbar((mu ** q * gamma ** p), v, diff_v, x, s):
            q = q + 1
        w = mu ** q * gamma ** p
        x_next = x + w*s
        beta = np.dot((diff_v(x_next) - diff_v(x)), diff_v(x_next))/np.linalg.norm(diff_v(x))**2
        s = - diff_v(x_next) + beta*s
        x = x_next
        print(v(x), "   ", np.linalg.norm(diff_v(x)))


# Secant Algorithm
def secant(cost, diff_cost, x0, H, gamma=1.2, mu=0.8, report_print=False):
    x = x0
    v, diff_v = cost, diff_cost
    print(v(x))
    for i in range(100):
        # Search direction
        s = - np.dot(H, diff_v(x))
        # Calculate the step size
        # Taken from Armijo backstep rule
        p, q = 0, 0
        # print("1)  ", v(x - diff_v(x) * (gamma ** p)), " <<<   >>>>", vbar((gamma ** p), v, diff_v, x, s))
        while v(x - diff_v(x) * (gamma ** p)) < vbar((gamma ** p), v, diff_v, x, s):
            p = p + 1
        # print("2)  ", v(x - diff_v(x) * (mu ** q * gamma ** p)), " <<<   >>>>",
        #       vbar((mu ** q * gamma ** p), v, diff_v, x, s))
        while v(x - diff_v(x) * (mu ** q * gamma ** p)) > vbar((mu ** q * gamma ** p), v, diff_v, x, s):
            q = q + 1
        # print("p: ", p, "q: ", q)
        w = mu ** q * gamma ** p

        # Choose new H
        delta_g = diff_v(x + w*s) - diff_v(x)
        if np.linalg.norm(delta_g) < 10**-9:
            if report_print:
                print("Iteration:", i, "Gradient norm:", np.linalg.norm(diff_cost(x)))
            return x
        delta_x = w*s
        tt = np.dot(H, delta_g)
        H = H + (np.outer(delta_x, delta_x)/np.dot(delta_x, delta_g)) - (np.outer(tt, tt))/np.dot(tt, delta_g)
        x = x + w*s
        print(v(x))


if __name__ == "__main__":
    
    x2 = np.ones(2)
    print("Steepest Descent: ")
    opt_1 = steepest_descent(cost=v2, diff_cost=diff_v2, x0=x2, w=0.01, report_print=True)
    print("Minimum = ", v2(opt_1))
    # print("Conjugate: ")
    # conjugate_gradient(cost=v2, diff_cost=diff_v2, x0=x2)
    print("Secant Algo: ")
    opt_2 = secant(cost=v2, diff_cost=diff_v2, x0=x2, H=np.identity(2), report_print=True)
    print("Minimum = ", v2(opt_2))
    print("Steepest Again")
    opt_3 = steepest_descent(cost=v2, diff_cost=diff_v2, x0=opt_2, report_print=True)
    print("Minimum = ", v2(opt_3))
    # print("Armijo Rule: ")
    # armijo_step(cost=v2, diff_cost=diff_v2, x0=x2, gamma=1.2)

