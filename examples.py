# Examples functions

import numpy as np
from math import sqrt, log, cos, sin

def v1(x):
    b = np.array([1, 4, 5, 4, 2, 1])
    c = np.array([[9, 1, 7, 4, 5, 7],
                  [1, 11, 4, 2, 7, 5],
                  [7, 4, 13, 5, 0, 7],
                  [5, 2, 5, 17, 1, 9],
                  [4, 7, 0, 1, 21, 15],
                  [7, 5, 7, 9, 15, 27]])
    a = 5

    cost = a + np.dot(b, x) + np.dot(np.dot(x, c), x)
    return cost


def diff_v1(x):
    b = np.array([1, 4, 5, 4, 2, 1])
    c = np.array([[9, 1, 7, 4, 5, 7],
                  [1, 11, 4, 2, 7, 5],
                  [7, 4, 13, 5, 0, 7],
                  [5, 2, 5, 17, 1, 9],
                  [4, 7, 0, 1, 21, 15],
                  [7, 5, 7, 9, 15, 27]])
    diff = b + 2*np.dot(c, x)
    return diff


def v2(x):
    x1, x2 = x[0], x[1]
    a = sqrt(x1**2 + 1)
    b = sqrt(2*x2**2 + 1)
    c = x1**2 + x2**2 + 0.5
    cost = -a*b/c
    return cost


def diff_v2(x):
    x1, x2 = x[0], x[1]
    a = sqrt(x1**2 + 1)
    b = sqrt(2*x2**2 + 1)
    c = x1**2 + x2**2 + 0.5

    diff1 = b*c/a - 2*a*b*x1
    diff2 = 2*a*c/b - 2*a*b*x2
    diff = - np.array([diff1, diff2])/(c**2)
    return diff


def v3(x):
    x1, x2 = x[0], x[1]
    quad = 1 + x1 + 2*x2 + (6*x1**2 + 3*x1*x2 + 5*x2**2)
    log_part = 10*(log(1+x1**4)*sin(100*x1) + log(1+x2**4)*cos(100*x2**2))
    cost = quad + log_part
    return cost


def diff_v3(x):
    x1, x2 = x[0], x[1]
    diff1 = (1 + 12*x1 + 3*x2) + \
            10*(4*x1**3*sin(100*x1)/(1+x1**4) +
                log(1+x1**4)*cos(100*x1)*100)
    diff2 = (2 + 3*x1 + 10*x2) + \
            10*(4*x2*cos(100*x2)/(1+x2**4) -
                log(1+x2**4)*sin(100*x2)*100)
    diff = np.array([diff1, diff2])
    return diff


def vbar(w, v, diffv, x, s):
    # For the Armijo step selection.
    # Descent direction is assumed to be negative gradient
    # Expression taken from Page 5.4 of Handwritten.pdf
    # add another parameter and change the expression below
    # to have a different descent direction.
    ans = v(x) + 0.5*w*np.dot(diffv(x), s)
    return ans


if __name__ == "__main__":
    b = np.array([1, 4, 5, 4, 2, 1])
    c = np.array([[9, 1, 7, 4, 5, 7],
                  [1, 11, 4, 2, 7, 5],
                  [7, 4, 13, 5, 0, 7],
                  [5, 2, 5, 17, 1, 9],
                  [4, 7, 0, 1, 21, 15],
                  [7, 5, 7, 9, 15, 27]])
    x_opt = -0.5*np.dot(np.linalg.inv(c), b)
    # print(diff_v1(x_opt))
    # print(v1(x_opt))

    a = np.array([0, 3])
    v2(a)





