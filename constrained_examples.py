# Functions for examples to verify code written for unconstrained optimization


import numpy as np
from math import log
# from unconstrained import *


# Signum function written since Python does not have an inbuilt funciton for this
def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


# Cost function for Ex1 of Part2
# The cost function is written in the Exterior Penalty Function form to minimize
# To be minimized using Penalty Function Method
def vc1(x, sigma=0):
    x1, x2 = x[0], x[1]
    return abs(x1-2) + abs(x2-2) + sigma*(max(0, x2**2-x1) + ((x1**2 + x2**2 - 1)**2))


# Gradient of the above function.
def diff_vc1(x, sigma=0):
    x1, x2 = x[0], x[1]
    diff1 = sign(x1 - 2) + sigma*(4*x1*(x1**2 + x2**2 - 1))
    diff2 = sign(x2 - 2) + sigma*(4*x2*(x1**2 + x2**2 - 1))
    if x1 - x2**2 < 0:
        diff1 = diff1 - sigma*1
        diff2 = diff2 + sigma*2*x2
    return np.array([diff1, diff2])


# Cost function for Ex2 of Part2
# Written in the form of Barrier Penalty Function
# To be minimized using Barrier Function Method
# Barrier Function is negative logarithm
def vc2(x, r=0):
    x1, x2 = x[0], x[1]
    cost = -x1*x2 - r*(log(1 - x1**2 - x2**2) + log(x1 + x2))
    return cost


# Gradient of the above function
def diff_vc2(x, r=0):
    x1, x2 = x[0], x[1]
    diff1 = -x2 - r*(-2*x2/(1 - x1**2 - x2**2) + 1/(x1 + x2))
    diff2 = -x1 - r*(-2*x1/(1 - x1**2 - x2**2) + 1/(x1 + x2))
    return np.array([diff1, diff2])


# Cost function for Ex3 of Part2
# The cost function is written in the Exterior Penalty Function form to minimize
# To be minimized using Penalty Function Method
def vc3(x, sigma=0):
    x1, x2 = x[0], x[1]
    return log(x1) - x2 + sigma*(max(0, 1 - x1) + ((x1**2 + x2**2 - 4)**2))


# Gradient of the above function
def diff_vc3(x, sigma=0):
    x1, x2 = x[0], x[1]
    diff1 = 1/x1 + sigma*(2*x1*(x1**2 + x2**2 - 4))
    diff2 = -1 + sigma*(2*x2*(x1**2 + x2**2 - 4))
    if x1 < 1:
        diff1 = diff1 - sigma
    return np.array([diff1, diff2])


if __name__ == '__main__':
    print("Main of Constrained Examples is Running!")
