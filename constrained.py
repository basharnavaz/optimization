import numpy as np
from constrained_examples import *


# Steepest Descent Algorithm for Constrained Optimization
# The code is the same as the one in the Unconstrained Optimization Algos
# This function takes the extra parameter sigma for Barrier and Penalty function evaluations
def steepest_constrained(cost, diff_cost, x0, w=0.01, sigma=0.0, report_print=False):
    x, i = x0, 0
    while np.linalg.norm(diff_cost(x, sigma)) > 10**-6:
        i = i + 1
        x = x - w*diff_cost(x, sigma)
        if i > 10000:  # Break if exceeds number of iteration limit
            break
    if report_print:
        print("Iterations: ", i, ",   Grad norm = ", np.linalg.norm(diff_cost(x)))
    return x


if __name__ == '__main__':
    print("Something")
    # # Penalty Function Method for Ex1 of Part2
    # print("Penalty Function Method for Ex1 of Part2")
    # xc1 = np.array([1, 0])
    # for i in range(10):
    #     xc1 = steepest_constrained(cost=vc1, diff_cost=diff_vc1, x0=xc1, sigma=2*i)
    #     print(i, xc1, "Cost:", vc1(xc1), " Const1:", (xc1[0]-xc1[1]**2), "Const2:", (xc1[0]**2+xc1[1]**2))

    # # Barrier Point Method for Ex2 of Part 2
    # print("Penalty Function Method for Ex2 of Part2")
    # xc2 = np.array([0.5, 0.5])
    # for i in range(9):
    #     xc2 = steepest_constrained(cost=vc2, diff_cost=diff_vc2, sigma=0.5**i, x0=xc2)
    #     print(i, ") ", xc2, "Cost:", vc2(xc2), " Const1:", (xc2[0] + xc2[1]), "Const2:", (xc2[0]**2 + xc2[1]**2))

    # # Penalty Function Method for Ex2 of Part2
    # print("Penalty Function Method for Ex3 of Part2")
    # xc3 = np.array([2, 0])
    # for i in range(7):
    #     xc3 = steepest_constrained(cost=vc3, diff_cost=diff_vc3, x0=xc3, sigma=7)
    #     print(i, xc3, "Cost:", vc3(xc3), " Const1:", (xc3[0]), "Const2:", (xc3[0]**2+xc3[1]**2))


