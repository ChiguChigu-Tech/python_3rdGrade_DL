from t0 import *


def and_test(x1, x2):
    w1, w2, theta = 0.5, 0.5 ,0.8
    ans = w1*x1 + w2*x2
    
    if ans > theta:
        return 1
    else:
        return 0

dis_results(and_test)