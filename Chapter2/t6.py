from t0 import *

def other_test(x1, x2):
    w1, w2, theta = 0.7, -0.3, 0.2  
    y = w1 * x1 + w2 * x2
    if y > theta:
        return 1
    else:
        return 0

dis_results(other_test)