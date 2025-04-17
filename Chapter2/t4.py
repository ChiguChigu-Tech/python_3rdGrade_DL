from t0 import *

def or_test(x1, x2):
    w1, w2, theta = 1, 1, 0.5
    y = w1 * x1 + w2 * x2
    if y > theta:
        return 1
    else:
        return 0
    
dis_results(or_test)