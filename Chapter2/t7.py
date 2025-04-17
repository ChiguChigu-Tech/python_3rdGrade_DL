from t0 import *
from t2 import and_test
from t4 import or_test
from t5 import nand_test

def xor_test(x1, x2):
    if or_test(x1, x2) > 0:
        s1 = 1
    else:
        s1 = 0
        
        
    if nand_test(x1, x2) > 0:
        s2 = 1
    else:
        s2 = 0

    ans = and_test(s1, s2)
    
    
    if ans > 0:
        return 1
    else:
        return 0


dis_results(xor_test)
