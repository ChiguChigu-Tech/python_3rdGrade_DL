from t0 import x1, x2, dis_results

def test(x1, x2):
    if x1 == 1 and x2 == 1:
        return 1
    else:
        return 0


dis_results(test)
